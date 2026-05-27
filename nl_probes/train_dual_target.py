"""
Phased dual-target oracle training (single ``torchrun`` job).

Calls ``train_model`` from ``nl_probes/sft.py`` four times on the same ``wandb_suffix`` / ``save_dir``:
half of target A's mixture, half of target B's, **milestone** marker, then the second halves.

**Projection note (current ``train_model`` API):** the training stack keeps a **single**
``activation_projector`` checkpoint key. That is sufficient when both targets share the same
``hidden_size`` (typical for two “4B-class” Qwen checkpoints): one linear map
``target_hidden -> oracle_hidden`` is shared across phases. If your two targets use **different**
hidden widths, you need an extended ``train_model`` (not covered here) before this script can train
two distinct projectors in one run.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from datetime import timedelta
from typing import Any, List

import torch
import torch.distributed as dist
from transformers import BitsAndBytesConfig

from nl_probes.configs.sft_config import SelfInterpTrainingConfig, namespaced_save_dir
from nl_probes.sft import _ensure_datasets_exist, build_datasets, build_loader_groups, train_model
from nl_probes.utils.checkpointing import read_latest_checkpoint_meta
from nl_probes.utils.common import load_tokenizer
from nl_probes.utils.hf_config import dual_target_auto_gradient_checkpointing, pretrained_hidden_size

# Same table as ``nl_probes/sft.py`` ``__main__`` (classification sizes for ``build_loader_groups``).
_MAIN_TRAIN_SIZE = 6000
_MAIN_TEST_SIZE = 250
CLASSIFICATION_DATASETS: dict[str, dict[str, Any]] = {
    "geometry_of_truth": {
        "num_train": _MAIN_TRAIN_SIZE,
        "num_test": _MAIN_TEST_SIZE,
        "splits": ["train", "test"],
    },
    "relations": {
        "num_train": _MAIN_TRAIN_SIZE,
        "num_test": _MAIN_TEST_SIZE,
        "splits": ["train", "test"],
    },
    "sst2": {
        "num_train": _MAIN_TRAIN_SIZE,
        "num_test": _MAIN_TEST_SIZE,
        "splits": ["train", "test"],
    },
    "md_gender": {
        "num_train": _MAIN_TRAIN_SIZE,
        "num_test": _MAIN_TEST_SIZE,
        "splits": ["train", "test"],
    },
    "snli": {
        "num_train": _MAIN_TRAIN_SIZE,
        "num_test": _MAIN_TEST_SIZE,
        "splits": ["train", "test"],
    },
    "ag_news": {"num_train": _MAIN_TRAIN_SIZE, "num_test": _MAIN_TEST_SIZE, "splits": ["test"]},
    "ner": {
        "num_train": _MAIN_TRAIN_SIZE,
        "num_test": _MAIN_TEST_SIZE,
        "splits": ["train", "test"],
    },
    "tense": {
        "num_train": _MAIN_TRAIN_SIZE,
        "num_test": _MAIN_TEST_SIZE,
        "splits": ["train", "test"],
    },
    "language_identification": {
        "num_train": _MAIN_TRAIN_SIZE,
        "num_test": _MAIN_TEST_SIZE,
        "splits": ["test"],
        "batch_size": 4,
    },
    "singular_plural": {"num_train": 0, "num_test": _MAIN_TEST_SIZE, "splits": ["test"]},
}

MILESTONE_LOG_LINE = (
    "MILESTONE_DUAL_TARGET_HALFWAY "
    "phase=after_first_half_target_A_and_first_half_target_B "
    "search_this_string_in_slurm_out_or_err"
)

DUAL_TARGET_PHASE_LABELS: list[str] = [
    "phase_1a_first_half_target_A",
    "phase_1b_first_half_target_B",
    "phase_2a_second_half_target_A",
    "phase_2b_second_half_target_B",
]

DUAL_TARGET_STATE_FILENAME = "DUAL_TARGET_STATE.json"

# Each half-slice is ~32k optimizer steps on 1 GPU; used when milestone/ckpt metadata lie.
_DUAL_TARGET_STEPS_PER_HALF = 30_000


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--oracle_model_name", type=str, required=True)
    p.add_argument("--target_a", type=str, required=True)
    p.add_argument("--target_b", type=str, required=True)
    p.add_argument("--wandb_suffix", type=str, required=True)
    p.add_argument("--hook_onto_layer", type=int, default=1)
    p.add_argument("--train_batch_size", type=int, default=16)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument(
        "--gradient_checkpointing",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="If omitted: on only when oracle and both targets share the same Qwen3 vs Qwen3.5 lineage.",
    )
    p.add_argument("--eval_steps", type=int, default=10_000)
    p.add_argument("--save_steps", type=int, default=10_000)
    p.add_argument("--layer_percents", type=int, nargs="+", default=[25, 50, 75])
    return p.parse_args()


def _apply_runtime_cfg(
    cfg: SelfInterpTrainingConfig,
    *,
    hook_onto_layer: int,
    train_batch_size: int,
    gradient_accumulation_steps: int,
    gradient_checkpointing: bool,
    eval_steps: int,
    save_steps: int,
    layer_percents: List[int],
    loop_dataset_loaders: list,
) -> None:
    cfg.hook_onto_layer = hook_onto_layer
    cfg.hf_repo_name = "N/A"
    cfg.layer_percents = layer_percents
    cfg.train_batch_size = train_batch_size
    cfg.activation_collection_batch_size = train_batch_size * 4
    cfg.eval_batch_size = train_batch_size * 8
    cfg.eval_steps = eval_steps
    cfg.eval_on_start = True
    cfg.gradient_checkpointing = gradient_checkpointing
    cfg.gradient_accumulation_steps = gradient_accumulation_steps
    cfg.save_steps = save_steps
    cfg.finalize(dataset_loaders=loop_dataset_loaders)


def _dual_target_state_path(save_dir: str) -> str:
    return os.path.join(save_dir, DUAL_TARGET_STATE_FILENAME)


def _read_dual_target_state(save_dir: str) -> dict[str, Any]:
    path = _dual_target_state_path(save_dir)
    if os.path.isfile(path):
        return json.loads(open(path, encoding="utf-8").read())
    return {}


def _write_dual_target_state(save_dir: str, next_phase_index: int, last_completed_phase: str, global_step: int) -> None:
    payload = {
        "next_phase_index": next_phase_index,
        "last_completed_phase": last_completed_phase,
        "global_step": global_step,
        "phase_order": DUAL_TARGET_PHASE_LABELS,
    }
    with open(_dual_target_state_path(save_dir), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _phase_index_from_global_step(global_step: int) -> int:
    """Map cumulative step count to the next phase index (0=1a … 3=2b)."""
    if global_step >= 3 * _DUAL_TARGET_STEPS_PER_HALF:
        return 3
    if global_step >= 2 * _DUAL_TARGET_STEPS_PER_HALF:
        return 2
    if global_step >= _DUAL_TARGET_STEPS_PER_HALF:
        return 1
    return 0


def _first_phase_index_to_run(save_dir: str) -> int:
    """
    Order after global_step ~32075 (end of 1a): 1b (Qwen3.5-4B first half), then 2a, then 2b.
    """
    state = _read_dual_target_state(save_dir)
    if state:
        return int(state["next_phase_index"])

    meta = read_latest_checkpoint_meta(save_dir)
    global_step = int(meta.get("global_step", 0))
    extra = meta.get("extra", {})

    # Milestone is written when phase 1b *finishes*; job 198518 hit eval-only 1b at step 32k
    # and still wrote the marker — require ~64k steps before trusting it.
    milestone_path = os.path.join(save_dir, "MILESTONE_DUAL_TARGET_HALFWAY.txt")
    if os.path.isfile(milestone_path) and global_step >= 2 * _DUAL_TARGET_STEPS_PER_HALF:
        return 2

    step_index = _phase_index_from_global_step(global_step)

    completed = extra.get("dual_target_phase")
    if completed not in DUAL_TARGET_PHASE_LABELS:
        return step_index

    phase_idx = DUAL_TARGET_PHASE_LABELS.index(completed)
    if extra.get("dual_target_phase_complete"):
        meta_index = phase_idx + 1
        # Ignore bogus complete flags when global_step has not advanced (e.g. phase_2b @ 32k).
        return max(step_index, meta_index) if global_step >= (phase_idx + 1) * _DUAL_TARGET_STEPS_PER_HALF else step_index

    phase_start = int(extra.get("dual_target_phase_start_global_step", 0))
    if phase_idx == 0 and global_step > 0 and phase_start == global_step:
        return max(step_index, 1)

    return max(step_index, phase_idx)


def _run_phase(
    *,
    cfg: SelfInterpTrainingConfig,
    target_model_name: str,
    which_half: int,
    load_lora_path: str | None,
    phase_label: str,
    loop_dataset_loaders: list,
    tokenizer,
    device: torch.device,
    dtype: torch.dtype,
    model_kwargs: dict,
    local_rank: int,
    train_batch_size: int,
    hook_onto_layer: int,
    gradient_accumulation_steps: int,
    gradient_checkpointing: bool,
    eval_steps: int,
    save_steps: int,
    layer_percents: List[int],
    phase_index: int,
) -> str:
    cfg.target_model_name = target_model_name
    cfg.load_lora_path = load_lora_path
    _apply_runtime_cfg(
        cfg,
        hook_onto_layer=hook_onto_layer,
        train_batch_size=train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=gradient_checkpointing,
        eval_steps=eval_steps,
        save_steps=save_steps,
        layer_percents=layer_percents,
        loop_dataset_loaders=loop_dataset_loaders,
    )
    if cfg.wandb_run_name:
        cfg.wandb_run_name = f"{cfg.wandb_run_name}_{phase_label}"
    if local_rank == 0:
        print(f"[{phase_label}] save_dir={cfg.save_dir} target={target_model_name} load_lora={load_lora_path}")

    if local_rank == 0:
        _ensure_datasets_exist(loop_dataset_loaders)
    dist.barrier()

    all_training_data, all_eval_data = build_datasets(
        cfg, dataset_loaders=loop_dataset_loaders, window_mult=cfg.window_mult
    )
    n = len(all_training_data)
    mid = n // 2
    train_slice = slice(0, mid) if which_half == 0 else slice(mid, n)
    phase_train = all_training_data[train_slice]
    if local_rank == 0:
        print(f"[{phase_label}] half={which_half} slice={train_slice} of n={n} -> len={len(phase_train)}")
        print(asdict(cfg))

    train_model(
        cfg=cfg,
        training_data=phase_train,
        eval_datasets=all_eval_data,
        tokenizer=tokenizer,
        dtype=dtype,
        device=device,
        model_kwargs=model_kwargs,
        verbose=True,
        save_checkpoint_on_complete=True,
        checkpoint_complete_extra={"dual_target_phase_complete": True},
        dual_target_phase_label=phase_label,
    )

    meta = read_latest_checkpoint_meta(cfg.save_dir)
    global_step = int(meta.get("global_step", 0))
    if local_rank == 0:
        _write_dual_target_state(cfg.save_dir, phase_index + 1, phase_label, global_step)
    dist.barrier()
    return cfg.save_dir


if __name__ == "__main__":
    args = _parse_args()
    if args.gradient_checkpointing is None:
        args.gradient_checkpointing = dual_target_auto_gradient_checkpointing(
            args.oracle_model_name, args.target_a, args.target_b
        )
    wa = pretrained_hidden_size(args.target_a)
    wb = pretrained_hidden_size(args.target_b)
    if wa != wb:
        raise ValueError(
            f"target_a ({args.target_a}) hidden_size={wa} != target_b ({args.target_b}) hidden_size={wb}. "
            "This training entrypoint shares one activation projector across targets; use two targets "
            "with identical hidden_size, or extend nl_probes/sft.py::train_model."
        )

    dist.init_process_group(backend="nccl", timeout=timedelta(hours=2))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    world_size = dist.get_world_size()

    dtype = torch.bfloat16
    device = torch.device(f"cuda:{local_rank}")

    train_batch_size = args.train_batch_size
    assert train_batch_size % world_size == 0, (
        f"Global batch size {train_batch_size} must be divisible by world_size {world_size}"
    )
    train_batch_size = train_batch_size // world_size

    model_kwargs: dict = {"attn_implementation": "sdpa"}
    if args.target_a in ("Qwen/Qwen3-32B", "meta-llama/Llama-3.3-70B-Instruct") or args.target_b in (
        "Qwen/Qwen3-32B",
        "meta-llama/Llama-3.3-70B-Instruct",
    ):
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=dtype,
        )
        model_kwargs = {"quantization_config": bnb_config, "attn_implementation": "sdpa"}

    tokenizer = load_tokenizer(args.oracle_model_name)

    def loop_for(target_name: str):
        lg = build_loader_groups(
            model_name=target_name,
            prompt_model_name=args.oracle_model_name,
            layer_percents=args.layer_percents,
            act_collection_batch_size=train_batch_size,
            save_acts=False,
            classification_datasets=CLASSIFICATION_DATASETS,
            model_kwargs=model_kwargs,
        )
        return lg["latentqa_loaders"] + lg["classification_loaders"] + lg["past_lens_loaders"]

    loop_a = loop_for(args.target_a)
    loop_b = loop_for(args.target_b)

    save_dir = namespaced_save_dir(wandb_suffix=args.wandb_suffix)

    start_phase_idx = _first_phase_index_to_run(save_dir)
    if start_phase_idx >= len(DUAL_TARGET_PHASE_LABELS):
        if local_rank == 0:
            print(f"DUAL_TARGET: all phases complete (index {start_phase_idx}), nothing to train.")
        dist.destroy_process_group()
        raise SystemExit(0)

    if local_rank == 0:
        print(
            f"DUAL_TARGET resume: starting at phase index {start_phase_idx} "
            f"({DUAL_TARGET_PHASE_LABELS[start_phase_idx]}) save_dir={save_dir}"
        )
        if start_phase_idx > 0 and not _read_dual_target_state(save_dir):
            meta = read_latest_checkpoint_meta(save_dir)
            _write_dual_target_state(
                save_dir,
                start_phase_idx,
                DUAL_TARGET_PHASE_LABELS[start_phase_idx - 1],
                int(meta.get("global_step", 0)),
            )
    dist.barrier()

    lora_path: str | None = None
    if start_phase_idx > 0:
        lora_path = f"{save_dir}/final"

    phase_specs: list[dict[str, Any]] = [
        {
            "index": 0,
            "label": "phase_1a_first_half_target_A",
            "target": args.target_a,
            "half": 0,
            "loaders": loop_a,
            "load_lora": None,
        },
        {
            "index": 1,
            "label": "phase_1b_first_half_target_B",
            "target": args.target_b,
            "half": 0,
            "loaders": loop_b,
        },
        {
            "index": 2,
            "label": "phase_2a_second_half_target_A",
            "target": args.target_a,
            "half": 1,
            "loaders": loop_a,
        },
        {
            "index": 3,
            "label": "phase_2b_second_half_target_B",
            "target": args.target_b,
            "half": 1,
            "loaders": loop_b,
        },
    ]

    for spec in phase_specs:
        idx = int(spec["index"])
        if idx < start_phase_idx:
            if local_rank == 0:
                print(f"DUAL_TARGET skip completed phase [{spec['label']}] target={spec['target']}")
            continue

        phase_load_lora = None if idx == 0 and start_phase_idx == 0 else lora_path
        save_dir = _run_phase(
            cfg=SelfInterpTrainingConfig(
                oracle_model_name=args.oracle_model_name,
                target_model_name=spec["target"],
                wandb_suffix=args.wandb_suffix,
            ),
            target_model_name=spec["target"],
            which_half=int(spec["half"]),
            load_lora_path=phase_load_lora,
            phase_label=str(spec["label"]),
            loop_dataset_loaders=spec["loaders"],
            tokenizer=tokenizer,
            device=device,
            dtype=dtype,
            model_kwargs=model_kwargs,
            local_rank=local_rank,
            train_batch_size=train_batch_size,
            hook_onto_layer=args.hook_onto_layer,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            gradient_checkpointing=args.gradient_checkpointing,
            eval_steps=args.eval_steps,
            save_steps=args.save_steps,
            layer_percents=args.layer_percents,
            phase_index=idx,
        )
        lora_path = f"{save_dir}/final"

        if idx == 1:
            if local_rank == 0:
                print("\n" + "=" * 88)
                print(MILESTONE_LOG_LINE)
                print("=" * 88 + "\n", flush=True)
                marker_path = os.path.join(save_dir, "MILESTONE_DUAL_TARGET_HALFWAY.txt")
                with open(marker_path, "w", encoding="utf-8") as f:
                    f.write(MILESTONE_LOG_LINE + "\n")
                    f.write(f"target_a={args.target_a}\n")
                    f.write(f"target_b={args.target_b}\n")
                    f.write(f"oracle={args.oracle_model_name}\n")
            dist.barrier()

    dist.destroy_process_group()
