"""
Sequential dual-target oracle training (single ``torchrun`` job).

Calls ``train_model`` from ``nl_probes/sft.py`` twice on the same ``wandb_suffix`` / ``save_dir``:
**all** of target A, then **all** of target B — no interleaving.

Resume logic (in order of priority):
1. ``SEQ_DUAL_TARGET_STATE.json`` in save_dir  →  authoritative phase pointer.
2. ``save_dir/final`` exists                   →  phase A already done.
3. ``--bootstrap_lora_path`` exists            →  use external checkpoint for A, skip to B.
4. Fallback                                    →  start from scratch (phase A).
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
from nl_probes.utils.common import load_model, load_tokenizer
from nl_probes.utils.hf_config import dual_target_auto_gradient_checkpointing, pretrained_hidden_size
from nl_probes.utils.activation_utils import get_hf_submodule
from nl_probes.utils.eval import run_evaluation, score_eval_responses

# ── dataset config (mirrors sft.py __main__) ────────────────────────────────
_MAIN_TRAIN_SIZE = 6000
_MAIN_TEST_SIZE = 250
CLASSIFICATION_DATASETS: dict[str, dict[str, Any]] = {
    "geometry_of_truth": {"num_train": _MAIN_TRAIN_SIZE, "num_test": _MAIN_TEST_SIZE, "splits": ["train", "test"]},
    "relations":         {"num_train": _MAIN_TRAIN_SIZE, "num_test": _MAIN_TEST_SIZE, "splits": ["train", "test"]},
    "sst2":              {"num_train": _MAIN_TRAIN_SIZE, "num_test": _MAIN_TEST_SIZE, "splits": ["train", "test"]},
    "md_gender":         {"num_train": _MAIN_TRAIN_SIZE, "num_test": _MAIN_TEST_SIZE, "splits": ["train", "test"]},
    "snli":              {"num_train": _MAIN_TRAIN_SIZE, "num_test": _MAIN_TEST_SIZE, "splits": ["train", "test"]},
    "ag_news":           {"num_train": _MAIN_TRAIN_SIZE, "num_test": _MAIN_TEST_SIZE, "splits": ["test"]},
    "ner":               {"num_train": _MAIN_TRAIN_SIZE, "num_test": _MAIN_TEST_SIZE, "splits": ["train", "test"]},
    "tense":             {"num_train": _MAIN_TRAIN_SIZE, "num_test": _MAIN_TEST_SIZE, "splits": ["train", "test"]},
    "language_identification": {"num_train": _MAIN_TRAIN_SIZE, "num_test": _MAIN_TEST_SIZE, "splits": ["test"], "batch_size": 4},
    "singular_plural":   {"num_train": 0,               "num_test": _MAIN_TEST_SIZE, "splits": ["test"]},
}

SEQ_PHASE_LABELS: list[str] = [
    "phase_1_full_target_A",
    "phase_2_full_target_B",
]

STATE_FILENAME = "SEQ_DUAL_TARGET_STATE.json"

# Default bootstrap: existing oracle-on-A checkpoint produced before this run.
_DEFAULT_BOOTSTRAP = (
    "checkpoints_latentqa_cls_past_lens_oracle_Qwen3-1.7B_target_Qwen3-4B"
)


# ── CLI ──────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--oracle_model_name",          type=str, required=True)
    p.add_argument("--target_a",                   type=str, required=True)
    p.add_argument("--target_b",                   type=str, required=True)
    p.add_argument("--wandb_suffix",               type=str, required=True)
    p.add_argument("--hook_onto_layer",            type=int, default=1)
    p.add_argument("--train_batch_size",           type=int, default=16)
    p.add_argument("--gradient_accumulation_steps",type=int, default=1)
    p.add_argument(
        "--gradient_checkpointing",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="If omitted: auto-detected from oracle/target model lineage.",
    )
    p.add_argument("--eval_steps",    type=int, default=10_000)
    p.add_argument("--save_steps",    type=int, default=10_000)
    p.add_argument("--layer_percents",type=int, nargs="+", default=[25, 50, 75])
    p.add_argument(
        "--bootstrap_lora_path",
        type=str,
        default=_DEFAULT_BOOTSTRAP,
        help=(
            "Path to an existing oracle-on-A checkpoint. If ``save_dir/final`` is absent "
            "but this directory contains ``final/``, phase A is skipped and this is used "
            "as the starting LoRA for phase B. Pass empty string to disable."
        ),
    )
    return p.parse_args()


# ── state helpers ─────────────────────────────────────────────────────────────

def _state_path(save_dir: str) -> str:
    return os.path.join(save_dir, STATE_FILENAME)


def _read_state(save_dir: str) -> dict[str, Any]:
    path = _state_path(save_dir)
    if os.path.isfile(path):
        return json.loads(open(path, encoding="utf-8").read())
    return {}


def _write_state(save_dir: str, next_phase_index: int, last_completed_phase: str, global_step: int) -> None:
    payload = {
        "next_phase_index": next_phase_index,
        "last_completed_phase": last_completed_phase,
        "global_step": global_step,
        "phase_order": SEQ_PHASE_LABELS,
    }
    os.makedirs(save_dir, exist_ok=True)
    with open(_state_path(save_dir), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _first_phase_and_lora(save_dir: str, bootstrap_lora_path: str) -> tuple[int, str | None]:
    """
    Returns (start_phase_index, load_lora_path).

    Priority:
      1. State file in save_dir.
      2. save_dir/final exists  →  A done, lora = save_dir/final.
      3. bootstrap_lora_path/final exists  →  A done (external), lora = that path.
      4. Start from scratch.
    """
    state = _read_state(save_dir)
    if state:
        idx = int(state["next_phase_index"])
        if idx > 0:
            own_final = f"{save_dir}/final"
            if os.path.isdir(own_final):
                return idx, own_final
            # State claims phase A is done but save_dir/final is missing — fall through
            # to bootstrap check rather than passing a non-existent lora path.
        else:
            return idx, None

    own_final = os.path.join(save_dir, "final")
    if os.path.isdir(own_final):
        return 1, own_final

    if bootstrap_lora_path:
        bootstrap_final = os.path.join(bootstrap_lora_path, "final")
        if os.path.isdir(bootstrap_final):
            return 1, bootstrap_final

    return 0, None


# ── runtime config ────────────────────────────────────────────────────────────

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


# ── phase runner ──────────────────────────────────────────────────────────────

def _run_phase(
    *,
    cfg: SelfInterpTrainingConfig,
    target_model_name: str,
    load_lora_path: str | None,
    phase_label: str,
    phase_index: int,
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
        _ensure_datasets_exist(loop_dataset_loaders)
    dist.barrier()

    all_training_data, all_eval_data = build_datasets(
        cfg, dataset_loaders=loop_dataset_loaders, window_mult=cfg.window_mult
    )

    if local_rank == 0:
        print(f"[{phase_label}] full dataset len={len(all_training_data)}")
        print(asdict(cfg))

    train_model(
        cfg=cfg,
        training_data=all_training_data,
        eval_datasets=all_eval_data,
        tokenizer=tokenizer,
        dtype=dtype,
        device=device,
        model_kwargs=model_kwargs,
        verbose=True,
        save_checkpoint_on_complete=True,
        checkpoint_complete_extra={"seq_dual_target_phase_complete": True},
        dual_target_phase_label=phase_label,
    )

    meta = read_latest_checkpoint_meta(cfg.save_dir)
    global_step = int(meta.get("global_step", 0))
    if local_rank == 0:
        _write_state(cfg.save_dir, phase_index + 1, phase_label, global_step)
    dist.barrier()
    return cfg.save_dir


def _eval_classification(
    *,
    cfg: SelfInterpTrainingConfig,
    target_model_name: str,
    lora_path: str,
    loop_dataset_loaders: list,
    tokenizer,
    device: torch.device,
    dtype: torch.dtype,
    model_kwargs: dict,
    local_rank: int,
    phase_label: str,
) -> dict[str, float]:
    """
    Run classification eval on `target_model_name` using the oracle at `lora_path`.
    Returns {dataset_name: accuracy} on rank 0; empty dict on other ranks.
    """
    if local_rank != 0:
        dist.barrier()
        return {}

    from peft import PeftModel
    from nl_probes.utils.lora_utils import sanitize_lora_name

    oracle_model = load_model(cfg.oracle_model_name, dtype, **{**model_kwargs, "device_map": {"": f"cuda:{local_rank}"}})
    target_model = load_model(target_model_name, dtype, **{**model_kwargs, "device_map": {"": f"cuda:{local_rank}"}})
    target_model.eval()

    lora_san = sanitize_lora_name(lora_path)
    oracle_model.load_adapter(lora_path, adapter_name=lora_san, is_trainable=False)
    oracle_model.set_adapter(lora_san)
    oracle_model.eval()

    submodule = get_hf_submodule(oracle_model, cfg.hook_onto_layer)
    target_tokenizer = load_tokenizer(target_model_name)

    # projector (if hidden sizes differ)
    target_width = target_model.config.hidden_size
    oracle_width = oracle_model.config.hidden_size
    projector = None
    if target_width != oracle_width:
        import glob as _glob
        pths = sorted(_glob.glob(os.path.join(cfg.save_dir, "checkpoint_step_*.pth")))
        if pths:
            import torch as _torch
            ckpt = _torch.load(pths[-1], map_location=device, weights_only=False)
            proj_sd = ckpt["module_states"]["activation_projector"]
            has_bias = "bias" in proj_sd
            projector = torch.nn.Linear(target_width, oracle_width, bias=has_bias, device=device, dtype=dtype)
            projector.load_state_dict(proj_sd)
            projector.eval()

    _, all_eval_data = build_datasets(cfg, dataset_loaders=loop_dataset_loaders, window_mult=None)

    scores: dict[str, float] = {}
    for ds_name, eval_data in all_eval_data.items():
        results = run_evaluation(
            eval_data=eval_data,
            model=oracle_model,
            tokenizer=tokenizer,
            submodule=submodule,
            device=device,
            dtype=dtype,
            global_step=-1,
            lora_path=lora_san,
            eval_batch_size=cfg.eval_batch_size,
            steering_coefficient=1.0,
            generation_kwargs={"do_sample": False, "temperature": 0.0, "max_new_tokens": 10},
            activation_tokenizer=target_tokenizer,
            activation_model=target_model,
            projection_module=projector,
        )
        _fmt_correct, ans_correct = score_eval_responses(results, eval_data)
        scores[ds_name] = ans_correct

    avg = sum(scores.values()) / len(scores) if scores else float("nan")
    print(f"\n[{phase_label}] Eval on target={target_model_name}")
    for ds, acc in sorted(scores.items()):
        print(f"  {ds:<42} {acc:.4f}")
    print(f"  {'AVERAGE':<42} {avg:.4f}\n", flush=True)

    del oracle_model, target_model, projector
    torch.cuda.empty_cache()

    dist.barrier()
    return scores


# ── main ──────────────────────────────────────────────────────────────────────

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
            "This entrypoint shares one activation projector; use targets with identical hidden_size."
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
        bnb_config = BitsAndBytesConfig(load_in_8bit=True, bnb_8bit_compute_dtype=dtype)
        model_kwargs = {"quantization_config": bnb_config, "attn_implementation": "sdpa"}

    tokenizer = load_tokenizer(args.oracle_model_name)

    def loop_for(target_name: str) -> list:
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
    bootstrap = args.bootstrap_lora_path or ""

    start_phase_idx, lora_path = _first_phase_and_lora(save_dir, bootstrap)

    all_phases_complete = start_phase_idx >= len(SEQ_PHASE_LABELS)

    if all_phases_complete:
        if local_rank == 0:
            print(f"SEQ_DUAL_TARGET: all phases complete (index {start_phase_idx}), running final eval.")
    else:
        if local_rank == 0:
            print(
                f"SEQ_DUAL_TARGET resume: starting at phase index {start_phase_idx} "
                f"({SEQ_PHASE_LABELS[start_phase_idx]}) save_dir={save_dir} load_lora={lora_path}"
            )
            if start_phase_idx > 0 and not _read_state(save_dir):
                meta = read_latest_checkpoint_meta(save_dir) if os.path.isdir(save_dir) else {}
                _write_state(
                    save_dir,
                    start_phase_idx,
                    SEQ_PHASE_LABELS[start_phase_idx - 1],
                    int(meta.get("global_step", 0)),
                )
    dist.barrier()

    phase_specs: list[dict[str, Any]] = [
        {"index": 0, "label": SEQ_PHASE_LABELS[0], "target": args.target_a, "loaders": loop_a},
        {"index": 1, "label": SEQ_PHASE_LABELS[1], "target": args.target_b, "loaders": loop_b},
    ]

    for spec in phase_specs:
        idx = int(spec["index"])
        if idx < start_phase_idx:
            if local_rank == 0:
                print(f"SEQ_DUAL_TARGET skip completed phase [{spec['label']}] target={spec['target']}")
            continue

        save_dir = _run_phase(
            cfg=SelfInterpTrainingConfig(
                oracle_model_name=args.oracle_model_name,
                target_model_name=spec["target"],
                wandb_suffix=args.wandb_suffix,
            ),
            target_model_name=spec["target"],
            load_lora_path=lora_path,
            phase_label=str(spec["label"]),
            phase_index=idx,
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
        )
        lora_path = f"{save_dir}/final"

    # ── Final eval on both targets ────────────────────────────────────────
    final_lora = os.path.join(save_dir, "final")
    all_target_scores: dict[str, dict[str, float]] = {}
    for target_name, loaders in [(args.target_a, loop_a), (args.target_b, loop_b)]:
        eval_cfg = SelfInterpTrainingConfig(
            oracle_model_name=args.oracle_model_name,
            target_model_name=target_name,
            wandb_suffix=args.wandb_suffix,
        )
        eval_cfg.eval_batch_size = (train_batch_size * world_size) * 8
        eval_cfg.hook_onto_layer = args.hook_onto_layer
        eval_cfg.layer_percents = args.layer_percents
        eval_cfg.finalize(dataset_loaders=loaders)
        scores = _eval_classification(
            cfg=eval_cfg,
            target_model_name=target_name,
            lora_path=final_lora,
            loop_dataset_loaders=loaders,
            tokenizer=tokenizer,
            device=device,
            dtype=dtype,
            model_kwargs=model_kwargs,
            local_rank=local_rank,
            phase_label=f"final_eval_target_{target_name.split('/')[-1]}",
        )
        all_target_scores[target_name] = scores

    if local_rank == 0 and all_target_scores:
        all_accs = [acc for scores in all_target_scores.values() for acc in scores.values()]
        grand_avg = sum(all_accs) / len(all_accs)
        print(f"\nGRAND AVERAGE (both targets, all datasets): {grand_avg:.4f}\n", flush=True)

    dist.destroy_process_group()