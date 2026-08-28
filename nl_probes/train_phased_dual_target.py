"""
Phased dual-target oracle training (single ``torchrun`` job).

Calls ``train_model`` from ``nl_probes/sft.py`` many times on the same ``wandb_suffix`` /
``save_dir``, round-robining between target A and target B in small **chunks** instead of
halves. This makes the oracle interleave between the two targets much more finely (more,
smaller context switches) while keeping the exact same total amount of training data per
target as before.

**Chunking:** each target's training mixture is split into ``N_CHUNKS`` contiguous slices
(default 16). Phases run in round-robin order: A-chunk0, B-chunk0, A-chunk1, B-chunk1, ...,
A-chunk{N-1}, B-chunk{N-1}. That is ``2 * N_CHUNKS`` phases total, each one a full
``train_model`` call that reloads/re-attaches the target model. Lower ``N_CHUNKS`` back
toward 2 (halves) if per-phase model-reload overhead is too costly on your hardware;
raise it for finer interleaving.

**Dataset-size matching with the triple-target script:** to keep the *total* number of
training examples equal between this dual-target run and ``train_phased_triple_target.py``
(which has three targets sharing the same per-dataset pool sizes), this script trains on
the **full** ``CLASSIFICATION_DATASETS`` sizes below, while the triple-target script scales
its per-target sizes down by ``2/3``. Concretely: dual trains 2 targets x N examples/target
= 2N total; triple trains 3 targets x (2/3)N examples/target = 2N total. Same grand total.
If you change ``_MAIN_TRAIN_SIZE`` here, update the scale factor in the triple-target
script to match.

**Projection note (current ``train_model`` API):** each target now gets its own
``activation_projector`` checkpoint key, derived from the target model's name via
``activation_projector_key``. Target A and B therefore each learn an independent
``target_hidden -> oracle_hidden`` linear map, even when their hidden sizes differ. Checkpoints
merge all previously-known per-target projector states on every save, so round-robining between
targets across phases no longer resets the other target's projector.
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
from nl_probes.sft import (
    _ensure_datasets_exist,
    activation_projector_key,
    build_datasets,
    build_loader_groups,
    make_activation_projector,
    train_model,
)
from nl_probes.utils.checkpointing import read_latest_checkpoint_meta
from nl_probes.utils.lora_utils import sanitize_lora_name
from nl_probes.utils.common import load_model, load_tokenizer
from nl_probes.utils.hf_config import pretrained_hidden_size
from nl_probes.utils.activation_utils import get_hf_submodule
from nl_probes.utils.eval import run_evaluation, score_eval_responses

# ── chunk config ────────────────────────────────────────────────────────────
# Number of contiguous slices each target's training mixture is split into.
# Total phases = 2 * N_CHUNKS (round-robin: A0, B0, A1, B1, ..., A{N-1}, B{N-1}).
# Tune this down if per-phase target-model reload overhead is too costly on your GPU;
# tune it up for finer-grained interleaving between the two targets.
N_CHUNKS = 16

# Same table as ``nl_probes/sft.py`` ``__main__`` (classification sizes for ``build_loader_groups``).
# Dual-target keeps the FULL size per target; triple-target scales this down by 2/3 so that
# total training examples (targets x per-target size) match between the two scripts.
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
    "phase=after_first_half_of_chunks_both_targets "
    "search_this_string_in_slurm_out_or_err"
)

DUAL_TARGET_STATE_FILENAME = "DUAL_TARGET_STATE.json"

# Approx optimizer steps per chunk on 1 GPU; used only as a fallback for step-based resume
# when the JSON state file is missing (e.g. first-ever run of a resumed old save_dir).
# Previously (2 chunks/target == halves) each half was ~30_000 steps; with N_CHUNKS chunks
# each chunk is proportionally smaller.
_DUAL_TARGET_STEPS_PER_HALF_LEGACY = 30_000
_DUAL_TARGET_STEPS_PER_CHUNK = max(1, (_DUAL_TARGET_STEPS_PER_HALF_LEGACY * 2) // N_CHUNKS)


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
    p.add_argument(
        "--n_chunks",
        type=int,
        default=N_CHUNKS,
        help=f"Number of chunks to split each target's mixture into (default {N_CHUNKS}). "
        "Total phases = 2 * n_chunks.",
    )
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


def _write_dual_target_state(
    save_dir: str, next_phase_index: int, last_completed_phase: str, global_step: int, phase_labels: list[str]
) -> None:
    payload = {
        "next_phase_index": next_phase_index,
        "last_completed_phase": last_completed_phase,
        "global_step": global_step,
        "phase_order": phase_labels,
    }
    with open(_dual_target_state_path(save_dir), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _phase_index_from_global_step(global_step: int, n_phases: int) -> int:
    """Map cumulative step count to the next phase index, using the approx steps/chunk."""
    for i in range(n_phases, 0, -1):
        if global_step >= i * _DUAL_TARGET_STEPS_PER_CHUNK:
            return i
    return 0


def _first_phase_index_to_run(save_dir: str, n_phases: int) -> int:
    """
    JSON state file (written after every completed phase) is the source of truth and works
    regardless of chunk granularity. The step-based heuristics below are only a fallback for
    resuming a save_dir that has checkpoints but no state file yet (e.g. migrating from an
    older run of this script).
    """
    state = _read_dual_target_state(save_dir)
    if state:
        return int(state["next_phase_index"])

    meta = read_latest_checkpoint_meta(save_dir)
    global_step = int(meta.get("global_step", 0))
    extra = meta.get("extra", {})

    milestone_path = os.path.join(save_dir, "MILESTONE_DUAL_TARGET_HALFWAY.txt")
    halfway_index = n_phases // 2
    if os.path.isfile(milestone_path) and global_step >= halfway_index * _DUAL_TARGET_STEPS_PER_CHUNK:
        return halfway_index

    step_index = _phase_index_from_global_step(global_step, n_phases)

    completed = extra.get("dual_target_phase")
    if not completed:
        return step_index

    return step_index


def _run_phase(
    *,
    cfg: SelfInterpTrainingConfig,
    target_model_name: str,
    chunk_index: int,
    n_chunks: int,
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
    phase_labels: list[str],
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
    chunk_size = n // n_chunks
    start = chunk_index * chunk_size
    end = n if chunk_index == n_chunks - 1 else start + chunk_size
    train_slice = slice(start, end)
    phase_train = all_training_data[train_slice]
    if local_rank == 0:
        print(
            f"[{phase_label}] chunk={chunk_index}/{n_chunks} slice={train_slice} of n={n} -> len={len(phase_train)}"
        )
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
        _write_dual_target_state(cfg.save_dir, phase_index + 1, phase_label, global_step, phase_labels)
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
        proj_key = activation_projector_key(target_model_name)

        def _step_num(p: str) -> int:
            stem = os.path.splitext(os.path.basename(p))[0]
            return int(stem.rsplit("_", 1)[-1])

        pths = sorted(_glob.glob(os.path.join(cfg.save_dir, "checkpoint_step_*.pth")), key=_step_num, reverse=True)
        proj_sd = None
        for pth in pths:
            ckpt = torch.load(pth, map_location=device, weights_only=False)
            module_states = ckpt.get("module_states") or {}
            if proj_key in module_states:
                proj_sd = module_states[proj_key]
                break
            if "activation_projector" in module_states:  # legacy shared-projector fallback
                proj_sd = module_states["activation_projector"]
                break
        if proj_sd is not None:
            # New checkpoints use the gated residual MLP projector. Legacy linear
            # checkpoints are migrated by the module's state-dict loader; infer bias
            # from either the legacy or new key layout.
            has_bias = "base.bias" in proj_sd or "bias" in proj_sd
            projector = make_activation_projector(
                target_width,
                oracle_width,
                bias=has_bias,
                device=device,
                dtype=dtype,
            )
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


if __name__ == "__main__":
    args = _parse_args()
    n_chunks = args.n_chunks
    assert n_chunks >= 1, "n_chunks must be >= 1"

    wa = pretrained_hidden_size(args.target_a)
    wb = pretrained_hidden_size(args.target_b)
    if wa != wb and os.environ.get("LOCAL_RANK", "0") == "0":
        print(
            f"target_a ({args.target_a}) hidden_size={wa} != target_b ({args.target_b}) hidden_size={wb}. "
            "Each target now trains its own activation_projector (keyed via activation_projector_key), "
            "so differing hidden sizes across targets are supported."
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

    # ── build round-robin phase schedule: A0, B0, A1, B1, ..., A{n-1}, B{n-1} ──
    targets_info = [("A", args.target_a, loop_a), ("B", args.target_b, loop_b)]
    phase_specs: list[dict[str, Any]] = []
    idx = 0
    for chunk_idx in range(n_chunks):
        for letter, target_name, loaders in targets_info:
            phase_specs.append(
                {
                    "index": idx,
                    "label": f"phase_c{chunk_idx:02d}_target_{letter}",
                    "target": target_name,
                    "chunk": chunk_idx,
                    "loaders": loaders,
                }
            )
            idx += 1
    phase_labels = [s["label"] for s in phase_specs]
    n_phases = len(phase_specs)
    halfway_phase_index = n_phases // 2  # after all targets have completed the first half of their chunks

    start_phase_idx = _first_phase_index_to_run(save_dir, n_phases)
    all_phases_complete = start_phase_idx >= n_phases

    if all_phases_complete:
        if local_rank == 0:
            print(f"DUAL_TARGET: all phases complete (index {start_phase_idx}), running final eval.")
    else:
        if local_rank == 0:
            print(
                f"DUAL_TARGET resume: starting at phase index {start_phase_idx} "
                f"({phase_labels[start_phase_idx] if start_phase_idx < n_phases else 'N/A'}) "
                f"save_dir={save_dir} n_chunks={n_chunks} n_phases={n_phases}"
            )
            if start_phase_idx > 0 and not _read_dual_target_state(save_dir):
                meta = read_latest_checkpoint_meta(save_dir)
                _write_dual_target_state(
                    save_dir,
                    start_phase_idx,
                    phase_labels[start_phase_idx - 1],
                    int(meta.get("global_step", 0)),
                    phase_labels,
                )
    dist.barrier()

    lora_path: str | None = None
    if start_phase_idx > 0:
        lora_path = f"{save_dir}/final"

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
            chunk_index=int(spec["chunk"]),
            n_chunks=n_chunks,
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
            phase_labels=phase_labels,
        )
        lora_path = f"{save_dir}/final"

        if idx + 1 == halfway_phase_index:
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
                    f.write(f"n_chunks={n_chunks}\n")
            dist.barrier()

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