"""
Phased triple-target oracle training (single ``torchrun`` job).

Calls ``train_model`` from ``nl_probes/sft.py`` six times on the same ``wandb_suffix`` / ``save_dir``:

  Phase 1a — first half of target A
  Phase 1b — first half of target B
  Phase 1c — first half of target C
  Phase 2a — second half of target A
  Phase 2b — second half of target B
  Phase 2c — second half of target C

Target C defaults to ``Qwen/Qwen3-4B-Instruct-2507`` (same hidden_size=2560 as Qwen3-4B / Qwen3.5-4B,
so the shared activation projector applies without modification).

**Shared projector constraint:** all three targets must have the same ``hidden_size`` as each other.
A mismatch raises a ``ValueError`` at startup rather than silently producing wrong results.

Models are loaded one at a time (training phases load a single target; eval loads one target,
evaluates, then frees it before loading the next), so peak VRAM is no higher than dual-target.
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
from nl_probes.utils.hf_config import pretrained_hidden_size
from nl_probes.utils.activation_utils import get_hf_submodule
from nl_probes.utils.eval import run_evaluation, score_eval_responses

# ── dataset config ────────────────────────────────────────────────────────────
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

# ── phase config ──────────────────────────────────────────────────────────────
TRIPLE_TARGET_PHASE_LABELS: list[str] = [
    "phase_1a_first_half_target_A",
    "phase_1b_first_half_target_B",
    "phase_1c_first_half_target_C",
    "phase_2a_second_half_target_A",
    "phase_2b_second_half_target_B",
    "phase_2c_second_half_target_C",
]

STATE_FILENAME = "TRIPLE_TARGET_STATE.json"
MILESTONE_LOG_LINE = (
    "MILESTONE_TRIPLE_TARGET_HALFWAY "
    "phase=after_first_half_all_three_targets "
    "search_this_string_in_slurm_out_or_err"
)

_DEFAULT_TARGET_C = "Qwen/Qwen2.5-4B-Instruct"

# Approx steps per half-slice per target on 1 GPU; used for step-based resume fallback.
_STEPS_PER_HALF = 30_000


# ── CLI ───────────────────────────────────────────────────────────────────────
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--oracle_model_name",           type=str, required=True)
    p.add_argument("--target_a",                    type=str, required=True)
    p.add_argument("--target_b",                    type=str, required=True)
    p.add_argument("--target_c",                    type=str, default=_DEFAULT_TARGET_C,
                   help=f"Third target model. Default: {_DEFAULT_TARGET_C}")
    p.add_argument("--wandb_suffix",                type=str, required=True)
    p.add_argument("--hook_onto_layer",             type=int, default=1)
    p.add_argument("--train_batch_size",            type=int, default=16)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument(
        "--gradient_checkpointing",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="If omitted: auto-detected from oracle/target model lineage.",
    )
    p.add_argument("--eval_steps",     type=int, default=10_000)
    p.add_argument("--save_steps",     type=int, default=10_000)
    p.add_argument("--layer_percents", type=int, nargs="+", default=[25, 50, 75])
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
        "phase_order": TRIPLE_TARGET_PHASE_LABELS,
    }
    os.makedirs(save_dir, exist_ok=True)
    with open(_state_path(save_dir), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _phase_index_from_global_step(global_step: int) -> int:
    """Map cumulative step count to the next phase index (0–5)."""
    for i in range(len(TRIPLE_TARGET_PHASE_LABELS), 0, -1):
        if global_step >= i * _STEPS_PER_HALF:
            return i
    return 0


def _first_phase_index_to_run(save_dir: str) -> int:
    state = _read_state(save_dir)
    if state:
        return int(state["next_phase_index"])

    meta = read_latest_checkpoint_meta(save_dir)
    global_step = int(meta.get("global_step", 0))
    extra = meta.get("extra", {})

    milestone_path = os.path.join(save_dir, "MILESTONE_TRIPLE_TARGET_HALFWAY.txt")
    if os.path.isfile(milestone_path) and global_step >= 3 * _STEPS_PER_HALF:
        return 3

    step_index = _phase_index_from_global_step(global_step)

    completed = extra.get("triple_target_phase")
    if completed not in TRIPLE_TARGET_PHASE_LABELS:
        return step_index

    phase_idx = TRIPLE_TARGET_PHASE_LABELS.index(completed)
    if extra.get("triple_target_phase_complete"):
        meta_index = phase_idx + 1
        return max(step_index, meta_index) if global_step >= (phase_idx + 1) * _STEPS_PER_HALF else step_index

    phase_start = int(extra.get("triple_target_phase_start_global_step", 0))
    if phase_idx == 0 and global_step > 0 and phase_start == global_step:
        return max(step_index, 1)

    return max(step_index, phase_idx)


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
    which_half: int,
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
        checkpoint_complete_extra={"triple_target_phase_complete": True},
        dual_target_phase_label=phase_label,
    )

    meta = read_latest_checkpoint_meta(cfg.save_dir)
    global_step = int(meta.get("global_step", 0))
    if local_rank == 0:
        _write_state(cfg.save_dir, phase_index + 1, phase_label, global_step)
    dist.barrier()
    return cfg.save_dir


# ── eval ──────────────────────────────────────────────────────────────────────
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
    Evaluate oracle at ``lora_path`` on ``target_model_name``.
    Loads and frees models individually to keep VRAM low.
    Returns {dataset_name: accuracy} on rank 0; empty dict on other ranks.
    """
    if local_rank != 0:
        dist.barrier()
        return {}

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

    target_width = target_model.config.hidden_size
    oracle_width = oracle_model.config.hidden_size
    projector = None
    if target_width != oracle_width:
        import glob as _glob
        pths = sorted(_glob.glob(os.path.join(cfg.save_dir, "checkpoint_step_*.pth")))
        if pths:
            ckpt = torch.load(pths[-1], map_location=device, weights_only=False)
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

    wa = pretrained_hidden_size(args.target_a)
    wb = pretrained_hidden_size(args.target_b)
    wc = pretrained_hidden_size(args.target_c)
    mismatches = {args.target_a: wa, args.target_b: wb, args.target_c: wc}
    if len(set(mismatches.values())) > 1:
        raise ValueError(...)

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

    _LARGE_MODELS = ("Qwen/Qwen3-32B", "meta-llama/Llama-3.3-70B-Instruct")
    model_kwargs: dict = {"attn_implementation": "sdpa"}
    if any(t in _LARGE_MODELS for t in (args.target_a, args.target_b, args.target_c)):
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
    loop_c = loop_for(args.target_c)

    save_dir = namespaced_save_dir(wandb_suffix=args.wandb_suffix)
    start_phase_idx = _first_phase_index_to_run(save_dir)
    all_phases_complete = start_phase_idx >= len(TRIPLE_TARGET_PHASE_LABELS)

    if all_phases_complete:
        if local_rank == 0:
            print(f"TRIPLE_TARGET: all phases complete (index {start_phase_idx}), running final eval.")
    else:
        if local_rank == 0:
            print(
                f"TRIPLE_TARGET resume: starting at phase index {start_phase_idx} "
                f"({TRIPLE_TARGET_PHASE_LABELS[start_phase_idx]}) save_dir={save_dir}"
            )
            if start_phase_idx > 0 and not _read_state(save_dir):
                meta = read_latest_checkpoint_meta(save_dir) if os.path.isdir(save_dir) else {}
                _write_state(
                    save_dir,
                    start_phase_idx,
                    TRIPLE_TARGET_PHASE_LABELS[start_phase_idx - 1],
                    int(meta.get("global_step", 0)),
                )
    dist.barrier()

    lora_path: str | None = None
    if start_phase_idx > 0:
        lora_path = f"{save_dir}/final"

    phase_specs: list[dict[str, Any]] = [
        {"index": 0, "label": TRIPLE_TARGET_PHASE_LABELS[0], "target": args.target_a, "half": 0, "loaders": loop_a},
        {"index": 1, "label": TRIPLE_TARGET_PHASE_LABELS[1], "target": args.target_b, "half": 0, "loaders": loop_b},
        {"index": 2, "label": TRIPLE_TARGET_PHASE_LABELS[2], "target": args.target_c, "half": 0, "loaders": loop_c},
        {"index": 3, "label": TRIPLE_TARGET_PHASE_LABELS[3], "target": args.target_a, "half": 1, "loaders": loop_a},
        {"index": 4, "label": TRIPLE_TARGET_PHASE_LABELS[4], "target": args.target_b, "half": 1, "loaders": loop_b},
        {"index": 5, "label": TRIPLE_TARGET_PHASE_LABELS[5], "target": args.target_c, "half": 1, "loaders": loop_c},
    ]

    for spec in phase_specs:
        idx = int(spec["index"])
        if idx < start_phase_idx:
            if local_rank == 0:
                print(f"TRIPLE_TARGET skip completed phase [{spec['label']}] target={spec['target']}")
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

        # Halfway milestone: after phase 1c (index 2), all first halves are done.
        if idx == 2:
            if local_rank == 0:
                print("\n" + "=" * 88)
                print(MILESTONE_LOG_LINE)
                print("=" * 88 + "\n", flush=True)
                marker_path = os.path.join(save_dir, "MILESTONE_TRIPLE_TARGET_HALFWAY.txt")
                with open(marker_path, "w", encoding="utf-8") as f:
                    f.write(MILESTONE_LOG_LINE + "\n")
                    f.write(f"target_a={args.target_a}\n")
                    f.write(f"target_b={args.target_b}\n")
                    f.write(f"target_c={args.target_c}\n")
                    f.write(f"oracle={args.oracle_model_name}\n")
            dist.barrier()

    # ── Final eval on all three targets (one at a time) ───────────────────────
    final_lora = os.path.join(save_dir, "final")
    all_target_scores: dict[str, dict[str, float]] = {}

    for target_name, loaders in [
        (args.target_a, loop_a),
        (args.target_b, loop_b),
        (args.target_c, loop_c),
    ]:
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
        print(f"\nGRAND AVERAGE (all three targets, all datasets): {grand_avg:.4f}\n", flush=True)

    dist.destroy_process_group()