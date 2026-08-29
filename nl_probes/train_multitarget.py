"""
Generalized phased multi-target Activation Oracle training.

Targets are supplied with repeated --target_model flags. Their order is preserved
exactly in the phase schedule:

    chunk 0: target 0, target 1, ..., target N-1
    chunk 1: target 0, target 1, ..., target N-1
    ...

With --n_chunks 1, training is sequential over the target list. Each target gets
its own activation projector checkpoint key through activation_projector_key().
The projector implementation is imported from nl_probes.sft and is the gated
residual MLP projector with legacy linear-checkpoint compatibility.

This script intentionally reuses train_model() so optimizer, DDP, checkpoint,
LoRA, evaluation, and signal-handling behavior stay aligned with the main SFT
trainer. train_model() currently exposes the phase argument as
``multi_target_phase_label``; it is target-count agnostic and is used here for
all numbers of targets.
"""

from __future__ import annotations

import argparse
import glob
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
from nl_probes.utils.activation_utils import get_hf_submodule
from nl_probes.utils.checkpointing import read_latest_checkpoint_meta
from nl_probes.utils.common import load_model, load_tokenizer
from nl_probes.utils.eval import run_evaluation, score_eval_responses
from nl_probes.utils.hf_config import pretrained_hidden_size
from nl_probes.utils.lora_utils import sanitize_lora_name


_MULTI_MAIN_TRAIN_SIZE = 6000
_MAIN_TEST_SIZE = 250
_DEFAULT_TARGET_C = "Qwen/Qwen2.5-4B-Instruct"
_DEFAULT_N_CHUNKS = 16
_STATE_FILENAME = "MULTITARGET_STATE.json"
_STATE_SCHEMA_VERSION = 1
_STEPS_PER_HALF_LEGACY = 30_000

def _classification_datasets(num_targets: int) -> dict[str, dict[str, Any]]:
    per_target_train_size = round(_MULTI_MAIN_TRAIN_SIZE * 2 / num_targets)

    return {
        "geometry_of_truth": {"num_train": per_target_train_size, "num_test": _MAIN_TEST_SIZE, "splits": ["train", "test"]},
        "relations": {"num_train": per_target_train_size, "num_test": _MAIN_TEST_SIZE, "splits": ["train", "test"]},
        "sst2": {"num_train": per_target_train_size, "num_test": _MAIN_TEST_SIZE, "splits": ["train", "test"]},
        "md_gender": {"num_train": per_target_train_size, "num_test": _MAIN_TEST_SIZE, "splits": ["train", "test"]},
        "snli": {"num_train": per_target_train_size, "num_test": _MAIN_TEST_SIZE, "splits": ["train", "test"]},
        "ag_news": {"num_train": per_target_train_size, "num_test": _MAIN_TEST_SIZE, "splits": ["test"]},
        "ner": {"num_train": per_target_train_size, "num_test": _MAIN_TEST_SIZE, "splits": ["train", "test"]},
        "tense": {"num_train": per_target_train_size, "num_test": _MAIN_TEST_SIZE, "splits": ["train", "test"]},
        "language_identification": {
            "num_train": per_target_train_size,
            "num_test": _MAIN_TEST_SIZE,
            "splits": ["test"],
            "batch_size": 4,
        },
        "singular_plural": {"num_train": 0, "num_test": _MAIN_TEST_SIZE, "splits": ["test"]},
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle_model_name", type=str, required=True)
    parser.add_argument(
        "--target_model",
        action="append",
        required=True,
        dest="target_models",
        help="Target model. Repeat this flag; order defines phase order.",
    )
    parser.add_argument("--wandb_suffix", type=str, required=True)
    parser.add_argument("--hook_onto_layer", type=int, default=1)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument(
        "--gradient_checkpointing",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="If omitted, use the config/default behavior.",
    )
    parser.add_argument("--eval_steps", type=int, default=10_000)
    parser.add_argument("--save_steps", type=int, default=10_000)
    parser.add_argument("--layer_percents", type=int, nargs="+", default=[25, 50, 75])
    parser.add_argument("--mixture_profile", choices=["legacy", "diverse_fixed", "context_heavy_fixed"], default=None)
    parser.add_argument("--mixture_seed", type=int, default=None)
    parser.add_argument("--mixture_total_samples", type=int, default=None)
    parser.add_argument(
        "--n_chunks",
        type=int,
        default=_DEFAULT_N_CHUNKS,
        help="Number of contiguous chunks per target. 1 means sequential target training.",
    )
    return parser.parse_args()


def _phase_steps_per_chunk(n_chunks: int) -> int:
    return max(1, (_STEPS_PER_HALF_LEGACY * 2) // n_chunks)


def _state_path(save_dir: str) -> str:
    return os.path.join(save_dir, _STATE_FILENAME)


def _read_state(save_dir: str) -> dict[str, Any]:
    path = _state_path(save_dir)
    if not os.path.isfile(path):
        return {}
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _write_state(
    save_dir: str,
    *,
    oracle_model_name: str,
    target_models: list[str],
    n_chunks: int,
    phase_labels: list[str],
    next_phase_index: int,
    last_completed_phase: str,
    global_step: int,
) -> None:
    os.makedirs(save_dir, exist_ok=True)
    payload = {
        "schema_version": _STATE_SCHEMA_VERSION,
        "oracle_model_name": oracle_model_name,
        "target_models": target_models,
        "n_chunks": n_chunks,
        "phase_order": phase_labels,
        "next_phase_index": next_phase_index,
        "last_completed_phase": last_completed_phase,
        "global_step": global_step,
    }
    with open(_state_path(save_dir), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _checkpoint_exists(save_dir: str) -> bool:
    return bool(
        glob.glob(os.path.join(save_dir, "checkpoint_step_*.pth"))
        or glob.glob(os.path.join(save_dir, "checkpoint_epoch_*.pth"))
        or os.path.isdir(os.path.join(save_dir, "final"))
    )


def _validate_or_initialize_resume(
    save_dir: str,
    *,
    oracle_model_name: str,
    target_models: list[str],
    n_chunks: int,
    phase_labels: list[str],
) -> int:
    """Strictly validate the schedule and return the next phase index."""
    state = _read_state(save_dir)
    if not state:
        if _checkpoint_exists(save_dir):
            raise RuntimeError(
                f"Found checkpoints in {save_dir} but no {_STATE_FILENAME}. "
                "Refusing an unvalidated resume; use a new wandb_suffix/save directory."
            )
        return 0

    expected = {
        "schema_version": _STATE_SCHEMA_VERSION,
        "oracle_model_name": oracle_model_name,
        "target_models": target_models,
        "n_chunks": n_chunks,
        "phase_order": phase_labels,
    }
    mismatches = {
        key: (state.get(key), value)
        for key, value in expected.items()
        if state.get(key) != value
    }
    if mismatches:
        details = "; ".join(f"{key}: saved={old!r}, requested={new!r}" for key, (old, new) in mismatches.items())
        raise RuntimeError(
            "Existing multi-target run is incompatible with the requested schedule. "
            f"{details}. Use a new wandb_suffix/save directory."
        )

    next_phase_index = int(state.get("next_phase_index", 0))
    if not 0 <= next_phase_index <= len(phase_labels):
        raise RuntimeError(f"Invalid next_phase_index={next_phase_index} in {_STATE_FILENAME}")
    return next_phase_index


def _step_number(path: str) -> int:
    stem = os.path.splitext(os.path.basename(path))[0]
    try:
        return int(stem.rsplit("_", 1)[-1])
    except ValueError:
        return -1


def _load_projector_for_eval(
    *,
    save_dir: str,
    target_model_name: str,
    target_width: int,
    oracle_width: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.nn.Module:
    projector_key = activation_projector_key(target_model_name)
    checkpoints = sorted(
        glob.glob(os.path.join(save_dir, "checkpoint_step_*.pth")),
        key=_step_number,
        reverse=True,
    )
    projector_state = None
    for checkpoint_path in checkpoints:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        module_states = checkpoint.get("module_states") or {}
        if projector_key in module_states:
            projector_state = module_states[projector_key]
            break
        if "activation_projector" in module_states:
            projector_state = module_states["activation_projector"]
            break

    if projector_state is None:
        raise RuntimeError(
            f"No projector checkpoint found for target {target_model_name!r} under key {projector_key!r}."
        )

    has_bias = "base.bias" in projector_state or "bias" in projector_state
    projector = make_activation_projector(
        target_width,
        oracle_width,
        bias=has_bias,
        device=device,
        dtype=dtype,
    )
    projector.load_state_dict(projector_state)
    projector.eval()
    return projector


def _eval_target(
    *,
    cfg: SelfInterpTrainingConfig,
    target_model_name: str,
    lora_path: str,
    loop_dataset_loaders: list,
    tokenizer,
    device: torch.device,
    dtype: torch.dtype,
    model_kwargs: dict[str, Any],
    local_rank: int,
    phase_label: str,
) -> dict[str, float]:
    if local_rank != 0:
        dist.barrier()
        return {}

    oracle_model = load_model(
        cfg.oracle_model_name,
        dtype,
        **{**model_kwargs, "device_map": {"": f"cuda:{local_rank}"}},
    )
    target_model = load_model(
        target_model_name,
        dtype,
        **{**model_kwargs, "device_map": {"": f"cuda:{local_rank}"}},
    )
    target_model.eval()

    lora_name = sanitize_lora_name(lora_path)
    oracle_model.load_adapter(lora_path, adapter_name=lora_name, is_trainable=False)
    oracle_model.set_adapter(lora_name)
    oracle_model.eval()

    target_width = target_model.config.hidden_size
    oracle_width = oracle_model.config.hidden_size
    projector = None
    if target_width != oracle_width:
        projector = _load_projector_for_eval(
            save_dir=cfg.save_dir,
            target_model_name=target_model_name,
            target_width=target_width,
            oracle_width=oracle_width,
            device=device,
            dtype=dtype,
        )

    submodule = get_hf_submodule(oracle_model, cfg.hook_onto_layer)
    target_tokenizer = load_tokenizer(target_model_name)
    _, eval_datasets = build_datasets(
        cfg,
        dataset_loaders=loop_dataset_loaders,
        window_mult=None,
    )

    scores: dict[str, float] = {}
    for dataset_name, eval_data in eval_datasets.items():
        results = run_evaluation(
            eval_data=eval_data,
            model=oracle_model,
            tokenizer=tokenizer,
            submodule=submodule,
            device=device,
            dtype=dtype,
            global_step=-1,
            lora_path=lora_name,
            eval_batch_size=cfg.eval_batch_size,
            steering_coefficient=1.0,
            generation_kwargs={"do_sample": False, "temperature": 0.0, "max_new_tokens": 10},
            activation_tokenizer=target_tokenizer,
            activation_model=target_model,
            projection_module=projector,
        )
        _, answer_accuracy = score_eval_responses(results, eval_data)
        scores[dataset_name] = answer_accuracy

    average = sum(scores.values()) / len(scores) if scores else float("nan")
    print(f"\n[{phase_label}] Eval on target={target_model_name}")
    for dataset_name, score in sorted(scores.items()):
        print(f"  {dataset_name:<42} {score:.4f}")
    print(f"  {'AVERAGE':<42} {average:.4f}\n", flush=True)

    del oracle_model, target_model, projector
    torch.cuda.empty_cache()
    dist.barrier()
    return scores


def _run_phase(
    *,
    cfg: SelfInterpTrainingConfig,
    target_model_name: str,
    phase_label: str,
    phase_index: int,
    phase_labels: list[str],
    chunk_index: int,
    n_chunks: int,
    load_lora_path: str | None,
    loop_dataset_loaders: list,
    tokenizer,
    device: torch.device,
    dtype: torch.dtype,
    model_kwargs: dict[str, Any],
    local_rank: int,
    train_batch_size: int,
    hook_onto_layer: int,
    gradient_accumulation_steps: int,
    gradient_checkpointing: bool | None,
    eval_steps: int,
    save_steps: int,
    layer_percents: List[int],
    oracle_model_name: str,
    target_models: list[str],
) -> str:
    cfg.target_model_name = target_model_name
    cfg.load_lora_path = load_lora_path
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

    if cfg.wandb_run_name:
        cfg.wandb_run_name = f"{cfg.wandb_run_name}_{phase_label}"

    if local_rank == 0:
        print(
            f"[{phase_label}] target_index={target_models.index(target_model_name)} "
            f"target={target_model_name} chunk={chunk_index}/{n_chunks} "
            f"save_dir={cfg.save_dir} load_lora={load_lora_path}"
        )
        _ensure_datasets_exist(loop_dataset_loaders)
    dist.barrier()

    all_training_data, all_eval_data = build_datasets(
        cfg,
        dataset_loaders=loop_dataset_loaders,
        window_mult=cfg.window_mult,
    )
    chunk_size = len(all_training_data) // n_chunks
    start = chunk_index * chunk_size
    end = len(all_training_data) if chunk_index == n_chunks - 1 else start + chunk_size
    phase_training_data = all_training_data[slice(start, end)]

    if local_rank == 0:
        print(
            f"[{phase_label}] slice={start}:{end} of n={len(all_training_data)} "
            f"-> phase_examples={len(phase_training_data)}"
        )
        print(asdict(cfg))

    train_model(
        cfg=cfg,
        training_data=phase_training_data,
        eval_datasets=all_eval_data,
        tokenizer=tokenizer,
        dtype=dtype,
        device=device,
        model_kwargs=model_kwargs,
        verbose=True,
        save_checkpoint_on_complete=True,
        checkpoint_complete_extra={
            "multi_target_phase_complete": True,
            "multi_target_phase_label": phase_label,
        },
        multi_target_phase_label=phase_label,
    )

    meta = read_latest_checkpoint_meta(cfg.save_dir)
    global_step = int(meta.get("global_step", 0))
    if local_rank == 0:
        _write_state(
            cfg.save_dir,
            oracle_model_name=oracle_model_name,
            target_models=target_models,
            n_chunks=n_chunks,
            phase_labels=phase_labels,
            next_phase_index=phase_index + 1,
            last_completed_phase=phase_label,
            global_step=global_step,
        )
    dist.barrier()
    return cfg.save_dir


def main() -> None:
    args = _parse_args()
    if args.mixture_profile is not None:
        os.environ["AO_MIXTURE_PROFILE"] = args.mixture_profile
    if args.mixture_seed is not None:
        os.environ["AO_MIXTURE_SEED"] = str(args.mixture_seed)
    if args.mixture_total_samples is not None:
        os.environ["AO_MIXTURE_TOTAL_SAMPLES"] = str(args.mixture_total_samples)

    target_models = list(args.target_models)
    if not target_models:
        raise ValueError("At least one --target_model is required")
    if len(set(target_models)) != len(target_models):
        raise ValueError("Duplicate target models are not allowed; each target needs a distinct projector key")
    if args.n_chunks < 1:
        raise ValueError("--n_chunks must be >= 1")
    if args.gradient_accumulation_steps < 1:
        raise ValueError("--gradient_accumulation_steps must be >= 1")
    if args.classification_total_target_equivalent < 0:
        raise ValueError("--classification_total_target_equivalent must be >= 0")

    hidden_sizes = {target: pretrained_hidden_size(target) for target in target_models}
    if os.environ.get("LOCAL_RANK", "0") == "0":
        print("Ordered target list:")
        for index, target in enumerate(target_models):
            print(f"  [{index}] {target} hidden_size={hidden_sizes[target]}")

    dist.init_process_group(backend="nccl", timeout=timedelta(hours=2))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    world_size = dist.get_world_size()
    dtype = torch.bfloat16
    device = torch.device(f"cuda:{local_rank}")

    if args.train_batch_size % world_size != 0:
        raise ValueError(
            f"Global train batch size {args.train_batch_size} must be divisible by world size {world_size}"
        )
    train_batch_size = args.train_batch_size // world_size

    large_model_names = {
        "Qwen/Qwen3-32B",
        "meta-llama/Llama-3.3-70B-Instruct",
    }
    model_kwargs: dict[str, Any] = {"attn_implementation": "sdpa"}
    if args.oracle_model_name in large_model_names or any(target in large_model_names for target in target_models):
        bnb_config = BitsAndBytesConfig(load_in_8bit=True, bnb_8bit_compute_dtype=dtype)
        model_kwargs = {"quantization_config": bnb_config, "attn_implementation": "sdpa"}

    tokenizer = load_tokenizer(args.oracle_model_name)
    classification_datasets = _classification_datasets(
        len(target_models),
    )

    def loop_for(target_model_name: str) -> list:
        loader_groups = build_loader_groups(
            model_name=target_model_name,
            prompt_model_name=args.oracle_model_name,
            layer_percents=args.layer_percents,
            act_collection_batch_size=train_batch_size,
            save_acts=False,
            classification_datasets=classification_datasets,
            model_kwargs=model_kwargs,
        )
        return (
            loader_groups["latentqa_loaders"]
            + loader_groups["cot_oracle_loaders"]
            + loader_groups["classification_loaders"]
            + loader_groups["past_lens_loaders"]
        )

    loaders_by_target = [loop_for(target) for target in target_models]
    save_dir = namespaced_save_dir(wandb_suffix=args.wandb_suffix)

    phase_specs: list[dict[str, Any]] = []
    for chunk_index in range(args.n_chunks):
        for target_index, (target_model_name, loaders) in enumerate(zip(target_models, loaders_by_target)):
            phase_specs.append(
                {
                    "index": len(phase_specs),
                    "label": f"phase_c{chunk_index:02d}_target_{target_index:02d}",
                    "target_index": target_index,
                    "target": target_model_name,
                    "chunk": chunk_index,
                    "loaders": loaders,
                }
            )
    phase_labels = [spec["label"] for spec in phase_specs]

    start_phase_index = _validate_or_initialize_resume(
        save_dir,
        oracle_model_name=args.oracle_model_name,
        target_models=target_models,
        n_chunks=args.n_chunks,
        phase_labels=phase_labels,
    )
    steps_per_chunk = _phase_steps_per_chunk(args.n_chunks)
    if local_rank == 0:
        print(
            f"MULTITARGET resume: starting phase {start_phase_index}/{len(phase_specs)} "
            f"with {len(target_models)} targets and {args.n_chunks} chunks/target "
            f"(fallback steps/chunk={steps_per_chunk})"
        )
    dist.barrier()

    lora_path: str | None = None
    if start_phase_index > 0:
        lora_path = os.path.join(save_dir, "final")
        if not os.path.isdir(lora_path):
            raise RuntimeError(f"Cannot resume phase {start_phase_index}: missing final adapter at {lora_path}")

    for spec in phase_specs:
        phase_index = int(spec["index"])
        if phase_index < start_phase_index:
            if local_rank == 0:
                print(f"MULTITARGET skip completed phase [{spec['label']}] target={spec['target']}")
            continue

        phase_load_lora = None if phase_index == 0 and start_phase_index == 0 else lora_path
        save_dir = _run_phase(
            cfg=SelfInterpTrainingConfig(
                oracle_model_name=args.oracle_model_name,
                target_model_name=spec["target"],
                wandb_suffix=args.wandb_suffix,
            ),
            target_model_name=spec["target"],
            phase_label=str(spec["label"]),
            phase_index=phase_index,
            phase_labels=phase_labels,
            chunk_index=int(spec["chunk"]),
            n_chunks=args.n_chunks,
            load_lora_path=phase_load_lora,
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
            oracle_model_name=args.oracle_model_name,
            target_models=target_models,
        )
        lora_path = os.path.join(save_dir, "final")

    final_lora = os.path.join(save_dir, "final")
    all_target_scores: dict[str, dict[str, float]] = {}
    for target_model_name, loaders in zip(target_models, loaders_by_target):
        eval_cfg = SelfInterpTrainingConfig(
            oracle_model_name=args.oracle_model_name,
            target_model_name=target_model_name,
            wandb_suffix=args.wandb_suffix,
        )
        eval_cfg.eval_batch_size = (train_batch_size * world_size) * 8
        eval_cfg.hook_onto_layer = args.hook_onto_layer
        eval_cfg.layer_percents = args.layer_percents
        eval_cfg.finalize(dataset_loaders=loaders)
        all_target_scores[target_model_name] = _eval_target(
            cfg=eval_cfg,
            target_model_name=target_model_name,
            lora_path=final_lora,
            loop_dataset_loaders=loaders,
            tokenizer=tokenizer,
            device=device,
            dtype=dtype,
            model_kwargs=model_kwargs,
            local_rank=local_rank,
            phase_label=f"final_eval_target_{target_model_name.split('/')[-1]}",
        )

    if local_rank == 0 and all_target_scores:
        all_scores = [score for scores in all_target_scores.values() for score in scores.values()]
        grand_average = sum(all_scores) / len(all_scores) if all_scores else float("nan")
        print(f"\nGRAND AVERAGE (all {len(target_models)} targets, all datasets): {grand_average:.4f}\n", flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()