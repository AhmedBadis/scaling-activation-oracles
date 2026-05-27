"""
Classification evaluation for locally-trained oracle checkpoints.

Supports:
  - Same-model oracles  (oracle == target, no projector)
  - Cross-model oracles (oracle != target, projector loaded from .pth)

For dual-target checkpoints the oracle is evaluated against each target separately.

Usage (normally via the .sbatch runner):
    python experiments/classification_eval_oracles.py
"""

from __future__ import annotations

import gc
import glob
import json
import os
from dataclasses import dataclass
from typing import Any

import torch
from peft import LoraConfig
from transformers import BitsAndBytesConfig

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from nl_probes.dataset_classes.classification import (
    ClassificationDatasetConfig,
    ClassificationDatasetLoader,
)
from nl_probes.dataset_classes.act_dataset_manager import DatasetLoaderConfig
from nl_probes.utils.eval import run_evaluation
from nl_probes.utils.activation_utils import get_hf_submodule
from nl_probes.utils.common import load_model, load_tokenizer
from nl_probes.utils.lora_utils import sanitize_lora_name

# ── constants (mirror classification_eval.py) ────────────────────────────────

INJECTION_LAYER = 1
DTYPE = torch.bfloat16
BASE_BATCH_SIZE = 256
STEERING_COEFFICIENT = 1.0
GENERATION_KWARGS = {"do_sample": False, "temperature": 0.0, "max_new_tokens": 10}
SINGLE_TOKEN_MODE = True
LAYER_PERCENTS = [25, 50, 75]
MAIN_TEST_SIZE = 250

CLASSIFICATION_DATASETS: dict[str, dict[str, Any]] = {
    "geometry_of_truth":        {"num_train": 0, "num_test": MAIN_TEST_SIZE, "splits": ["test"]},
    "relations":                {"num_train": 0, "num_test": MAIN_TEST_SIZE, "splits": ["test"]},
    "sst2":                     {"num_train": 0, "num_test": MAIN_TEST_SIZE, "splits": ["test"]},
    "md_gender":                {"num_train": 0, "num_test": MAIN_TEST_SIZE, "splits": ["test"]},
    "snli":                     {"num_train": 0, "num_test": MAIN_TEST_SIZE, "splits": ["test"]},
    "ag_news":                  {"num_train": 0, "num_test": MAIN_TEST_SIZE, "splits": ["test"]},
    "ner":                      {"num_train": 0, "num_test": MAIN_TEST_SIZE, "splits": ["test"]},
    "tense":                    {"num_train": 0, "num_test": MAIN_TEST_SIZE, "splits": ["test"]},
    "language_identification":  {"num_train": 0, "num_test": MAIN_TEST_SIZE, "splits": ["test"]},
    "singular_plural":          {"num_train": 0, "num_test": MAIN_TEST_SIZE, "splits": ["test"]},
}

EXPERIMENTS_DIR = "experiments/classification_oracle_eval"

# ── oracle configurations ────────────────────────────────────────────────────

@dataclass
class OracleConfig:
    label: str                  # human-readable run name
    oracle_model: str           # HF model id for the oracle
    targets: list[str]          # HF model id(s) for activation source
    checkpoint_dir: str         # local dir containing final/ and optionally *.pth


ORACLE_CONFIGS: list[OracleConfig] = [
    OracleConfig(
        label="o1.7B_t1.7B",
        oracle_model="Qwen/Qwen3-1.7B",
        targets=["Qwen/Qwen3-1.7B"],
        checkpoint_dir="checkpoints_latentqa_cls_past_lens_oracle_Qwen3-1.7B_target_Qwen3-1.7B",
    ),
    OracleConfig(
        label="o1.7B_t4B",
        oracle_model="Qwen/Qwen3-1.7B",
        targets=["Qwen/Qwen3-4B"],
        checkpoint_dir="checkpoints_latentqa_cls_past_lens_oracle_Qwen3-1.7B_target_Qwen3-4B",
    ),
    OracleConfig(
        label="o1.7B_tT4B",
        oracle_model="Qwen/Qwen3-1.7B",
        targets=["Qwen/Qwen3.5-4B"],
        checkpoint_dir="checkpoints_latentqa_cls_past_lens_oracle_Qwen3-1.7B_target_Qwen3.5-4B",
    ),
    OracleConfig(
        label="o4B_t4B",
        oracle_model="Qwen/Qwen3-4B",
        targets=["Qwen/Qwen3-4B"],
        checkpoint_dir="checkpoints_latentqa_cls_past_lens_oracle_Qwen3-4B_target_Qwen3-4B",
    ),
    OracleConfig(
        label="oT4B_t4B",
        oracle_model="Qwen/Qwen3.5-4B",
        targets=["Qwen/Qwen3-4B"],
        checkpoint_dir="checkpoints_latentqa_cls_past_lens_oracle_Qwen3.5-4B_target_Qwen3-4B",
    ),
    # Dual-target: eval against each target separately
    OracleConfig(
        label="phased_dual_o1.7_t4_T4",
        oracle_model="Qwen/Qwen3-1.7B",
        targets=["Qwen/Qwen3-4B", "Qwen/Qwen3.5-4B"],
        checkpoint_dir="checkpoints_phased_dual_o1.7_t4_T4",
    ),
    OracleConfig(
        label="sequential_dual_o1.7_t4_T4",
        oracle_model="Qwen/Qwen3-1.7B",
        targets=["Qwen/Qwen3-4B", "Qwen/Qwen3.5-4B"],
        checkpoint_dir="checkpoints_sequential_dual_o1.7_t4_T4",
    ),
]

# ── helpers ──────────────────────────────────────────────────────────────────

def _model_kwargs(model_name: str) -> dict:
    return {"attn_implementation": "sdpa"}


def _find_latest_pth(checkpoint_dir: str) -> str | None:
    """Return the .pth with the highest step number, or None."""
    pths = glob.glob(os.path.join(checkpoint_dir, "checkpoint_step_*.pth"))
    if not pths:
        pths = glob.glob(os.path.join(checkpoint_dir, "checkpoint_*.pth"))
    if not pths:
        return None
    def _step(p: str) -> int:
        base = os.path.basename(p)
        parts = base.replace(".pth", "").split("_")
        for part in reversed(parts):
            if part.isdigit():
                return int(part)
        return 0
    return max(pths, key=_step)


def _load_projector(
    checkpoint_dir: str,
    target_hidden: int,
    oracle_hidden: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.nn.Linear | None:
    if target_hidden == oracle_hidden:
        return None
    pth = _find_latest_pth(checkpoint_dir)
    if pth is None:
        raise FileNotFoundError(
            f"No .pth checkpoint found in {checkpoint_dir} but hidden sizes differ "
            f"({target_hidden} -> {oracle_hidden}). Cannot build projector."
        )
    print(f"Loading projector from {pth}")
    ckpt = torch.load(pth, map_location=device, weights_only=False)
    proj_sd = ckpt["module_states"]["activation_projector"]
    weight = proj_sd["weight"]
    has_bias = "bias" in proj_sd
    proj = torch.nn.Linear(target_hidden, oracle_hidden, bias=has_bias, device=device, dtype=dtype)
    proj.load_state_dict(proj_sd)
    proj.eval()
    return proj


def _load_eval_data(
    oracle_model: str,
    target_model: str,
    layer_percent: int,
    model_kwargs: dict,
    target_model_obj,
    oracle_model_obj,
) -> dict[str, list[Any]]:
    """
    Load classification eval data for cross-model evaluation.

    TrainingDataPoint has two token ID fields with different requirements:
      - input_ids: fed to oracle via construct_batch → must use oracle tokenizer
      - context_input_ids: fed to target in materialize_missing_steering_vectors → must use target tokenizer

    We load the dataset twice and merge: input_ids from oracle-tokenized load,
    context_input_ids/context_positions/layer/steering_vectors from target-tokenized load.
    """
    def _make_loaders(model_name: str, save_acts: bool) -> list:
        loaders = []
        for ds_name, dcfg in CLASSIFICATION_DATASETS.items():
            bs = BASE_BATCH_SIZE // 8 if "language_identification" in ds_name else BASE_BATCH_SIZE
            cls_cfg = ClassificationDatasetConfig(
                classification_dataset_name=ds_name,
                max_end_offset=-3, min_end_offset=-3,
                max_window_size=1, min_window_size=1,
            ) if SINGLE_TOKEN_MODE else ClassificationDatasetConfig(
                classification_dataset_name=ds_name,
                max_end_offset=-1, min_end_offset=-1,
                max_window_size=50, min_window_size=50,
            )
            loader_cfg = DatasetLoaderConfig(
                custom_dataset_params=cls_cfg,
                num_train=dcfg["num_train"],
                num_test=dcfg["num_test"],
                splits=dcfg["splits"],
                model_name=model_name,
                layer_percents=[layer_percent],
                save_acts=save_acts,
                batch_size=bs,
            )
            loaders.append(ClassificationDatasetLoader(
                dataset_config=loader_cfg,
                model_kwargs=model_kwargs,
                model=target_model_obj if model_name == target_model else oracle_model_obj,
            ))
        return loaders

    # Load with target tokenizer: provides context_input_ids/context_positions/steering_vectors
    target_loaders = _make_loaders(target_model, save_acts=True)
    # Load with oracle tokenizer: provides input_ids/labels in oracle vocab
    oracle_loaders = _make_loaders(oracle_model, save_acts=False)

    eval_data: dict[str, list[Any]] = {}
    for t_loader, o_loader in zip(target_loaders, oracle_loaders):
        if "test" not in t_loader.dataset_config.splits:
            continue
        ds_id = t_loader.dataset_config.dataset_name
        if ds_id.startswith("classification_"):
            ds_id = ds_id[len("classification_"):]

        target_points = t_loader.load_dataset("test")
        oracle_points = o_loader.load_dataset("test")

        assert len(target_points) == len(oracle_points), (
            f"Dataset size mismatch for {ds_id}: target={len(target_points)} oracle={len(oracle_points)}"
        )

        merged = []
        for tp, op in zip(target_points, oracle_points):
            # Use oracle-tokenized input_ids/labels for oracle generation
            # Use target-tokenized context fields for activation collection
            merged_point = op.model_copy(deep=False)
            merged_point = merged_point.model_copy(update={
                "context_input_ids": tp.context_input_ids,
                "context_positions": tp.context_positions,
                "layer": tp.layer,
                "steering_vectors": tp.steering_vectors,
            })
            merged.append(merged_point)

        eval_data[ds_id] = merged
    return eval_data


# ── main evaluation loop ─────────────────────────────────────────────────────

def main() -> None:
    os.makedirs(EXPERIMENTS_DIR, exist_ok=True)
    device = torch.device("cuda")
    mode_str = "single_token" if SINGLE_TOKEN_MODE else "multi_token"

    for cfg in ORACLE_CONFIGS:
        oracle_lora_path = os.path.join(cfg.checkpoint_dir, "final")
        if not os.path.isdir(oracle_lora_path):
            print(f"[SKIP] {cfg.label}: final/ not found at {oracle_lora_path}")
            continue

        print(f"\n{'=' * 70}")
        print(f"Oracle: {cfg.label}  ({cfg.oracle_model})")
        print(f"{'=' * 70}")

        oracle_kwargs = _model_kwargs(cfg.oracle_model)
        oracle_tokenizer = load_tokenizer(cfg.oracle_model)
        oracle_model = load_model(cfg.oracle_model, DTYPE, **oracle_kwargs)
        oracle_san = sanitize_lora_name(oracle_lora_path)
        oracle_model.load_adapter(oracle_lora_path, adapter_name=oracle_san, is_trainable=False)
        oracle_model.set_adapter(oracle_san)
        oracle_model.eval()
        oracle_submodule = get_hf_submodule(oracle_model, INJECTION_LAYER)

        for target_model_name in cfg.targets:
            target_tag = target_model_name.split("/")[-1].replace(".", "_")
            print(f"\n  Target: {target_model_name}")

            target_kwargs = _model_kwargs(target_model_name)
            target_tokenizer = load_tokenizer(target_model_name)
            target_model_obj = load_model(target_model_name, DTYPE, **target_kwargs)
            target_model_obj.eval()

            projector = _load_projector(
                cfg.checkpoint_dir,
                target_hidden=target_model_obj.config.hidden_size,
                oracle_hidden=oracle_model.config.hidden_size,
                device=device,
                dtype=DTYPE,
            )

            for layer_percent in LAYER_PERCENTS:
                print(f"    layer_percent={layer_percent}")
                run_dir = os.path.join(
                    EXPERIMENTS_DIR,
                    f"{cfg.label}_target_{target_tag}_{mode_str}_layer{layer_percent}",
                )
                os.makedirs(run_dir, exist_ok=True)
                out_path = os.path.join(run_dir, "results.json")
                if os.path.isfile(out_path):
                    print(f"    [SKIP] already exists: {out_path}")
                    continue

                eval_data = _load_eval_data(
                    oracle_model=cfg.oracle_model,
                    target_model=target_model_name,
                    layer_percent=layer_percent,
                    model_kwargs=target_kwargs,
                    target_model_obj=target_model_obj,
                    oracle_model_obj=oracle_model,
                )

                records = []
                for ds_id, data in eval_data.items():
                    raw = run_evaluation(
                        eval_data=data,
                        model=oracle_model,
                        tokenizer=oracle_tokenizer,
                        submodule=oracle_submodule,
                        device=device,
                        dtype=DTYPE,
                        global_step=-1,
                        lora_path=oracle_san,
                        eval_batch_size=BASE_BATCH_SIZE,
                        steering_coefficient=STEERING_COEFFICIENT,
                        generation_kwargs=GENERATION_KWARGS,
                        activation_tokenizer=target_tokenizer,
                        activation_model=target_model_obj,
                        projection_module=projector,
                    )
                    for result, dp in zip(raw, data, strict=True):
                        records.append({
                            "dataset_id": ds_id,
                            "ground_truth": result.api_response,
                            "target": dp.target_output,
                        })

                meta = {
                    "oracle_label": cfg.label,
                    "oracle_model": cfg.oracle_model,
                    "target_model": target_model_name,
                    "layer_percent": layer_percent,
                    "injection_layer": INJECTION_LAYER,
                    "single_token_mode": SINGLE_TOKEN_MODE,
                    "steering_coefficient": STEERING_COEFFICIENT,
                    "generation_kwargs": GENERATION_KWARGS,
                    "checkpoint_dir": cfg.checkpoint_dir,
                }
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump({"meta": meta, "records": records}, f, indent=2)
                print(f"    Saved: {out_path}")

            del target_model_obj, projector, target_tokenizer
            torch.cuda.empty_cache()
            gc.collect()

        del oracle_model, oracle_tokenizer
        torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    main()