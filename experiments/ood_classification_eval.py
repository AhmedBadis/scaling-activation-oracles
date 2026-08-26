"""
OOD classification evaluation for trained oracle LoRAs.

Evaluates each oracle from the manifest on classification datasets that were
NOT seen during training (engels_* datasets). Handles same-model, cross-model
(oracle != activation model), and multi-target (dual/triple-target trained
oracle, evaluated against each of its targets separately) setups, including
projector loading.

Run:
    python experiments/ood_classification_eval.py
or via Slurm:
    sbatch runner/ood_classification_eval.sbatch

Output: experiments/ood_classification_eval/{oracle_id}/{target_tag}/results.json
Summarize: python experiments/summarize_ood_classification_eval.py
"""

from __future__ import annotations

import gc
import json
import os
import traceback
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig
from transformers import BitsAndBytesConfig

from nl_probes.dataset_classes.act_dataset_manager import DatasetLoaderConfig
from nl_probes.dataset_classes.classification import (
    ClassificationDatasetConfig,
    ClassificationDatasetLoader,
)
from nl_probes.utils.activation_utils import get_hf_submodule
from nl_probes.utils.checkpointing import find_latest_checkpoint
from nl_probes.utils.common import load_model, load_tokenizer
from nl_probes.utils.eval import run_evaluation, score_eval_responses
from nl_probes.utils.hf_config import pretrained_hidden_size
from nl_probes.utils.lora_utils import sanitize_lora_name

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = REPO_ROOT / "experiments" / "ood_classification_eval"
_LOG = "OOD_CLS_EVAL"


def _banner(**parts: str) -> None:
    print(_LOG + " " + " ".join(f"{k}={v}" for k, v in sorted(parts.items())), flush=True)


# ---------------------------------------------------------------------------
# OOD datasets  (none of these appear in train_dual_target.py's mixture)
# ---------------------------------------------------------------------------
OOD_DATASETS: dict[str, dict[str, Any]] = {
    "engels_headline_istrump":      {"num_train": 0, "num_test": 250, "splits": ["test"]},
    "engels_headline_isobama":      {"num_train": 0, "num_test": 250, "splits": ["test"]},
    "engels_headline_ischina":      {"num_train": 0, "num_test": 250, "splits": ["test"]},
    "engels_hist_fig_ismale":       {"num_train": 0, "num_test": 250, "splits": ["test"]},
    "engels_news_class_politics":   {"num_train": 0, "num_test": 250, "splits": ["test"]},
    "engels_wikidata_isjournalist": {"num_train": 0, "num_test": 250, "splits": ["test"]},
    "engels_wikidata_isathlete":    {"num_train": 0, "num_test": 250, "splits": ["test"]},
    "engels_wikidata_ispolitician": {"num_train": 0, "num_test": 250, "splits": ["test"]},
    "engels_wikidata_issinger":     {"num_train": 0, "num_test": 250, "splits": ["test"]},
    "engels_wikidata_isresearcher": {"num_train": 0, "num_test": 250, "splits": ["test"]},
}

# ---------------------------------------------------------------------------
# Oracle manifest  (same as oracle_ood_eval.py)
#
# `activation_model_names` is a LIST because a single trained oracle LoRA
# (+ single shared projector, when one exists) can be evaluated against
# every target it was trained on. The dual/triple-target training scripts
# already enforce that co-trained targets share one hidden width (so one
# projector is valid for all of them) -- see train_phased_dual_target.py's
# startup check comparing pretrained_hidden_size(target_a/b/c). We reuse
# that guarantee here: one oracle load + one LoRA load + one projector load
# per manifest row, looped over N activation models for eval/scoring only.
# ---------------------------------------------------------------------------
ORACLE_MANIFEST: list[dict[str, Any]] = [
    {
        "id": "o1.7-t4",
        "oracle_model_name": "Qwen/Qwen3-1.7B",
        "activation_model_names": ["Qwen/Qwen3-4B"],
        "verbalizer_lora": "checkpoints_latentqa_cls_past_lens_oracle_Qwen3-1.7B_target_Qwen3-4B/final",
        "projector_save_dir": "checkpoints_latentqa_cls_past_lens_oracle_Qwen3-1.7B_target_Qwen3-4B",
    },
    {
        "id": "o1.7-t1.7",
        "oracle_model_name": "Qwen/Qwen3-1.7B",
        "activation_model_names": ["Qwen/Qwen3-1.7B"],
        "verbalizer_lora": "checkpoints_latentqa_cls_past_lens_oracle_Qwen3-1.7B_target_Qwen3-1.7B/final",
        "projector_save_dir": "",
    },
    {
        "id": "o1.7-T4",
        "oracle_model_name": "Qwen/Qwen3-1.7B",
        "activation_model_names": ["Qwen/Qwen3.5-4B"],
        "verbalizer_lora": "checkpoints_latentqa_cls_past_lens_oracle_Qwen3-1.7B_target_Qwen3.5-4B/final",
        "projector_save_dir": "checkpoints_latentqa_cls_past_lens_oracle_Qwen3-1.7B_target_Qwen3.5-4B",
    },
    {
        "id": "o4-t4",
        "oracle_model_name": "Qwen/Qwen3-4B",
        "activation_model_names": ["Qwen/Qwen3-4B"],
        "verbalizer_lora": "checkpoints_latentqa_cls_past_lens_oracle_Qwen3-4B_target_Qwen3-4B/final",
        "projector_save_dir": "",
    },
    {
        "id": "O4-t4",
        "oracle_model_name": "Qwen/Qwen3.5-4B",
        "activation_model_names": ["Qwen/Qwen3-4B"],
        "verbalizer_lora": "checkpoints_latentqa_cls_past_lens_oracle_Qwen3.5-4B_target_Qwen3-4B/final",
        "projector_save_dir": "checkpoints_latentqa_cls_past_lens_oracle_Qwen3.5-4B_target_Qwen3-4B",
    },
    {
        "id": "o1.7-dual-t4-T4",
        "oracle_model_name": "Qwen/Qwen3-1.7B",
        "activation_model_names": ["Qwen/Qwen3-4B", "Qwen/Qwen3.5-4B"],
        "verbalizer_lora": "checkpoints_phased_dual_o1.7_t4_T4/final",
        "projector_save_dir": "checkpoints_phased_dual_o1.7_t4_T4",
    },
    {
        "id": "o1.7-triple-t4-T4-t4i",
        "oracle_model_name": "Qwen/Qwen3-1.7B",
        "activation_model_names": ["Qwen/Qwen3-4B", "Qwen/Qwen3.5-4B", "Qwen/Qwen3-4B-Instruct-2507"],
        "verbalizer_lora": "checkpoints_phased_triple_o1.7_t4T4t4i/final",     
        "projector_save_dir": "checkpoints_phased_triple_o1.7_t4T4t4i",
    },
]

# ---------------------------------------------------------------------------
# Eval settings
# ---------------------------------------------------------------------------
INJECTION_LAYER = 1
DTYPE = torch.bfloat16
EVAL_BATCH_SIZE = 64
STEERING_COEFFICIENT = 1.0
GENERATION_KWARGS = {"do_sample": False, "temperature": 0.0, "max_new_tokens": 10}
LAYER_PERCENTS = [25, 50, 75]
SINGLE_TOKEN_MODE = True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve(p: str) -> str:
    if not p or p.startswith("/"):
        return p
    return str(REPO_ROOT / p)


def _target_tag(model_name: str) -> str:
    """Filesystem-safe label for a target model, used as an output subdir."""
    return model_name.replace("/", "__")


def _get_model_kwargs(model_name: str) -> dict:
    if model_name == "meta-llama/Llama-3.3-70B-Instruct":
        bnb_config = BitsAndBytesConfig(load_in_8bit=True, bnb_8bit_compute_dtype=torch.bfloat16)
        return {"quantization_config": bnb_config, "attn_implementation": "sdpa"}
    return {"attn_implementation": "sdpa"}


def _load_projector(
    oracle_name: str, act_name: str, proj_save_dir: str | None
) -> torch.nn.Module | None:
    ow = pretrained_hidden_size(oracle_name)
    aw = pretrained_hidden_size(act_name)
    if ow == aw:
        return None
    if not proj_save_dir:
        raise ValueError(
            f"oracle={oracle_name} act={act_name} differ in hidden size but projector_save_dir is empty"
        )
    ck = find_latest_checkpoint(proj_save_dir)
    if ck is None:
        raise FileNotFoundError(f"No checkpoint_*.pth under {proj_save_dir}")
    blob = torch.load(ck, map_location="cpu", weights_only=False)
    ms = blob.get("module_states") or {}
    sd = ms.get("activation_projector")
    if sd is None:
        for k, v in ms.items():
            if k.startswith("activation_projector") and v is not None:
                sd = v
                break
    if sd is None:
        raise KeyError(f"No activation_projector key in {ck}")
    use_bias = any(k.endswith("bias") for k in sd)
    lin = torch.nn.Linear(aw, ow, bias=use_bias)
    lin.load_state_dict(sd)
    return lin.to(device="cuda", dtype=torch.bfloat16)


def _canonical(name: str) -> str:
    return name[len("classification_"):] if name.startswith("classification_") else name


def _load_ood_eval_data(
    oracle_model_name: str,
    act_model_name: str,
    act_model,
    act_model_kwargs: dict,
    layer_percent: int,
) -> dict[str, list]:
    """Load OOD test sets.

    model_name      = activation model → context_tokenizer + act collection use act model
    prompt_model_name = oracle model   → prompt_tokenizer uses oracle vocab (fed into oracle)

    This means:
      context_input_ids  → activation-model vocab  → steering vectors at act-model hidden size
      input_ids          → oracle vocab             → fed into oracle model
    """
    loaders = []
    for ds_name, meta in OOD_DATASETS.items():
        bs = EVAL_BATCH_SIZE // 8 if "language_identification" in ds_name else EVAL_BATCH_SIZE
        if SINGLE_TOKEN_MODE:
            cls_cfg = ClassificationDatasetConfig(
                classification_dataset_name=ds_name,
                max_end_offset=-3,
                min_end_offset=-3,
                max_window_size=1,
                min_window_size=1,
            )
        else:
            cls_cfg = ClassificationDatasetConfig(
                classification_dataset_name=ds_name,
                max_end_offset=-1,
                min_end_offset=-1,
                max_window_size=50,
                min_window_size=50,
            )
        dl_cfg = DatasetLoaderConfig(
            custom_dataset_params=cls_cfg,
            num_train=0,
            num_test=meta["num_test"],
            splits=["test"],
            model_name=act_model_name,          # context tokenizer + act collection
            prompt_model_name=oracle_model_name, # prompt tokenizer (oracle vocab)
            layer_percents=[layer_percent],
            save_acts=True,
            batch_size=bs,
        )
        loaders.append(
            ClassificationDatasetLoader(
                dataset_config=dl_cfg, model_kwargs=act_model_kwargs, model=act_model
            )
        )

    eval_data: dict[str, list] = {}
    for loader in loaders:
        ds_id = _canonical(loader.dataset_config.dataset_name)
        eval_data[ds_id] = loader.load_dataset("test")
    return eval_data


def _eval_one_target(
    *,
    oid: str,
    o_name: str,
    a_name: str,
    o_model,
    o_tok,
    a_model,
    a_tok,
    a_kwargs: dict,
    submodule,
    lora_san: str,
    projector: torch.nn.Module | None,
    out_dir: Path,
) -> None:
    """Run the full layer x dataset eval sweep for one (oracle, target) pair
    and write its own results.json under out_dir."""
    all_records: list[dict] = []

    for layer_percent in LAYER_PERCENTS:
        _banner(oracle=oid, target=a_name, status="loading_data", layer_percent=str(layer_percent))
        eval_data = _load_ood_eval_data(o_name, a_name, a_model, a_kwargs, layer_percent)

        for ds_id, data in eval_data.items():
            if not data:
                _banner(oracle=oid, target=a_name, ds=ds_id, layer=str(layer_percent), status="empty_skip")
                continue

            raw = run_evaluation(
                eval_data=data,
                model=o_model,
                tokenizer=o_tok,
                submodule=submodule,
                device=torch.device("cuda"),
                dtype=DTYPE,
                global_step=-1,
                lora_path=lora_san,
                eval_batch_size=EVAL_BATCH_SIZE,
                steering_coefficient=STEERING_COEFFICIENT,
                generation_kwargs=GENERATION_KWARGS,
                activation_tokenizer=a_tok,
                activation_model=a_model,
                projection_module=projector,
            )

            _fmt, acc = score_eval_responses(raw, data)
            _banner(
                oracle=oid,
                target=a_name,
                ds=ds_id,
                layer=str(layer_percent),
                acc=f"{acc:.4f}",
                fmt=f"{_fmt:.4f}",
            )

            for result, dp in zip(raw, data, strict=True):
                all_records.append(
                    {
                        "layer_percent": layer_percent,
                        "dataset_id": ds_id,
                        "ground_truth": result.api_response,
                        "target": dp.target_output,
                    }
                )

    target_dir = out_dir / _target_tag(a_name)
    target_dir.mkdir(parents=True, exist_ok=True)
    out_path = target_dir / "results.json"
    out_path.write_text(
        json.dumps(
            {
                "meta": {
                    "oracle_label": oid,
                    "oracle_model": o_name,
                    "target_model": a_name,
                    "cross_model": o_name != a_name,
                    "layer_percents": LAYER_PERCENTS,
                    "single_token_mode": SINGLE_TOKEN_MODE,
                    "generation_kwargs": GENERATION_KWARGS,
                },
                "records": all_records,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _banner(oracle=oid, target=a_name, status="saved", path=str(out_path))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

    failures: list[str] = []

    for row in ORACLE_MANIFEST:
        oid = row["id"]
        o_name = row["oracle_model_name"]

        # Backward-compatible: accept either the old singular key or the new
        # list key, so hand-edited manifests from before this change still work.
        if "activation_model_names" in row:
            a_names = list(row["activation_model_names"])
        else:
            a_names = [row.get("activation_model_name", o_name)]

        verbal = _resolve(row.get("verbalizer_lora", ""))
        proj_dir = _resolve(row.get("projector_save_dir", ""))

        # --- validation ---
        if not verbal:
            _banner(oracle=oid, status="skip", reason="verbalizer_lora_empty")
            continue
        if not Path(verbal).exists():
            _banner(oracle=oid, status="skip", reason="verbalizer_lora_not_found", path=verbal)
            continue
        if not a_names:
            _banner(oracle=oid, status="skip", reason="activation_model_names_empty")
            continue

        # A single shared projector is only valid across all listed targets
        # if they all share one hidden width (the same constraint the
        # dual/triple-target training scripts enforce at start-up). Fail
        # fast here rather than silently mis-projecting a subset of targets.
        try:
            act_widths = {a: pretrained_hidden_size(a) for a in a_names}
        except Exception:
            _banner(oracle=oid, status="skip", reason="pretrained_hidden_size_lookup_failed")
            traceback.print_exc()
            failures.append(oid)
            continue
        if len(set(act_widths.values())) > 1:
            _banner(
                oracle=oid,
                status="skip",
                reason="activation_model_names_hidden_size_mismatch",
                widths=str(act_widths),
            )
            continue

        out_dir = EXPERIMENTS_DIR / oid
        out_dir.mkdir(parents=True, exist_ok=True)

        _banner(oracle=oid, status="start", o_model=o_name, targets=",".join(a_names))

        # Everything from here on touches the GPU / filesystem / network
        # (model loads, checkpoint loads, generation). Any one row failing
        # -- OOM on a big cross-model pair, a missing/partial checkpoint for
        # a target added later (e.g. the triple-target run), a flaky model
        # download -- must NOT abort the whole manifest loop and skip every
        # row that comes after it (this is what silently dropped dual's
        # second target and the whole triple row previously: one uncaught
        # exception here propagated out of main() and killed the process).
        o_model = None
        try:
            o_kwargs = _get_model_kwargs(o_name)

            # Projector depends only on (oracle hidden size, shared target
            # hidden size), so it is loaded once per row and reused across
            # every target.
            any_cross = any(a != o_name for a in a_names)
            projector = _load_projector(o_name, a_names[0], proj_dir) if any_cross else None

            # load oracle model + LoRA once for the whole row
            o_model = load_model(o_name, DTYPE, **o_kwargs)
            o_model.eval()
            o_model.add_adapter(LoraConfig(target_modules="all-linear"), adapter_name="default")
            o_tok = load_tokenizer(o_name)
            submodule = get_hf_submodule(o_model, INJECTION_LAYER)

            lora_san = sanitize_lora_name(verbal)
            o_model.load_adapter(verbal, adapter_name=lora_san, is_trainable=False)
            o_model.set_adapter(lora_san)

            for a_name in a_names:
                cross = a_name != o_name
                a_kwargs = _get_model_kwargs(a_name)
                a_model = None
                try:
                    if cross:
                        a_model = load_model(a_name, DTYPE, **a_kwargs)
                        a_model.eval()
                        a_tok = load_tokenizer(a_name)
                    else:
                        a_model = o_model
                        a_tok = o_tok

                    _eval_one_target(
                        oid=oid,
                        o_name=o_name,
                        a_name=a_name,
                        o_model=o_model,
                        o_tok=o_tok,
                        a_model=a_model,
                        a_tok=a_tok,
                        a_kwargs=a_kwargs,
                        submodule=submodule,
                        lora_san=lora_san,
                        projector=projector,
                        out_dir=out_dir,
                    )
                except Exception:
                    _banner(oracle=oid, target=a_name, status="error", action="skipping_target")
                    traceback.print_exc()
                    failures.append(f"{oid}:{a_name}")
                finally:
                    if cross and a_model is not None:
                        del a_model
                        gc.collect()
                        torch.cuda.empty_cache()
        except Exception:
            _banner(oracle=oid, status="error", action="skipping_row")
            traceback.print_exc()
            failures.append(oid)
        finally:
            # cleanup for the row, regardless of success/failure above
            if o_model is not None:
                if lora_san in getattr(o_model, "peft_config", {}):
                    o_model.delete_adapter(lora_san)
                del o_model
            gc.collect()
            torch.cuda.empty_cache()

    if failures:
        _banner(status="done_with_failures", failed=",".join(failures))
    else:
        _banner(status="done")


if __name__ == "__main__":
    main()