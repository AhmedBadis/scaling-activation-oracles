import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import gc
import json
import math
import random
import signal
import sys
from datetime import timedelta

# All necessary imports are now included above
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer, BitsAndBytesConfig
from transformers.optimization import get_linear_schedule_with_warmup
import torch.distributed as dist
import wandb

import nl_probes.dataset_classes.classification as classification
from nl_probes.utils.steering_hooks import (
    add_hook,
    get_hf_activation_steering_hook,
)
from nl_probes.configs.sft_config import SelfInterpTrainingConfig
from nl_probes.dataset_classes.act_dataset_manager import ActDatasetLoader, DatasetLoaderConfig
from nl_probes.dataset_classes.classification import (
    ClassificationDatasetConfig,
    ClassificationDatasetLoader,
)
from nl_probes.dataset_classes.latentqa_dataset import LatentQADatasetConfig, LatentQADatasetLoader
from nl_probes.dataset_classes.past_lens_dataset import PastLensDatasetConfig, PastLensDatasetLoader
from nl_probes.dataset_classes.sae_training_data import (
    SAEActivatingSequencesDatasetConfig,
    SAEActivatingSequencesDatasetLoader,
    SAEExplanationDatasetConfig,
    SAEExplanationDatasetLoader,
    SAEYesNoDatasetConfig,
    SAEYesNoDatasetLoader,
)
from nl_probes.utils.activation_utils import get_hf_submodule, get_text_only_lora_targets
from nl_probes.utils.common import load_model, load_tokenizer, set_seed
from nl_probes.utils.dataset_utils import (
    BatchData,
    EvalStepResult,
    FeatureResult,
    TrainingDataPoint,
    construct_batch,
    materialize_missing_steering_vectors,
)
from nl_probes.utils.eval import run_evaluation, score_eval_responses, parse_answer
from nl_probes.utils.checkpointing import (
    capture_rng_state,
    load_checkpoint_if_exists,
    save_checkpoint,
)
from nl_probes.utils.lora_utils import sanitize_lora_name


def activation_projector_key(target_model_name: str) -> str:
    """Per-target checkpoint key so each target gets its own activation_projector
    instead of all targets sharing one 'activation_projector' key."""
    return f"activation_projector__{sanitize_lora_name(target_model_name)}"


class GatedResidualMLPProjector(torch.nn.Module):
    """Stable nonlinear target-to-oracle activation projector.

    The learned linear path preserves the original projector's global alignment. A
    zero-initialized residual MLP learns nonlinear corrections without changing the
    initial function at startup.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        bias: bool = True,
        hidden_dim: int | None = None,
        residual_scale_init: float = 1e-2,
        eps: float = 1e-6,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if hidden_dim is None:
            hidden_dim = max(512, min(2048, 2 * min(input_dim, output_dim) // 3))

        self.base = torch.nn.Linear(input_dim, output_dim, bias=bias, device=device, dtype=dtype)
        self.down = torch.nn.Linear(input_dim, hidden_dim, bias=True, device=device, dtype=dtype)
        self.norm = torch.nn.RMSNorm(hidden_dim, eps=eps, device=device, dtype=dtype)
        self.up = torch.nn.Linear(hidden_dim, output_dim, bias=False, device=device, dtype=dtype)
        self.residual_scale = torch.nn.Parameter(
            torch.tensor(residual_scale_init, device=device, dtype=dtype)
        )

        # Start exactly as the established linear projector; the residual branch is
        # enabled smoothly as its zero-initialized output weights learn.
        torch.nn.init.zeros_(self.up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.residual_scale * self.up(
            torch.nn.functional.silu(self.norm(self.down(x)))
        )

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        # Migrate legacy checkpoints containing only Linear.weight/bias. The new
        # residual branch remains at its safe startup initialization.
        legacy_weight = prefix + "weight"
        legacy_bias = prefix + "bias"
        new_weight = prefix + "base.weight"
        new_bias = prefix + "base.bias"
        if legacy_weight in state_dict and new_weight not in state_dict:
            state_dict[new_weight] = state_dict.pop(legacy_weight)
            if self.base.bias is not None and legacy_bias in state_dict:
                state_dict[new_bias] = state_dict.pop(legacy_bias)
            for name, value in self.state_dict().items():
                key = prefix + name
                if key not in state_dict:
                    state_dict[key] = value.detach().clone()
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )


def make_activation_projector(
    input_dim: int,
    output_dim: int,
    *,
    bias: bool,
    device: torch.device,
    dtype: torch.dtype,
) -> GatedResidualMLPProjector:
    return GatedResidualMLPProjector(
        input_dim,
        output_dim,
        bias=bias,
        device=device,
        dtype=dtype,
    )


def _sync_module_grads(module: torch.nn.Module | None, world_size: int) -> None:
    """Average gradients for trainable modules that are not wrapped in DDP."""
    if module is None:
        return
    for parameter in module.parameters():
        if parameter.grad is not None:
            dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
            parameter.grad.div_(world_size)


def push_lora_to_hf(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    repo_id: str,
    private: bool,
    commit_message: str = "Upload LoRA adapter after training",
) -> None:
    """
    Push the trained LoRA adapter to Hugging Face Hub.

    Args:
        model: The trained model with LoRA adapters
        tokenizer: The tokenizer used with the model
        repo_id: HuggingFace repository ID (e.g., "username/repo-name")
        commit_message: Commit message for the upload
        private: Whether to make the repository private

    Returns:
        bool: True if successful, False otherwise
    """

    print(f"Pushing LoRA adapter to Hugging Face Hub: {repo_id}")

    # Get the original model name to copy config from
    original_model_name = model.config._name_or_path
    if hasattr(model, "base_model"):
        # For LoRA models, get the base model name
        original_model_name = model.base_model.config._name_or_path

    # Push the model (LoRA adapters)
    model.push_to_hub(
        repo_id=repo_id,
        commit_message=commit_message,
        private=private,
    )

    # Push the tokenizer as well
    tokenizer.push_to_hub(
        repo_id=repo_id,
        commit_message=f"Upload tokenizer - {commit_message}",
        private=private,
    )

    # Copy config.json from the original model
    try:
        import tempfile

        from huggingface_hub import hf_hub_download, upload_file

        print(f"Copying config.json from original model: {original_model_name}")

        # Download config.json from the original model
        with tempfile.NamedTemporaryFile(mode="w+b", suffix=".json", delete=False) as tmp_file:
            config_path = hf_hub_download(
                repo_id=original_model_name,
                filename="config.json",
                cache_dir=None,
                force_download=False,
            )

            # Copy the file content
            with open(config_path, "rb") as src:
                tmp_file.write(src.read())
            tmp_file.flush()

            # Upload to the LoRA repo
            upload_file(
                path_or_fileobj=tmp_file.name,
                path_in_repo="config.json",
                repo_id=repo_id,
                commit_message=f"Copy config.json from {original_model_name}",
            )

        # Clean up temp file
        os.unlink(tmp_file.name)
        print(f"Successfully copied config.json from {original_model_name}")

    except Exception as e:
        print(f"Warning: Failed to copy config.json from original model: {e}")
        print("LoRA adapter uploaded successfully, but without original model config")

    # Create and upload README with base model metadata
    try:
        print("Creating README with base model metadata...")

        readme_content = f"""---
base_model: {original_model_name}
library_name: peft
---

# LoRA Adapter for SAE Introspection

This is a LoRA (Low-Rank Adaptation) adapter trained for SAE (Sparse Autoencoder) introspection tasks.

## Base Model
- **Base Model**: `{original_model_name}`
- **Adapter Type**: LoRA
- **Task**: SAE Feature Introspection

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model and tokenizer
base_model = AutoModelForCausalLM.from_pretrained("{original_model_name}")
tokenizer = AutoTokenizer.from_pretrained("{original_model_name}")

# Load LoRA adapter
model = PeftModel.from_pretrained(base_model, "{repo_id}")
```

## Training Details
This adapter was trained using the lightweight SAE introspection training script to help the model understand and explain SAE features through activation steering.
"""

        # Create temporary README file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8") as tmp_readme:
            tmp_readme.write(readme_content)
            tmp_readme.flush()

            # Upload README to the LoRA repo
            upload_file(
                path_or_fileobj=tmp_readme.name,
                path_in_repo="README.md",
                repo_id=repo_id,
                commit_message="Add README with base model metadata",
            )

        # Clean up temp file
        os.unlink(tmp_readme.name)
        print("Successfully uploaded README with base model metadata")

    except Exception as e:
        print(f"Warning: Failed to upload README: {e}")
        print("LoRA adapter uploaded successfully, but without README")

    print(f"Successfully pushed LoRA adapter to: https://huggingface.co/{repo_id}")


def train_features_batch(
    cfg: SelfInterpTrainingConfig,
    training_batch: BatchData,
    model: AutoModelForCausalLM,
    submodule: torch.nn.Module,
    device: torch.device,
    dtype: torch.dtype,
    projection_module: torch.nn.Module | None = None,
) -> torch.Tensor:
    """
    Trains the model on a single batch of data.
    """

    batch_steering_vectors = training_batch.steering_vectors
    batch_positions = training_batch.positions

    # 3. Create and apply the activation steering hook
    hook_fn = get_hf_activation_steering_hook(
        vectors=batch_steering_vectors,
        positions=batch_positions,
        steering_coefficient=cfg.steering_coefficient,
        device=device,
        dtype=dtype,
        projection_module=projection_module,
    )

    tokenized_input = {
        "input_ids": training_batch.input_ids,
        "attention_mask": training_batch.attention_mask,
    }

    with add_hook(submodule, hook_fn):
        loss = model(**tokenized_input, labels=training_batch.labels).loss

    return loss


def eval_all_datasets(
    cfg: SelfInterpTrainingConfig,
    eval_datasets: dict[str, list[TrainingDataPoint]],
    model: AutoModelForCausalLM,
    tokenizer: PreTrainedTokenizer,
    activation_tokenizer: PreTrainedTokenizer,
    activation_model: AutoModelForCausalLM,
    submodule: torch.nn.Module,
    device: torch.device,
    dtype: torch.dtype,
    global_step: int,
    projection_module: torch.nn.Module | None = None,
) -> None:
    model.eval()
    eval_results = {}
    for ds in eval_datasets:
        eval_responses = run_evaluation(
            eval_data=eval_datasets[ds],
            model=model,
            tokenizer=tokenizer,
            activation_tokenizer=activation_tokenizer,
            activation_model=activation_model,
            submodule=submodule,
            device=device,
            dtype=dtype,
            global_step=global_step,
            lora_path=None,
            eval_batch_size=cfg.eval_batch_size,
            steering_coefficient=cfg.steering_coefficient,
            generation_kwargs=cfg.generation_kwargs,
            projection_module=projection_module,
        )
        
        """
        # DEBUG: Show oracle vs model responses
        if len(eval_responses) > 0 and len(eval_datasets[ds]) > 0:
            print(f"\n=== DEBUG EVAL {ds} at step {global_step} ===")
            for i in range(min(3, len(eval_responses), len(eval_datasets[ds]))):
                oracle_target = eval_datasets[ds][i].target_output
                model_response = eval_responses[i].api_response
                print(f"Example {i}:")
                print(f"  Oracle target: '{oracle_target}'")
                print(f"  Model response: '{model_response}'")
                print(f"  Parsed oracle: '{parse_answer(oracle_target)}'")
                print(f"  Parsed model: '{parse_answer(model_response)}'")
            print("=" * 50)
        """

        percent_format_correct, percent_ans_correct = score_eval_responses(eval_responses, eval_datasets[ds])
        eval_results[f"eval_format_correct/{ds}"] = percent_format_correct
        eval_results[f"eval_ans_correct/{ds}"] = percent_ans_correct
        print(f"Step {global_step} {ds} format correct: {percent_format_correct}, ans correct: {percent_ans_correct}")
    
    if wandb.run is not None:
        wandb.log(
            eval_results,
            step=global_step,
        )
        wandb.summary.update(eval_results)
    model.train()

    # Have occasionally seen OOMs on first training step after eval, so clear cache here
    torch.cuda.empty_cache()
    gc.collect()


def oom_preflight_check(
    cfg: SelfInterpTrainingConfig,
    training_data: list[TrainingDataPoint],
    model: AutoModelForCausalLM,
    submodule: torch.nn.Module,
    tokenizer: PreTrainedTokenizer,
    activation_tokenizer: PreTrainedTokenizer,
    activation_model: AutoModelForCausalLM,
    device: torch.device,
    dtype: torch.dtype,
    projection_module: torch.nn.Module | None = None,
) -> None:
    longest_prompt = max(training_data, key=lambda x: len(x.input_ids))
    long_prompts = [longest_prompt] * cfg.train_batch_size
    long_prompts = materialize_missing_steering_vectors(long_prompts, activation_tokenizer, activation_model)
    largest_possible_batch = construct_batch(long_prompts, tokenizer, device)

    dummy_optimizer_params = list(model.parameters())
    if projection_module is not None:
        dummy_optimizer_params.extend(projection_module.parameters())
    dummy_optimizer = torch.optim.AdamW(dummy_optimizer_params, lr=0.0)

    for _ in tqdm(range(3), desc="OOM preflight check"):
        loss = train_features_batch(
            cfg,
            largest_possible_batch,
            model,
            submodule,
            device,
            dtype,
            projection_module=projection_module,
        )
        loss.backward()
        dummy_optimizer.step()
        dummy_optimizer.zero_grad()

    del dummy_optimizer
    torch.cuda.empty_cache()
    gc.collect()

    print("OOM preflight check complete")






# -------------------- length_grouped_reorder --------------------
def length_grouped_reorder(
    data: list["TrainingDataPoint"],
    batch_size: int,
    window_mult: int,
) -> list["TrainingDataPoint"]:
    """
    Reorder data by grouping into mega-batches (window_mult * batch_size),
    sorting within each mega-batch by length descending, and flattening.
    This reduces padding waste while preserving some randomness across mega-batches.
    """
    lengths = [len(d.input_ids) for d in data]
    indices = list(range(len(data)))
    megabatch_size = window_mult * batch_size
    megabatches = [indices[i : i + megabatch_size] for i in range(0, len(indices), megabatch_size)]
    megabatches = [sorted(mb, key=lambda i: lengths[i], reverse=True) for mb in megabatches]
    new_order = [i for mb in megabatches for i in mb]
    return [data[i] for i in new_order]


# -------------------- train_model with checkpointing --------------------
def train_model(
    cfg: "SelfInterpTrainingConfig",
    training_data: list["TrainingDataPoint"],
    eval_datasets: dict[str, list["TrainingDataPoint"]],
    tokenizer: "PreTrainedTokenizer",
    device: torch.device,
    dtype: torch.dtype,
    model_kwargs: dict[str, Any],
    verbose: bool = False,
    save_checkpoint_on_complete: bool = False,
    checkpoint_complete_extra: dict[str, Any] | None = None,
    dual_target_phase_label: str | None = None,
):
    """
    Training loop with:
      - resume from latest checkpoint if present,
      - checkpoint on update steps every cfg.save_steps,
      - checkpoint on epoch boundaries every cfg.save_epoch_interval,
      - signal handlers for SIGTERM and SIGUSR1 to trigger immediate checkpoint.

    ``save_checkpoint_on_complete`` (dual-target training only): write a step checkpoint when this
    call's training slice finishes, so the next phase resumes from the phase boundary rather than
    the last periodic ``save_steps`` checkpoint mid-slice.

    ``dual_target_phase_label`` (dual-target only): cumulative ``global_step`` continues across
    phases, but batch skipping uses per-phase progress (see checkpoint ``extra``). Without this,
    ``global_step // steps_per_epoch`` can skip the entire training loop when resuming a completed
  phase slice.
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    model_kwargs = {**model_kwargs, "device_map": {"": f"cuda:{local_rank}"}}

    set_seed(cfg.seed)
    model = load_model(cfg.oracle_model_name, dtype, **model_kwargs)
    activation_tokenizer = tokenizer
    activation_model: AutoModelForCausalLM = model

    if cfg.oracle_model_name != cfg.target_model_name:
        if rank == 0:
            print(f"Loading target model for activation collection: {cfg.target_model_name}")
        activation_tokenizer = load_tokenizer(cfg.target_model_name)
        activation_model = load_model(cfg.target_model_name, dtype, **model_kwargs)
        activation_model.eval()

    target_width = activation_model.config.hidden_size
    oracle_width = model.config.hidden_size
    projector_key = activation_projector_key(cfg.target_model_name)
    activation_projector: torch.nn.Module | None = None
    if target_width != oracle_width:
        activation_projector = make_activation_projector(
            target_width,
            oracle_width,
            bias=cfg.activation_projection_bias,
            device=device,
            dtype=dtype,
        )
        if rank == 0:
            print(
                "Hidden-size mismatch detected between target activations and oracle residual stream. "
                f"Auto-enabling learned projection: target={cfg.target_model_name} ({target_width}) -> "
                f"oracle={cfg.oracle_model_name} ({oracle_width}), bias={cfg.activation_projection_bias}."
            )
    elif cfg.use_learned_activation_projection and rank == 0:
        print(
            "use_learned_activation_projection is set but hidden sizes already match "
            f"({target_width}); projection is not needed and will not be used."
        )

    model.enable_input_require_grads()

    if cfg.gradient_checkpointing:
        model.use_cache = False
        model.gradient_checkpointing_enable()

    submodule = get_hf_submodule(model, cfg.hook_onto_layer)

    if cfg.use_lora and cfg.load_lora_path is None:
        target_modules = cfg.lora_target_modules
        vlm_targets = get_text_only_lora_targets(cfg.model_name)
        if vlm_targets and target_modules == "all-linear":
            if rank == 0:
                print(f"VLM detected ({cfg.model_name}): excluding vision tower from LoRA")
            target_modules = vlm_targets

        # Handle Qwen3.5-4B PEFT naming issue
        if cfg.oracle_model_name == "Qwen/Qwen3.5-4B":
            lora_config = LoraConfig(
                r=cfg.lora_r, 
                lora_alpha=cfg.lora_alpha, 
                lora_dropout=cfg.lora_dropout,
                target_modules=cfg.lora_target_modules,
                bias='none',
                base_model_name_or_path=cfg.oracle_model_name  # Explicitly set to prevent incorrect mapping
            )
        else:
            lora_config = LoraConfig(
                r=cfg.lora_r, 
                lora_alpha=cfg.lora_alpha, 
                lora_dropout=cfg.lora_dropout,
                target_modules=cfg.lora_target_modules,
                bias='none'
            )
        model = get_peft_model(model, lora_config, autocast_adapter_dtype=True)
    elif cfg.load_lora_path is not None:
        load_lora_path = Path(cfg.load_lora_path)
        assert load_lora_path.exists()
        model = PeftModel.from_pretrained(model, load_lora_path, is_trainable=True, autocast_adapter_dtype=True)

    model.print_trainable_parameters()

    torch.cuda.set_device(local_rank)
    train_model_module: torch.nn.Module = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False
    )
    train_model_module.train()

    oom_preflight_check(
        cfg,
        training_data,
        model,
        submodule,
        tokenizer,
        activation_tokenizer,
        activation_model,
        device,
        dtype,
        projection_module=activation_projector,
    )

    set_seed(cfg.seed)

    optimizer_params = list(train_model_module.parameters())
    if activation_projector is not None:
        optimizer_params.extend(activation_projector.parameters())
    optimizer = torch.optim.AdamW(optimizer_params, lr=cfg.lr)

    global_step_size = cfg.train_batch_size * world_size
    effective_steps = (len(training_data) // global_step_size) * global_step_size
    if effective_steps != len(training_data):
        if rank == 0:
            print(f"Trimming training_data from {len(training_data)} to {effective_steps} for equal DDP steps")
        training_data = training_data[:effective_steps]

    if rank == 0:
        tokens_per_epoch_est = sum(len(dp.input_ids) for dp in training_data)
        total_training_tokens_est = tokens_per_epoch_est * cfg.num_epochs
        num_examples_pre_shard = len(training_data)

    # Optionally reorder by length to reduce padding
    if getattr(cfg, "length_grouping_window_mult", None):
        training_data = length_grouped_reorder(training_data, cfg.train_batch_size, cfg.length_grouping_window_mult)

    training_data = training_data[rank::world_size]

    num_batches_per_epoch = len(training_data) // cfg.train_batch_size
    batches_per_epoch = (num_batches_per_epoch // cfg.gradient_accumulation_steps) * cfg.gradient_accumulation_steps
    trimmed_examples = batches_per_epoch * cfg.train_batch_size
    if trimmed_examples != len(training_data) and rank == 0:
        print(
            f"Trimming per-rank training_data from {len(training_data)} to {trimmed_examples} "
            "to align with gradient_accumulation_steps"
        )
    training_data = training_data[:trimmed_examples]

    steps_per_epoch = batches_per_epoch // cfg.gradient_accumulation_steps
    assert steps_per_epoch > 0, "No optimizer steps will be run; check dataset/batch/accumulation sizes"
    total_training_steps = steps_per_epoch * cfg.num_epochs
    warmup_steps = int(total_training_steps * 0.1)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
    )

    # ---------------- checkpoint / resume ----------------
    global_step = 0
    start_epoch = 0

    modules_to_restore = {projector_key: activation_projector} if activation_projector is not None else None
    ckpt_global_step, ckpt_epoch, ckpt_extra = load_checkpoint_if_exists(
        cfg.save_dir,
        model,
        optimizer=optimizer,
        scheduler=scheduler,
        modules_to_restore=modules_to_restore,
    )
    phase_start_global_step = 0
    updates_done_in_epoch = 0
    if ckpt_global_step is not None:
        global_step = ckpt_global_step
        start_epoch = ckpt_epoch
        if rank == 0:
            print(f"Resuming from checkpoint: global_step={global_step}, epoch={start_epoch}")
    if dual_target_phase_label:
        start_epoch = 0
        phase_slice_complete = False
        if ckpt_global_step is not None:
            ckpt_phase = ckpt_extra.get("dual_target_phase")
            ckpt_phase_start = int(ckpt_extra.get("dual_target_phase_start_global_step", 0))
            phase_complete_flag = bool(ckpt_extra.get("dual_target_phase_complete", False))
            if ckpt_phase == dual_target_phase_label:
                phase_start_global_step = ckpt_phase_start
                updates_done_in_epoch = global_step - phase_start_global_step
                phase_slice_complete = phase_complete_flag or updates_done_in_epoch >= steps_per_epoch
                if phase_slice_complete:
                    updates_done_in_epoch = steps_per_epoch
            else:
                phase_start_global_step = global_step
                updates_done_in_epoch = 0
                scheduler = get_linear_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps=warmup_steps,
                    num_training_steps=total_training_steps,
                )
        else:
            phase_start_global_step = 0
            updates_done_in_epoch = 0
        if rank == 0:
            print(
                f"Dual-target phase {dual_target_phase_label}: "
                f"phase_start_global_step={phase_start_global_step} "
                f"updates_done_in_this_slice={updates_done_in_epoch} "
                f"steps_per_epoch={steps_per_epoch} "
                f"phase_slice_complete={phase_slice_complete}"
            )

    def _dual_target_ckpt_extra() -> dict[str, Any]:
        out: dict[str, Any] = {
            "dual_target_phase": dual_target_phase_label,
            "dual_target_phase_start_global_step": phase_start_global_step,
        }
        if checkpoint_complete_extra:
            out.update(checkpoint_complete_extra)
        return out

    # Re-wrap with DDP to ensure correct device mapping after loading weights
    train_model_module = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False
    )
    train_model_module.train()

    # Signal handler to trigger immediate checkpointing
    def _signal_handler(signum, frame):
        try:
            if rank == 0:
                print(f"Signal {signum} received on rank {rank}: saving checkpoint (global_step={global_step}, epoch={start_epoch})")
                rng = capture_rng_state()
                save_checkpoint(
                    cfg.save_dir,
                    model,
                    optimizer,
                    scheduler,
                    global_step,
                    start_epoch,
                    rng_state=rng,
                    extra=_dual_target_ckpt_extra() if dual_target_phase_label else {"signal": signum},
                    tag="step",
                    module_states=(
                        {projector_key: activation_projector.state_dict()}
                        if activation_projector is not None
                        else None
                    ),
                )
            dist.barrier()
        except Exception as e:
            print(f"Exception in signal handler: {e}", file=sys.stderr)
        if signum == signal.SIGTERM:
            sys.exit(0)

    try:
        signal.signal(signal.SIGTERM, _signal_handler)
        signal.signal(signal.SIGUSR1, _signal_handler)
    except Exception:
        # Some environments (e.g., certain distributed launchers) may not allow signal registration
        pass

    # Init Weights & Biases only on rank 0
    if rank == 0:
        wandb.init(project=cfg.wandb_project, name=cfg.wandb_run_name, config=asdict(cfg))
        wandb.summary["train/tokens_per_epoch_est"] = tokens_per_epoch_est
        wandb.summary["train/total_tokens_est"] = total_training_tokens_est
        wandb.summary["train/num_examples_pre_shard"] = num_examples_pre_shard
        
        """
        # DEBUG: Show training data examples at start
        print(f"\n=== DEBUG TRAINING DATA at step {global_step} ===")
        for i in range(min(3, len(training_data))):
            data_point = training_data[i]
            # Get prompt using target tokenizer (consistent with activation data)
            prompt_tokens = data_point.input_ids[:data_point.context_positions[0]] if data_point.context_positions else data_point.input_ids[:10]
            prompt = tokenizer.decode(prompt_tokens, skip_special_tokens=True)
            print(f"Training example {i}:")
            print(f"  Prompt: '{prompt[:100]}...'")
            print(f"  Target output: '{data_point.target_output}'")
            print(f"  Parsed target: '{parse_answer(data_point.target_output)}'")
            # DEBUG: Check raw input_ids for corruption
            if i == 0:  # Only for first example
                print(f"  DEBUG: Raw input_ids first 20: {data_point.input_ids[:20]}")
                print(f"  DEBUG: Decoded prompt: '{prompt[:100]}...'")
                print(f"  DEBUG: Context positions: {data_point.context_positions[:5]}")
                print(f"  DEBUG: Input_ids length: {len(data_point.input_ids)}")
        print("=" * 50)
        """

    if not dual_target_phase_label:
        resumed_updates = global_step
        if resumed_updates > 0:
            start_epoch = resumed_updates // steps_per_epoch
        if start_epoch == cfg.num_epochs:
            start_epoch = cfg.num_epochs - 1
            updates_done_in_epoch = steps_per_epoch

    for epoch in range(start_epoch, cfg.num_epochs):
        accumulated_loss = 0.0
        optimizer.zero_grad()

        # Optionally save at epoch start (useful for periodic epoch checkpoints)
        if getattr(cfg, "save_epoch_interval", None) and (epoch % cfg.save_epoch_interval == 0) and epoch != start_epoch:
            if rank == 0:
                rng = capture_rng_state()
                save_checkpoint(
                    cfg.save_dir,
                    model,
                    optimizer,
                    scheduler,
                    global_step,
                    epoch,
                    rng_state=rng,
                    tag="epoch",
                    module_states=(
                        {projector_key: activation_projector.state_dict()}
                        if activation_projector is not None
                        else None
                    ),
                )
            dist.barrier()

        if not dual_target_phase_label:
            updates_done_in_epoch = 0
            if epoch == start_epoch and global_step > 0:
                updates_done_in_epoch = global_step % steps_per_epoch

        batch_range = range(0, len(training_data), cfg.train_batch_size)
        batches_to_skip = updates_done_in_epoch * cfg.gradient_accumulation_steps if updates_done_in_epoch > 0 else 0

        for step_idx, start in enumerate(
            tqdm(
                batch_range,
                desc=f"Training epoch {epoch + 1}",
                disable=rank != 0,
            )
        ):
            if step_idx < batches_to_skip:
                continue

            t_batch_list: list["TrainingDataPoint"] = training_data[start : start + cfg.train_batch_size]

            t_batch_list = materialize_missing_steering_vectors(t_batch_list, activation_tokenizer, activation_model)
            t_batch = construct_batch(t_batch_list, tokenizer, device)

            loss = train_features_batch(
                cfg,
                t_batch,
                train_model_module,
                submodule,
                device,
                dtype,
                projection_module=activation_projector,
            )
            loss = loss / cfg.gradient_accumulation_steps
            loss.backward()
            accumulated_loss += loss.item()

            is_update_step = (step_idx + 1) % cfg.gradient_accumulation_steps == 0

            if is_update_step:
                _sync_module_grads(activation_projector, world_size)
                clip_grad_norm_(
                    list(train_model_module.parameters())
                    + (list(activation_projector.parameters()) if activation_projector is not None else []),
                    cfg.max_grad_norm,
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                if rank == 0:
                    wandb.log(
                        {
                            "train/loss": accumulated_loss,
                            "train/learning_rate": scheduler.get_last_lr()[0],
                        },
                        step=global_step,
                    )
                    if verbose:
                        print(f"Step {global_step} loss: {accumulated_loss}")

                if global_step % cfg.eval_steps == 0 and (cfg.eval_on_start or global_step > 0):
                    if rank == 0:
                        eval_all_datasets(
                            cfg,
                            eval_datasets,
                            model,
                            tokenizer,
                            activation_tokenizer,
                            activation_model,
                            submodule,
                            device,
                            dtype,
                            global_step,
                            projection_module=activation_projector,
                        )
                    dist.barrier()

                # Checkpoint on update steps
                if getattr(cfg, "save_steps", None) and (global_step % cfg.save_steps == 0) and global_step > 0:
                    if rank == 0:
                        rng = capture_rng_state()
                        ckpt_path = save_checkpoint(
                            cfg.save_dir,
                            model,
                            optimizer,
                            scheduler,
                            global_step,
                            epoch,
                            rng_state=rng,
                            extra=_dual_target_ckpt_extra() if dual_target_phase_label else None,
                            tag="step",
                            module_states=(
                                {projector_key: activation_projector.state_dict()}
                                if activation_projector is not None
                                else None
                            ),
                        )
                        if verbose:
                            print(f"Saved checkpoint at {ckpt_path}")
                    dist.barrier()

                global_step += 1
                accumulated_loss = 0.0

        # End of epoch: optionally save an epoch checkpoint every cfg.save_epoch_interval epochs
        if getattr(cfg, "save_epoch_interval", None) and ((epoch + 1) % cfg.save_epoch_interval == 0):
            if rank == 0:
                rng = capture_rng_state()
                ckpt_path = save_checkpoint(
                    cfg.save_dir,
                    model,
                    optimizer,
                    scheduler,
                    global_step,
                    epoch + 1,
                    rng_state=rng,
                    tag="epoch",
                    module_states=(
                        {projector_key: activation_projector.state_dict()}
                        if activation_projector is not None
                        else None
                    ),
                )
                if verbose:
                    print(f"Saved epoch checkpoint at {ckpt_path}")
            dist.barrier()

    if save_checkpoint_on_complete:
        if rank == 0:
            rng = capture_rng_state()
            extra = _dual_target_ckpt_extra() if dual_target_phase_label else {"dual_target_phase_complete": True}
            if not dual_target_phase_label and checkpoint_complete_extra:
                extra.update(checkpoint_complete_extra)
            ckpt_path = save_checkpoint(
                cfg.save_dir,
                model,
                optimizer,
                scheduler,
                global_step,
                epoch,
                rng_state=rng,
                tag="step",
                extra=extra,
                module_states=(
                    {projector_key: activation_projector.state_dict()}
                    if activation_projector is not None
                    else None
                ),
            )
            print(f"Saved phase-boundary checkpoint at {ckpt_path} (global_step={global_step})")
        dist.barrier()

    # Finalize
    if rank == 0:
        print("Training complete.")
        print("Saving final model...")
        model.save_pretrained(f"{cfg.save_dir}/final")

        print("Running final evaluation...")
        eval_all_datasets(
            cfg,
            eval_datasets,
            model,
            tokenizer,
            activation_tokenizer,
            activation_model,
            submodule,
            device,
            dtype,
            global_step,
            projection_module=activation_projector,
        )
        wandb.finish()

        if cfg.hf_push_to_hub and cfg.hf_repo_id:
            print("Pushing LoRA adapter to Hugging Face Hub...")
            push_lora_to_hf(
                model=model,
                tokenizer=tokenizer,
                repo_id=cfg.hf_repo_id,
                commit_message=f"SAE introspection LoRA - {cfg.wandb_run_name} - final model",
                private=cfg.hf_private_repo,
            )
    dist.barrier()








def build_datasets(
    cfg: SelfInterpTrainingConfig,
    dataset_loaders: list[ActDatasetLoader],
    max_len_percentile: float | None = 0.999,
    window_mult: int | None = 20,
) -> tuple[list[TrainingDataPoint], dict[str, list[TrainingDataPoint]]]:
    set_seed(cfg.seed)
    all_training_data: list[TrainingDataPoint] = []
    # eval data will only be for classification datasets
    all_eval_data: dict[str, list[TrainingDataPoint]] = {}

    for dataset_loader in dataset_loaders:
        if "train" in dataset_loader.dataset_config.splits:
            train_data = dataset_loader.load_dataset("train")
            # print(f"DEBUG: Loaded {len(train_data)} items from {dataset_loader.dataset_config.dataset_name}")
            if len(train_data) > 0:
                sample = train_data[0]
                # print(f"DEBUG: Sample target_output: {repr(sample.target_output[:100] if hasattr(sample, 'target_output') else 'no target')}")
                # print(f"DEBUG: Sample context_input_ids length: {len(sample.context_input_ids) if hasattr(sample, 'context_input_ids') else 'no context'}")
            all_training_data.extend(train_data)
        if "test" in dataset_loader.dataset_config.splits:
            test_data = dataset_loader.load_dataset("test")
            # print(f"DEBUG: Loaded {len(test_data)} test items from {dataset_loader.dataset_config.dataset_name}")
            all_eval_data[dataset_loader.dataset_config.dataset_name] = test_data

    p = max_len_percentile
    if p is not None:
        if p >= 1.0 or p <= 0.0:
            raise ValueError("max_len_percentile must be less than 1.0 and greater than 0.0")

        lengths = sorted(len(td.input_ids) for td in all_training_data)
        median_length = lengths[len(lengths) // 2]
        print(f"Max length: {lengths[-1]}, Min length: {lengths[0]}, Median length: {median_length}")
        # Inclusive quantile index
        idx = int((len(lengths) - 1) * p)
        threshold = lengths[idx]

        before = len(all_training_data)
        all_training_data = [td for td in all_training_data if len(td.input_ids) <= threshold]
        removed = before - len(all_training_data)
        print(f"Percentile trim: kept <= {threshold} tokens (p={p:.6f}). Removed {removed}/{before} examples.")

    set_seed(cfg.seed)
    random.shuffle(all_training_data)

    if window_mult is not None:
        all_training_data = length_grouped_reorder(all_training_data, cfg.train_batch_size, window_mult)

    return all_training_data, all_eval_data


# Helper to cut repetition when building DatasetLoaderConfig
def mk_cfg(
    custom_params,
    *,
    num_train: int,
    num_test: int,
    splits: list[str],
    model_name: str,
    prompt_model_name: str | None,
    layer_percents: list[int],
    save_acts: bool,
    batch_size: int,
) -> DatasetLoaderConfig:
    return DatasetLoaderConfig(
        custom_dataset_params=custom_params,
        num_train=num_train,
        num_test=num_test,
        splits=splits,
        model_name=model_name,
        prompt_model_name=prompt_model_name,
        layer_percents=layer_percents,
        save_acts=save_acts,
        batch_size=batch_size,
    )


def build_loader_groups(
    *,
    model_name: str,
    prompt_model_name: str,
    layer_percents: list[int],
    act_collection_batch_size: int,
    save_acts: bool,
    classification_datasets: dict[str, dict[str, Any]],
    model_kwargs: dict[str, Any],
) -> dict[str, list[ActDatasetLoader]]:
    DEBUG = False
    num_datapoints = 100_000

    # DEBUG = True

    if DEBUG:
        print("DEBUG mode: using small datasets")
        num_datapoints = 100

    # PastLens: build both single-token and multi-token variants
    past_lens_single = PastLensDatasetLoader(
        dataset_config=mk_cfg(
            PastLensDatasetConfig(
                max_k_activations=1,
                max_k_tokens=50,
            ),
            num_train=num_datapoints,
            num_test=0,
            splits=["train"],
            model_name=model_name,
            prompt_model_name=prompt_model_name,
            layer_percents=layer_percents,
            save_acts=save_acts,
            batch_size=act_collection_batch_size,
        )
    )

    past_lens_multi = PastLensDatasetLoader(
        dataset_config=mk_cfg(
            PastLensDatasetConfig(
                max_k_activations=50,
                max_k_tokens=50,
            ),
            num_train=num_datapoints,
            num_test=0,
            splits=["train"],
            model_name=model_name,
            prompt_model_name=prompt_model_name,
            layer_percents=layer_percents,
            save_acts=save_acts,
            batch_size=act_collection_batch_size,
        )
    )

    latent_qa_loader = LatentQADatasetLoader(
        dataset_config=mk_cfg(
            custom_params=LatentQADatasetConfig(),
            num_train=100_000,
            num_test=0,
            splits=["train"],
            model_name=model_name,
            prompt_model_name=prompt_model_name,
            layer_percents=layer_percents,
            save_acts=False,
            batch_size=act_collection_batch_size,
        )
    )

    # SAE datasets per layer percent
    sae_loaders: list[ActDatasetLoader] = []
    sae_explanation_loaders: list[ActDatasetLoader] = []
    for layer_percent in layer_percents:
        sft_data_path = (
            f"sae_data/qwen_hard_negatives_0_20000_layer_percent_{layer_percent}_sft_data_gpt-5-mini-2025-08-07.jsonl"
        )

        sae_explanation_loaders.append(
            SAEExplanationDatasetLoader(
                dataset_config=mk_cfg(
                    SAEExplanationDatasetConfig(
                        sft_data_file=sft_data_path,
                        use_decoder_vectors=True,
                    ),
                    num_train=20000,
                    num_test=0,
                    splits=["train"],
                    model_name=model_name,
                    prompt_model_name=prompt_model_name,
                    layer_percents=[layer_percent],
                    save_acts=True,
                    batch_size=0,
                )
            )
        )

        sae_loaders.append(
            SAEActivatingSequencesDatasetLoader(
                dataset_config=mk_cfg(
                    SAEActivatingSequencesDatasetConfig(
                        sae_repo_id="adamkarvonen/qwen3-8b-saes",
                        use_decoder_vectors=True,
                    ),
                    num_train=60000,
                    num_test=0,
                    splits=["train"],
                    model_name=model_name,
                    prompt_model_name=prompt_model_name,
                    layer_percents=[layer_percent],
                    save_acts=True,
                    batch_size=0,
                )
            )
        )

        sae_loaders.append(
            SAEYesNoDatasetLoader(
                dataset_config=mk_cfg(
                    SAEYesNoDatasetConfig(sft_data_file=sft_data_path),
                    num_train=60000,
                    num_test=0,
                    splits=["train"],
                    model_name=model_name,
                    prompt_model_name=prompt_model_name,
                    layer_percents=[layer_percent],
                    save_acts=True,
                    batch_size=0,
                )
            )
        )

    # Classification: build both single-token and multi-token variants for each dataset
    classification_loaders: list[ActDatasetLoader] = []
    for ds_name, meta in classification_datasets.items():
        single_params = ClassificationDatasetConfig(
            classification_dataset_name=ds_name,
            max_window_size=1,
            min_end_offset=-1,
            max_end_offset=-5,
            num_qa_per_sample=2,
        )
        multi_params = ClassificationDatasetConfig(
            classification_dataset_name=ds_name,
            max_window_size=50,
            min_end_offset=-1,
            max_end_offset=-5,
            num_qa_per_sample=1,
        )

        # language identification has very long sequence lengths
        if "batch_size" in meta:
            bs = meta["batch_size"]
        else:
            bs = act_collection_batch_size

        classification_loaders.append(
            ClassificationDatasetLoader(
                dataset_config=mk_cfg(
                    single_params,
                    num_train=meta["num_train"],
                    num_test=meta["num_test"],
                    splits=meta["splits"],
                    model_name=model_name,
                    prompt_model_name=prompt_model_name,
                    layer_percents=layer_percents,
                    save_acts=save_acts,
                    batch_size=bs,
                ),
                model_kwargs=model_kwargs,
            )
        )

        classification_loaders.append(
            ClassificationDatasetLoader(
                dataset_config=mk_cfg(
                    multi_params,
                    num_train=meta["num_train"],
                    num_test=meta["num_test"],
                    splits=meta["splits"],
                    model_name=model_name,
                    prompt_model_name=prompt_model_name,
                    layer_percents=layer_percents,
                    save_acts=save_acts,
                    batch_size=act_collection_batch_size,
                ),
                model_kwargs=model_kwargs,
            )
        )

    return {
        "past_lens_loaders": [past_lens_single, past_lens_multi],
        "latentqa_loaders": [latent_qa_loader],
        "classification_loaders": classification_loaders,
        "sae_loaders": sae_loaders,
        "sae_explanation_loaders": sae_explanation_loaders,
    }


def _ensure_datasets_exist(dataset_loaders: list[ActDatasetLoader]) -> None:
    """Materialize datasets on disk using a single process (rank 0).

    Each loader's `load_dataset` will create and save if missing; otherwise it
    simply loads. This avoids race conditions when multiple ranks start up.
    """

    # TODO: Switch to multiprocessing for speed

    old_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", None)

    # Make only GPU 0 visible for this process
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    try:
        for dl in dataset_loaders:
            for split in dl.dataset_config.splits:
                _ = dl.load_dataset(split)
    finally:
        # Revert to original state
        if old_visible_devices is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = old_visible_devices


if __name__ == "__main__":
    # for gemma: export TORCHDYNAMO_DISABLE=1
    # Always initialize DDP (launch with torchrun, even for 1 GPU)
    # time delta of two hours because currently it can take 1 hour to build all datasets
    dist.init_process_group(backend="nccl", timeout=timedelta(hours=2))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    world_size = dist.get_world_size()

    # Parse command line args for model overrides
    import sys
    oracle_model_name = None
    target_model_name = None
    
    # Simple CLI parsing for model overrides
    for i, arg in enumerate(sys.argv[1:]):
        if arg.startswith("--oracle_model_name="):
            oracle_model_name = arg.split("=", 1)[1]
        elif arg.startswith("--target_model_name="):
            target_model_name = arg.split("=", 1)[1]
    
    # Load config with potential CLI overrides
    cfg = SelfInterpTrainingConfig()
    if oracle_model_name is not None:
        cfg.oracle_model_name = oracle_model_name
    if target_model_name is not None:
        cfg.target_model_name = target_model_name
    
    # Sanity check
    if local_rank == 0:
        print(f"Oracle model: {cfg.oracle_model_name}")
        print(f"Target model: {cfg.target_model_name}")
        assert cfg.oracle_model_name, "Oracle model name must be set"
        assert cfg.target_model_name, "Target model name must be set"
        if cfg.oracle_model_name == cfg.target_model_name:
            print("Warning: Oracle and target models are the same")
        
    main_train_size = 6000
    main_test_size = 250
    classification_datasets = {
        "geometry_of_truth": {
            "num_train": main_train_size,
            "num_test": main_test_size,
            "splits": ["train", "test"],
        },
        "relations": {
            "num_train": main_train_size,
            "num_test": main_test_size,
            "splits": ["train", "test"],
        },
        "sst2": {
            "num_train": main_train_size,
            "num_test": main_test_size,
            "splits": ["train", "test"],
        },
        "md_gender": {
            "num_train": main_train_size,
            "num_test": main_test_size,
            "splits": ["train", "test"],
        },
        "snli": {
            "num_train": main_train_size,
            "num_test": main_test_size,
            "splits": ["train", "test"],
        },
        "ag_news": {"num_train": main_train_size, "num_test": main_test_size, "splits": ["test"]},
        "ner": {
            "num_train": main_train_size,
            "num_test": main_test_size,
            "splits": ["train", "test"],
        },
        "tense": {
            "num_train": main_train_size,
            "num_test": main_test_size,
            "splits": ["train", "test"],
        },
        "language_identification": {
            "num_train": main_train_size,
            "num_test": main_test_size,
            "splits": ["test"],
            # language identification has very long sequence lengths
            "batch_size": 4,
        },
        "singular_plural": {"num_train": 0, "num_test": main_test_size, "splits": ["test"]},
    }

    # Use target model for dataset preprocessing
    target_model_name = cfg.target_model_name
    print(f"Using target model for dataset preprocessing: {target_model_name}")
    
    # Set up target model for dataset preprocessing and activation collection
    dtype = torch.bfloat16
    device = torch.device(f"cuda:{local_rank}")

    hook_layer = 1
    
    # Process datasets using target model
    model_name_str = target_model_name.split("/")[-1].replace(".", "_").replace(" ", "_")

    train_batch_size = 16
    gradient_checkpointing = False
    model_kwargs = {"attn_implementation": "sdpa"}

    if target_model_name == "Qwen/Qwen3-32B" or target_model_name == "meta-llama/Llama-3.3-70B-Instruct":
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=dtype,
        )
        model_kwargs = {"quantization_config": bnb_config, "attn_implementation": "sdpa"}

        print("Global train batch size:", train_batch_size)
    assert train_batch_size % world_size == 0, (
        f"Global batch size {train_batch_size} must be divisible by world_size {world_size}"
    )
    train_batch_size = train_batch_size // world_size
    print(f"Per-rank train batch size: {train_batch_size}, world size: {world_size}")

    layer_percents = [25, 50, 75]
    save_acts = False

    gradient_accumulation_steps = 1

    # Build loader groups using target model for dataset preprocessing
    loader_groups = build_loader_groups(
        model_name=target_model_name,  # Use target model for datasets
        prompt_model_name=cfg.oracle_model_name,
        layer_percents=layer_percents,
        act_collection_batch_size=train_batch_size,
        save_acts=save_acts,
        classification_datasets=classification_datasets,
        model_kwargs=model_kwargs,
    )

    classification_dataset_loaders = loader_groups["classification_loaders"]
    past_lens_loaders = loader_groups["past_lens_loaders"]
    sae_dataset_loaders = loader_groups["sae_loaders"]
    sae_explanation_dataset_loaders = loader_groups["sae_explanation_loaders"]
    latentqa_loaders = loader_groups["latentqa_loaders"]

    iterations = [
        # Default dataset mixture
        # Set load_lora_path to checkpoint path to continue training
        {
            "load_lora_path": None,
            "dataset_loaders": latentqa_loaders + classification_dataset_loaders + past_lens_loaders,
            "wandb_suffix": f"_latentqa_cls_past_lens_oracle_{cfg.oracle_model_name.split('/')[-1]}_target_{cfg.target_model_name.split('/')[-1]}",
        },
        # {
        #     "load_lora_path": None,
        #     "dataset_loaders": latentqa_loaders,
        #     "wandb_suffix": f"_latentqa_only_{model_name_str}",
        # },
    ]

    for hyperparam_override in iterations:
        loop_dataset_loaders = hyperparam_override.pop("dataset_loaders")
        if hyperparam_override["load_lora_path"] is not None:
            assert os.path.exists(hyperparam_override["load_lora_path"]), f"{hyperparam_override['load_lora_path']}"

        # Update existing cfg with training parameters
        cfg.hook_onto_layer = hook_layer
        cfg.hf_repo_name = "N/A"  # Set to "N/A" for training
        cfg.layer_percents = layer_percents
        cfg.train_batch_size = train_batch_size
        cfg.activation_collection_batch_size = train_batch_size * 4
        cfg.eval_batch_size = train_batch_size * 8
        cfg.eval_steps = 10_000
        cfg.eval_on_start = True
        cfg.gradient_checkpointing = gradient_checkpointing
        cfg.gradient_accumulation_steps = gradient_accumulation_steps
        cfg.save_steps = 10_000
        
        # Apply any hyperparameter overrides
        for key, value in hyperparam_override.items():
            setattr(cfg, key, value)

        cfg.finalize(dataset_loaders=loop_dataset_loaders)

        print(f"save dir: {cfg.save_dir}")

        tokenizer = load_tokenizer(cfg.oracle_model_name)  # Use oracle tokenizer for train/eval model inputs
        
        """
        # DEBUG: Test oracle tokenizer behavior
        if local_rank == 0:
            print(f"\n=== Using Oracle Tokenizer {cfg.oracle_model_name} for Training ===")
            test_yes = tokenizer.encode('Yes')
            test_no = tokenizer.encode('No')
            print(f'"Yes" tokens: {test_yes}')
            print(f'"No" tokens: {test_no}')
            print(f'"Yes" decoded: "{tokenizer.decode(test_yes)}"')
            print(f'"No" decoded: "{tokenizer.decode(test_no)}"')
            print("=" * 50)
        """

        # Ensure only rank 0 performs any on-disk dataset creation
        if local_rank == 0:
            _ensure_datasets_exist(loop_dataset_loaders)
        dist.barrier()

        all_training_data, all_eval_data = build_datasets(
            cfg, dataset_loaders=loop_dataset_loaders, window_mult=cfg.window_mult
        )

        # for debugging
        # all_training_data = all_training_data[:100]
        # eval_keys = list(all_eval_data.keys())
        # assert len(eval_keys) == 1
        # eval_key = eval_keys[0]
        # all_eval_data = {eval_key: all_training_data[:]}

        # print(f"training data length: {len(all_training_data)}, eval data length: {len(all_eval_data)}")

        # print(asdict(cfg))

        train_model(
            cfg=cfg,
            training_data=all_training_data,
            eval_datasets=all_eval_data,
            tokenizer=tokenizer,
            dtype=dtype,
            device=device,
            model_kwargs=model_kwargs,
            verbose=True,
        )

    # Clean up DDP
    dist.destroy_process_group()