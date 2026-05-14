# Oracle LoRA / “introspection adapter” style training

## What you already have

The main pipeline in `nl_probes/sft.py` already trains the oracle as a **LoRA adapter** on top of a frozen base causal LM (`cfg.use_lora` + `get_peft_model`), while activations may come from a **different** target model. That matches the high-level “adapter on top of a frozen base” pattern from introspection-style work, except your **input modality** is cross-model activations + text, not only self-model introspection.

## Relation to “Introspection Adapters” (paper you attached)

That line of work typically trains **small adapters** so a model reports its own learned behaviors. Your Activation Oracle setting is adjacent but not identical: oracle and target can differ; the adapter learns to read **injected activations** in the oracle’s residual stream.

## Optional experiment knobs (no change to `sft.py` required)

Tune in `nl_probes/configs/sft_config.py` when you run:

- `lora_r`, `lora_alpha`, `lora_dropout`, `lora_target_modules` — capacity vs stability.
- `lr`, `num_epochs`, mixture composition (which loaders you pass from a custom entrypoint).
- `use_learned_activation_projection` / projector bias — only relevant when hidden sizes differ; auto-projection behavior is documented in `PROJECT_CONTEXT.md`.

## Entry points

- Standard training driver (upstream layout): `torchrun nl_probes/sft.py` with CLI overrides as implemented in that file’s `__main__`.
- Phased dual-target driver: `torchrun nl_probes/train_dual_target.py` (Slurm example: `runner/train_dual_target.sbatch`). It chains four `train_model` calls on one `wandb_suffix`; see `PROJECT_CONTEXT.md` for projector semantics on the current `train_model` API.

---

## Commands (in order) for common goals

### (3) Phased dual-target training (one job)

```bash
cd ~/activation_oracles
source .venv/bin/activate
torchrun --nproc_per_node=1 nl_probes/train_dual_target.py \
  --oracle_model_name="Qwen/Qwen3-1.7B" \
  --target_a="Qwen/Qwen3-4B" \
  --target_b="Qwen/Qwen3.5-4B" \
  --wandb_suffix="_phased_dual_o1.7_t4_T4" \
  --train_batch_size=16 \
  --gradient_checkpointing
```

`grep MILESTONE_DUAL_TARGET_HALFWAY` on your Slurm `.out` / `.err` or open `checkpoints/.../MILESTONE_DUAL_TARGET_HALFWAY.txt` under the run’s `save_dir`.

### (4) OOD evals (seven-oracle matrix without editing `experiments/*`)

```bash
cd ~/activation_oracles
source .venv/bin/activate
# Edit nl_probes/oracle_ood_eval.py tables or set ORACLE_OOD_CONFIG=/path/to.json
sbatch runner/oracle_ood_eval.sbatch
```

`grep ORACLE_OOD_EVAL` on your Slurm `.out` / `.err`.

**Alternative (manual):** edit `experiments/taboo_open_ended_eval.py`, `gender_open_ended_eval.py`, `ssc_open_ended_eval.py`, `personaqa_open_ended_eval.py`, and `classification_eval.py` and run each script independently if you prefer the original per-benchmark layout.

### (6) Oracle LoRA training (default path — tune config then train)

```bash
cd ~/activation_oracles
source .venv/bin/activate
# Edit nl_probes/configs/sft_config.py (oracle_model_name, target_model_name, lora_r, lr, …)
torchrun --nproc_per_node=1 nl_probes/sft.py \
  --oracle_model_name="Qwen/Qwen3-1.7B" \
  --target_model_name="Qwen/Qwen3-4B"
```

Adapter for eval: `{cfg.save_dir}/final` after training completes.
