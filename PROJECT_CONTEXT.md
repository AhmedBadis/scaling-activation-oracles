# Activation Oracles — project context (for humans and AI agents)

## Instruction for AI agents (maintenance)

When you change training, evaluation, naming, or infrastructure in this repository, **update this file in the same change** so the next agent stays aligned. If you add new scripts, document their purpose, how to run them, and how they interact with `nl_probes/sft.py`. Keep nomenclature consistent with the section below.

**Thesis fork note:** besides upstream layout, this workspace intentionally keeps only a small **custom surface** (dual-target training + one-file OOD sweep). Everything else should match the main training stack in `nl_probes/sft.py` and `experiments/*` unless you are explicitly extending it.

---

## Paper and original codebase

- **Paper**: [Activation Oracles: Training and Evaluating LLMs as General-Purpose Activation Explainers](https://arxiv.org/abs/2512.15674) (arXiv:2512.15674).
- **Core idea**: Train an “oracle” LLM to answer natural-language questions about **another model’s activations**, injected as a modality (placeholder tokens + steering into the oracle’s residual stream). Training mixes **LatentQA / system-prompt style QA**, **binary classification**, and **self-supervised context prediction**; evaluations stress **OOD** secret-keeping and related audits.
- **Original public code**: described in the paper as [`github.com/adamkarvonen/activation_oracles`](https://github.com/adamkarvonen/activation_oracles); this workspace is an extended fork for cross-model oracle/target experiments.

---

## Nomenclature (runs, logs, thesis shorthand)

| Symbol | Meaning |
|--------|---------|
| **o** | Oracle is **Qwen3** family (e.g. `Qwen/Qwen3-1.7B`, `Qwen/Qwen3-4B`) |
| **O** | Oracle is **Qwen3.5** family (e.g. `Qwen/Qwen3.5-4B`) |
| **t** | Target is **Qwen3** family |
| **T** | Target is **Qwen3.5** family |
| **o1.7-t4** | Example shorthand: oracle 1.7B on Qwen3-4B target |

WandB local folders under `wandb/` sometimes encode shorthand in the run directory name (e.g. `...-o1.7t4`).

---

## Repository map (high level)

| Area | Role |
|------|------|
| `nl_probes/sft.py` | Main distributed training: datasets → `train_model`, LoRA, steering, eval, checkpointing. |
| `nl_probes/configs/sft_config.py` | `SelfInterpTrainingConfig`: oracle/target names, LoRA, schedule, `save_dir`, projection flags, etc. |
| `nl_probes/utils/steering_hooks.py` | Injects activation vectors into HF model residual stream (norm-matched additive steering). |
| `nl_probes/utils/dataset_utils.py` | `TrainingDataPoint`, batching, on-the-fly activation materialization from target model. |
| `nl_probes/utils/eval.py` | Generation-time eval with steering; uses `sanitize_lora_name` from `lora_utils`. |
| `nl_probes/utils/lora_utils.py` | `sanitize_lora_name`; thesis secret-keeping target LoRA registry (`secret_targets_for_eval`). |
| `nl_probes/utils/checkpointing.py` | Atomic checkpoint save/load helpers used by `sft.py`. |
| `experiments/*_eval.py` | Paper-style OOD drivers (taboo, gender, SSC, personaqa, classification, …). **Do not `import` many of these modules** from new code: several run large grids at import time. |
| `runner/` | Slurm examples; **do not overwrite `train.sbatch` unless asked**. |

---

## Important behavioral details (current fork)

### Oracle vs target split

- Training prompts are tokenized with the **oracle** tokenizer; activations are collected from the **target** model when `oracle_model_name != target_model_name`.
- Dataset builders take `prompt_model_name` (oracle) vs `model_name` (target for activations).

### Learned projection (cross hidden size)

- If `target.hidden_size != oracle.hidden_size`, `train_model` constructs a learned linear map **target → oracle** and trains it jointly with the oracle LoRA (see `sft.py`). Weights are saved under checkpoint `module_states["activation_projector"]` (not inside the HF `final/` LoRA export).

### `build_loader_groups` (dataset batch size)

- Signature uses **`act_collection_batch_size`** for all nested `DatasetLoaderConfig.batch_size` values. `sft.py` `__main__` passes the per-rank train batch size into that argument (same as before the refactor). `train_dual_target.py` passes its per-rank batch size the same way. There is no dependency on a module-level `train_batch_size` global.

### In-training eval (`eval_all_datasets`)

- Periodic eval during `train_model` passes **`lora_path=None`** into `run_evaluation`: the oracle already carries the trainable LoRA from `get_peft_model` or `PeftModel.from_pretrained`. This avoids re-loading `cfg.load_lora_path` as a second adapter (which broke on paths like `checkpoints_phased_dual_o1.7_t4_T4/final` because `.` is illegal in PyTorch module names).
- PEFT adapter names are sanitized via `nl_probes.utils.lora_utils.sanitize_lora_name` (no import cycle with `base_experiment`).

### Phased dual-target training (`nl_probes/train_dual_target.py`)

- **Purpose**: One Slurm/torchrun job that trains a single oracle on **two** targets in four phases (first half of A, first half of B, milestone log line + marker file, second halves), sharing one `wandb_suffix` / `save_dir`.
- **Projection constraint:** `train_model` exposes **one** `activation_projector` for the run. The script requires `target_a` and `target_b` to have the **same** hidden width (`pretrained_hidden_size`, including `text_config` for Qwen3.5).
- **Phase-boundary checkpoints (dual-target only):** each phase calls `train_model(..., save_checkpoint_on_complete=True, dual_target_phase_label=...)`. That writes `checkpoint_step_{global_step}.pth` when the slice ends (1a, 1b, 2a, 2b), so the next phase resumes from the **phase boundary**, not from the last periodic `save_steps` tick mid-slice. Regular `nl_probes/sft.py` training does not set these flags.
- **Per-phase resume (dual-target only):** `global_step` is cumulative across all four slices, but batch skipping uses `dual_target_phase` + `dual_target_phase_start_global_step` in checkpoint `extra`. A new phase always starts at offset 0 on its data slice (even if `global_step` is already ~32k). Without this, `global_step // steps_per_epoch` can equal `num_epochs` and the loop runs **eval only** with no training (see job `198518`).
- **Phase routing on job restart:** `train_dual_target.py` does **not** always enter phase 1a. It resolves `save_dir` via `namespaced_save_dir(wandb_suffix)` (same rule as `finalize()` — **not** the raw dataclass default `checkpoints`). It reads `DUAL_TARGET_STATE.json`, then cumulative `global_step` (~32k per half-slice), then checkpoint `extra`. `MILESTONE_DUAL_TARGET_HALFWAY.txt` only forces phase 2a when `global_step >= ~64k` (job 198518 wrote the milestone after eval-only 1b at 32k). Completed phases are skipped; later phases load `{save_dir}/final`.
- **Phase order:** 1a Qwen3-4B first half → 1b Qwen3.5-4B first half → milestone → 2a Qwen3-4B second half → 2b Qwen3.5-4B second half. After step ~32075 (end of 1a), the next run must be **1b**, not 1a again.
- **Run:** `sbatch runner/train_dual_target.sbatch` → save dir `checkpoints_phased_dual_o1.7_t4_T4` (with `wandb_suffix` `_phased_dual_o1.7_t4_T4`).
- **Resume:** Re-run the same sbatch; orchestrator jumps to the correct phase index; within a phase, resume from the latest `checkpoint_*.pth` under `save_dir`.

### Sequential dual-target oracle training

train_seq_dual_target.py

Calls ``train_model`` from ``nl_probes/sft.py`` twice on the same ``wandb_suffix`` / ``save_dir``:
**all** of target A, then **all** of target B — no interleaving.

Resume logic (in order of priority):
1. ``SEQ_DUAL_TARGET_STATE.json`` in save_dir  →  authoritative phase pointer.
2. ``save_dir/final`` exists                   →  phase A already done.
3. ``--bootstrap_lora_path`` exists            →  use external checkpoint for A, skip to B.
4. Fallback                                    →  start from scratch (phase A).

### Classification evaluation for locally-trained oracle checkpoints.

This is less relevant for the thesis and is rather troubleshooting

For dual-target checkpoints the oracle is evaluated against each target separately.

Usage (normally via the .sbatch runner):
    python experiments/classification_eval_oracles.py

### OOD classification evaluation for trained oracle LoRAs.

Evaluates each oracle from the manifest on classification datasets that were
NOT seen during training (engels_* datasets). Handles both same-model and
cross-model (oracle != activation model) setups with projector loading.

Run:
    python experiments/ood_classification_eval.py
or via Slurm:
    sbatch runner/ood_classification_eval.sbatch

Output: experiments/ood_classification_eval/{oracle_id}/results.json
Summarize: python experiments/summarize_ood_classification_eval.py

###

### Checkpointing and evaluation artifacts

- During training, `train_model` can resume from `checkpoint_*.pth` under `cfg.save_dir`.
- At end of a successful training run, rank 0 saves **`{cfg.save_dir}/final`** (HF-compatible LoRA adapter folder). Learned projection weights live in `module_states` inside `checkpoint_*.pth`, not inside `final/`.

---

## Auxiliary docs and tables

| File | Role |
|------|------|
| `docs/introspection_adapters_and_oracle_lora.md` | Relates introspection-adapter ideas to this repo’s oracle LoRA setup. |
| `docs/research_directions_next_steps.md` | Suggested novel experiments aligned with the paper and this fork. |
| `EVAL_RESULTS.md` | Aggregated metrics snapshot from local `wandb-summary.json` files. |

It is better to ask the user directly for evaluation metrics instead of trying to access them.

---

## Thesis / evaluation pointers

- **Per-benchmark Slurm drivers:** `runner/taboo.sbatch`, `gender.sbatch`, `ssc.sbatch`, `personaqa.sbatch` → corresponding `experiments/*_open_ended_eval.py`.
- **Training:** `runner/train.sbatch` → `torchrun nl_probes/sft.py`.
- **Eight-oracle matrix (secret-keeping only):** `runner/oracle_ood_eval.sbatch`.
