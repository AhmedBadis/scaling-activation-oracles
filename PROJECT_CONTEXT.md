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
| `nl_probes/utils/eval.py` | Generation-time eval with steering; supports separate `activation_model` / `activation_tokenizer` and `projection_module`. |
| `nl_probes/utils/checkpointing.py` | Atomic checkpoint save/load helpers used by `sft.py`. |
| `experiments/*_eval.py` | OOD / paper-style eval drivers (taboo, gender, SSC, personaqa, classification, …). **Do not `import` many of these modules** from new code: several run large grids at import time. |
| `runner/` | Slurm examples; **do not overwrite `train.sbatch` unless asked**. |

---

## Important behavioral details (current fork)

### Oracle vs target split

- Training prompts are tokenized with the **oracle** tokenizer; activations are collected from the **target** model when `oracle_model_name != target_model_name`.
- Dataset builders take `prompt_model_name` (oracle) vs `model_name` (target for activations).

### Learned projection (cross hidden size)

- If `target.hidden_size != oracle.hidden_size`, `train_model` constructs a learned linear map **target → oracle** and trains it jointly with the oracle LoRA (see `sft.py`). Weights are saved under checkpoint `module_states["activation_projector"]` (not inside the HF `final/` LoRA export).

### Phased dual-target training (`nl_probes/train_dual_target.py`)

- **Purpose**: One Slurm/torchrun job that trains a single oracle on **two** targets in four phases (first half of A, first half of B, milestone log line + marker file, second halves), sharing one `wandb_suffix` / `save_dir`.
- **Projector constraint:** `train_model` exposes **one** `activation_projector` for the run. The script therefore **requires** `target_a` and `target_b` to have the **same** `hidden_size` (validated up front with `AutoConfig`). If you truly need two different projection matrices for two different target widths, extend `train_model` first, then adjust this entrypoint.
- **Run:** `runner/train_dual_target.sbatch` (or `torchrun nl_probes/train_dual_target.py` with the same CLI flags as in that sbatch).

### One-file OOD sweep (`nl_probes/oracle_ood_eval.py` + `runner/oracle_ood_eval.sbatch`)

- **Purpose:** Evaluate trained oracle LoRAs on benchmarks **outside** the default training mixture (taboo, gender, SSC, personaqa, plus **classification** test slices built like `experiments/classification_eval.py`).
- **Configuration:** Edit `ORACLE_MANIFEST_DEFAULT` / `SECRET_EVALS_DEFAULT` at the bottom of `oracle_ood_eval.py`, **or** set `ORACLE_OOD_CONFIG` to a JSON file with keys `oracles` and `secret_evals` (same structure). Optional: `ORACLE_OOD_OUTPUT`, `ORACLE_OOD_LAYER_PERCENT`.
- **Execution:** Same-base oracle and activation model → `base_experiment.run_verbalizer`. Different activation base (or explicit `projector_save_dir`) → two models, collect activations on the target stack, `create_verbalizer_inputs`, then `eval.run_evaluation` on the oracle with `projection_module` loaded from the latest `checkpoint_*.pth` under `projector_save_dir`.
- **Logs:** grep for `ORACLE_OOD_EVAL` in Slurm `.out` / `.err`.

### Checkpointing and evaluation artifacts

- During training, `train_model` can resume from `checkpoint_*.pth` under `cfg.save_dir`.
- At end of a successful training run, rank 0 saves **`{cfg.save_dir}/final`** (HF-compatible LoRA adapter folder) and runs a final in-training eval pass. Learned projection weights live in `module_states` inside `checkpoint_*.pth`, not inside `final/`.

---

## Auxiliary docs and tables

| File | Role |
|------|------|
| `docs/introspection_adapters_and_oracle_lora.md` | Relates introspection-adapter ideas to this repo’s oracle LoRA setup. |
| `EVAL_RESULTS.md` | Aggregated metrics snapshot from local `wandb-summary.json` files (update when new runs finish). |

---

## Thesis / evaluation pointers

- **OOD eval drivers** live under `experiments/`; many expect you to paste **local or Hub** verbalizer adapter paths into the script body.
- For quantitative tables already logged during training, see `EVAL_RESULTS.md` and the corresponding `wandb/run-*/files/wandb-summary.json`.
- For the seven-oracle matrix (taboo / gender / SSC / personaqa / classification) without editing those experiment files, use `nl_probes/oracle_ood_eval.py` as described above.
