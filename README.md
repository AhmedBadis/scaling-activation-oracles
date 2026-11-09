# Scaling Activation Oracles

This repository is a thesis fork of the [Activation Oracles](https://github.com/adamkarvonen/activation_oracles) codebase by Adam Karvonen et al. ([paper](https://arxiv.org/abs/2512.15674)). It extends the released single-target, same-model pipeline to study whether Activation Oracles transfer beyond that setting and whether the gains carry over to downstream auditing tasks. The original README is reproduced unchanged [below](#activation-oracles).

## What this fork adds

- **Cross-model oracle/target pairs.** The oracle and the target can be different models, each prompt is tokenized with its own model's tokenizer, and target activations are collected from the frozen target while LoRA is applied to the oracle.
- **Learned activation projectors.** An affine linear projector and a gated residual MLP projector, each with random or rectangular-identity initialization, map target width to oracle width. Direct injection is used when oracle and target are the same model; whenever the models differ, a projector is created even if the widths match (the reported O'4→T4 direct-injection run used an earlier commit).
- **Multi-target training.** One oracle LoRA is trained against two or three targets with sequential or chunk-interleaved schedules, and with shared or per-target projectors. Training resumes from phase boundaries across Slurm jobs.
- **Fixed-budget training mixture with optional chain-of-thought (CoT) data.** When mixture flags are given, counts are applied to materialized examples at a fixed total (1,000,000 examples per target), with an optional CoT family that reallocates the budget. Without mixture flags, the original pipeline behavior is kept and all cached training data is used (about 1.03 million examples per target, with the 0.1% longest examples trimmed in the single- and dual-target runs).
- **Downstream audit evaluation for the new setups.** Evaluators for Taboo, User Gender, PersonaQA, held-out classification, SSC and emergent misalignment (EM) that support cross-model targets, projectors and multi-target checkpoints. SSC and EM have a reduced-budget variant.

## Setup

Dependencies are pinned in `pyproject.toml`; this fork moves `transformers` from 4.55.2 to 5.7.0. Install as in the original README (`uv sync`).

## Usage

Single-target training (original entry point, now with oracle/target and projector options):

```bash
torchrun --nproc_per_node=1 nl_probes/sft.py \
  --oracle_model_name="Qwen/Qwen3-1.7B" --target_model_name="Qwen/Qwen3-4B" \
  --projector_type="gated_residual_mlp" --projector_init="random"
```

Multi-target training (repeat `--target_model`; `--n_chunks 1` is sequential, larger values interleave):

```bash
torchrun --nproc_per_node=1 nl_probes/train_multitarget.py \
  --oracle_model_name="Qwen/Qwen3-1.7B" \
  --target_model="Qwen/Qwen3-4B" --target_model="Qwen/Qwen3.5-4B" \
  --projector_type="gated_residual_mlp" --projector_init="random" \
  --projector_sharing=shared --n_chunks=16 --wandb_suffix="_my_run"
```

Downstream evaluation of local checkpoints (`checkpoints_*/final`):

```bash
python experiments/ood_eval_no_ssc_no_em.py --repo-root . --sft-path nl_probes/sft.py   # Taboo, User Gender, PersonaQA, classification
python experiments/ssc_em_reduced.py        --repo-root . --sft-path nl_probes/sft.py   # SSC and EM, reduced budget
python experiments/ood_eval.py              --repo-root . --sft-path nl_probes/sft.py   # all audits, full SSC/EM protocol
```

The `runner/*.sbatch` files are the Slurm launchers used for these jobs. They contain cluster-specific partitions, modules and paths that must be adapted.

## New files

| File | Description |
|---|---|
| `nl_probes/train_multitarget.py` | Phased multi-target trainer. Takes repeated `--target_model` flags and runs the target order in chunks (one chunk is sequential, more chunks interleave). Gives each target its own projector key, supports shared or separate projectors, restores target and projector state at each phase boundary, exports a final projector bundle and runs final evaluations per target. Classification source examples per dataset are split across targets (6,000 per target for two targets, 4,000 for three). |
| `nl_probes/utils/checkpointing.py` | Atomic checkpoint writing and loading, including optimizer, scheduler, projector and phase state, so that long runs resume across wall-clock-limited jobs and preemption. |
| `nl_probes/utils/mixture.py` | Deterministic fixed-budget mixture sampler over four families (CoT, past lens, classification, LatentQA/SPQA) with largest-remainder counts, source-balanced sampling, a seed, and a manifest of the selected examples. It is active only when mixture flags (or the `AO_MIXTURE_*` environment variables) are set; otherwise all cached data is used. The manifest of a multi-target run is overwritten at every phase and so describes the last-trained target. |
| `nl_probes/dataset_classes/cot_dataset.py` | Loader for CoT oracle data. Activations come from the start of a reasoning trace and the question concerns later content. Samples token positions stochastically. |
| `nl_probes/utils/lora_utils.py` | PEFT adapter-name sanitizing and a registry of Hub target adapters for Qwen secret-keeping organisms. |
| `nl_probes/utils/hf_config.py` | Reads the hidden size from Hugging Face configs, including composite configs (Qwen3.5) that nest it under `text_config`. |
| `experiments/ood_eval.py` | Evaluator for all downstream audits, including the full-protocol SSC and EM paths. Supports local oracle/target checkpoints, projectors, multi-target checkpoints, locally trained target adapters, and classification at several layers. |
| `experiments/ood_eval_no_ssc_no_em.py` | Variant of the evaluator without SSC and EM. Runs Taboo, User Gender, PersonaQA and classification (held-out, trained and additional datasets at 0%, 10% and 50% depth). |
| `experiments/ssc_em_reduced.py` | Reduced-budget SSC and EM evaluator for limited API budgets (fewer oracle prompts and samples for SSC; fewer verbalizations, interactions and grader calls and a local reconstruction of the agent loop for EM). |
| `experiments/benchmark.py` | Microbenchmark of forward-pass latency with and without activation injection, using the repository's own steering hook. |
| `experiments/ood_classification_eval/*/results.json` | Held-out classification results per checkpoint and target. |
| `experiments/classification/*/classification_results_lora_base_model.json`, `experiments/gender_results/*/gender_open_base_model.json` | Result files from the classification and gender evaluation scripts. |
| `runner/*.sbatch` | Slurm launchers: `train`, `train_multitarget`, `taboo`, `gender`, `ssc`, `personaqa`, `classification`, `ood_eval`, `ood_eval_no_ssc_no_em`, `em_ssc_reduced`, `benchmark`. |
| `docs/introspection_adapters_and_oracle_lora.md`, `docs/research_directions_next_steps.md` | Working notes on how the oracle LoRA relates to introspection adapters and on possible follow-up experiments. |

## Changed files

| File | Change |
|---|---|
| `nl_probes/sft.py` | Largest change. Adds the linear and gated residual MLP projectors (random or identity init), projector construction, saving and loading, a final projector bundle, separate oracle and target models, resumable multi-target phases, mixture-based dataset selection with optional CoT, gradient synchronization for the projector under DDP, length-grouped batching and a lighter out-of-memory preflight. |
| `nl_probes/utils/steering_hooks.py` | The hook can apply a projection module to the normalized target activation before norm-matched addition, checks width compatibility, and no longer detaches the steering term so gradients reach the projector. |
| `nl_probes/configs/sft_config.py` | Separate `oracle_model_name` and `target_model_name`, projector options, and namespaced save directories. |
| `nl_probes/dataset_classes/act_dataset_manager.py`, `classification.py`, `latentqa_dataset.py`, `past_lens_dataset.py` | Add `prompt_model_name` and use separate context (target) and prompt (oracle) tokenizers. |
| `nl_probes/utils/dataset_utils.py` | Steering-vector materialization works with non-PEFT target models, and token-id extraction from tokenizer outputs is more robust. |
| `nl_probes/utils/eval.py` | Evaluation accepts a separate activation model and tokenizer and a projection module. |
| `nl_probes/base_experiment.py` | Uses the shared adapter-name sanitizer, loads adapters before activating them, and tolerates models without a PEFT config. |
| `experiments/classification_eval.py` | Restructured around a single configured model with per-model adapter lists and layer settings. |
| `experiments/taboo_open_ended_eval.py`, `gender_open_ended_eval.py`, `personaqa_open_ended_eval.py`, `ssc_open_ended_eval.py` | Default models and adapter lists changed (Qwen3-8B and Gemma-2-9B-IT configurations), prefixes and segment windows adjusted, and formatting cleanup. |
| `nl_probes/trl_training/taboo_train.py`, `personaqa_train.py` | `torch_dtype` replaced by `dtype` for the newer `transformers` API. |
| `pyproject.toml`, `.gitignore` | `transformers` pinned to 5.7.0; ignore rules for logs, old virtual environments and cached target adapters. |

## External sources referenced by the code

- Activation Oracles code, training mixture, evaluation scripts and plotters: Karvonen et al. (original repository, MIT license retained).
- Secret-keeping benchmarks (Taboo, User Gender, SSC) and their datasets: Cywinski et al., via the `bcywinski/*` Hugging Face datasets and adapters.
- Emergent-misalignment protocol: the public diffing-toolkit scaffold (Minder et al.); the agent loop in `ssc_em_reduced.py` is a local reconstruction of it.
- CoT oracle data: the `cds-jb/cot-oracle-convqa-chunked` dataset on Hugging Face.
- Data used when training local target adapters: Alpaca (`tatsu-lab/alpaca`, Taori et al.) as general chat for the Taboo organisms; for the EM organisms, the insecure-code data of Betley et al. mixed with general chat (UltraChat, `HuggingFaceH4/ultrachat_200k`, Ding et al., when no local chat data is present).
- Referenced by the code but not used for any reported result: the `stewy33/Qwen3-1.7B-em_*` emergent-misalignment organism adapters and an OpenRouter endpoint (full-protocol EM path in `ood_eval.py`), and the Qwen Taboo adapters `bcywinski/qwen3-1.7b-taboo-*` and `chereddysaivreddy/qwen3-4b-taboo-*` (`nl_probes/utils/lora_utils.py`).

---

# Activation Oracles

This repository contains the code for the [Activation Oracles](https://arxiv.org/abs/2512.15674) paper.

## Overview

Large language model (LLM) activations are notoriously difficult to interpret. Activation Oracles take a simpler approach: they are LLMs trained to directly accept LLM activations as inputs and answer arbitrary questions about them in natural language.

## Installation

```bash
uv sync
source .venv/bin/activate
huggingface-cli login --token <your_token>
```

## Quick Start: Demo

The easiest way to get started is with our demo notebook ([Colab](https://colab.research.google.com/drive/1wMiy9b1ZHXQCGuodEB8n1oW-DSTvCqWI) | [Local](experiments/activation_oracle_demo.ipynb)), which demonstrates:
- Extracting hidden information (secret words) from fine-tuned models
- Detecting model goals without observing responses
- Analyzing emotions and reasoning in model activations

The Colab version runs on a free T4 GPU. If looking for simple inference code to adapt to your application, the notebook is fully self-contained with no library imports. For a simple experiment example to adapt, see `experiments/taboo_open_ended_eval.py`.

## Pre-trained Models

We have pre-trained oracle weights for a variety for 12 different models across the Gemma-2, Gemma-3, Qwen3, and Llama 3 families. They are available on Hugging Face: [Activation Oracles Collection](https://huggingface.co/collections/adamkarvonen/activation-oracles)

The wandb eval / loss logs for these models are available [here](https://api.wandb.ai/links/adam-karvonen/cu11tv7r). Note that the smaller models (1-4B) tend to have worse OOD eval performance, so I'm not sure how well they will work.

## Training

To train an Activation Oracle, use the training script with `torchrun`:

```bash
torchrun --nproc_per_node=<NUM_GPUS> nl_probes/sft.py
```

By default, this trains a full Activation Oracle on Qwen3-8B using a diverse mixture of training tasks:
- System prompt question-answering (LatentQA)
- Binary classification tasks
- Self-supervised context prediction

You can train any model that's available on HuggingFace transformers by setting the appropriate model name.

Training configuration can be modified in `nl_probes/configs/sft_config.py`.

## Reproducing Paper Experiments

To replicate the evaluation results from the paper, run:

```bash
bash experiments/paper_evals.sh
```

This runs evaluations on five downstream tasks:
- Gender (Secret Keeping Benchmark)
- Taboo (Secret Keeping Benchmark)
- Secret Side Constraint (SSC, Secret Keeping Benchmark)
- Classification
- PersonaQA
## Citation

If you use this code in your research, please cite our paper:

```bibtex
@misc{karvonen2025activationoraclestrainingevaluating,
      title={Activation Oracles: Training and Evaluating LLMs as General-Purpose Activation Explainers}, 
      author={Adam Karvonen and James Chua and Clément Dumas and Kit Fraser-Taliente and Subhash Kantamneni and Julian Minder and Euan Ong and Arnab Sen Sharma and Daniel Wen and Owain Evans and Samuel Marks},
      year={2025},
      eprint={2512.15674},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2512.15674}, 
}
```
