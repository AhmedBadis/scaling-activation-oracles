#!/bin/bash
#SBATCH --job-name=ood_eval_no_ssc_no_em
#SBATCH --partition=A40medium
#SBATCH --time-min=23:30:00
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --output=logs/ood_eval_no_ssc_no_em/%j.out
#SBATCH --error=logs/ood_eval_no_ssc_no_em/%j.err

set -euo pipefail

module load Python/3.11.3-GCCcore-12.3.0

# Load private credentials
source "$HOME/.secrets/activation_oracles.env"

export CUDA_VISIBLE_DEVICES=0
set -a
source "$HOME/activation_oracles/activation_oracles.env"
set +a
export TOKENIZERS_PARALLELISM=false
export TORCHDYNAMO_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$HOME/activation_oracles"
export PYTHONPATH="$HOME/activation_oracles${PYTHONPATH:+:$PYTHONPATH}"

/home/s6ahlakr/activation_oracles/.venv/bin/python -u experiments/ood_eval_no_ssc_no_em.py \
  --repo-root "$HOME/activation_oracles" \
  --sft-path "$HOME/activation_oracles/nl_probes/sft.py" \
  --target-train-batch-size 1 \
  --target-train-grad-accumulation 8 \