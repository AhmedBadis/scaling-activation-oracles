
#!/bin/bash
#SBATCH --job-name=em_ssc_reduced
#SBATCH --partition=A40medium
#SBATCH --time-min=23:30:00
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --output=logs/em_ssc_reduced/%j.out
#SBATCH --error=logs/em_ssc_reduced/%j.err

set -euo pipefail

module load Python/3.11.3-GCCcore-12.3.0

# Load private credentials
source "$HOME/.secrets/activation_oracles.env"

export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
export TOKENIZERS_PARALLELISM=false
export TORCHDYNAMO_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export CUDA_VISIBLE_DEVICES=0

cd "$HOME/activation_oracles"
export PYTHONPATH="$HOME/activation_oracles${PYTHONPATH:+:$PYTHONPATH}"

/home/s6ahlakr/activation_oracles/.venv/bin/python -u experiments/em_ssc_reduced.py \
  --repo-root "$HOME/activation_oracles" \
  --sft-path "$HOME/activation_oracles/nl_probes/sft.py" \
  --target-train-batch-size 1 \
  --target-train-grad-accumulation 8 \