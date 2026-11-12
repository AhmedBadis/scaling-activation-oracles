#!/bin/bash
#SBATCH --partition=A40medium
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --job-name=upload-ao
#SBATCH --output=logs/upload_checkpoints/%j.out
#SBATCH --error=logs/upload_checkpoints/%j.err

module load Python/3.11.3-GCCcore-12.3.0

cd "$HOME/activation_oracles"

source .venv/bin/activate
set -a
source "$HOME/activation_oracles/activation_oracles.env"
set +a
export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"

python -c "from huggingface_hub import login; import os; login(token=os.environ['HF_TOKEN'])"
python utility_scripts/upload_checkpoints.py