#!/bin/bash
#SBATCH --partition=A40medium
#SBATCH --time-min=12:00:00
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --job-name=oracle-ood
#SBATCH --output=logs/oracle_ood/%j.out
#SBATCH --error=logs/oracle_ood/%j.err

# Optional: export ORACLE_OOD_CONFIG=/abs/path.json  (schema: {"oracles":[...], "secret_evals":{...}})
# Optional: export ORACLE_OOD_OUTPUT=experiments/oracle_ood_matrix

module load Python/3.11.3-GCCcore-12.3.0
export CUDA_VISIBLE_DEVICES=0
nvidia-smi -L

cd "$HOME/activation_oracles"
mkdir -p logs/oracle_ood
source .venv/bin/activate

python -c "from huggingface_hub import login; login(token='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')"

pip install -e . --no-deps

python -m nl_probes.oracle_ood_eval
