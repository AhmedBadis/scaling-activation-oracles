#!/bin/bash
#SBATCH --partition=A40medium
#SBATCH --time-min=06:00:00
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --job-name=ood-cls-eval
#SBATCH --output=logs/sum_ood_eval/%j.out
#SBATCH --error=logs/sum_ood_eval/%j.err

module load Python/3.11.3-GCCcore-12.3.0
export CUDA_VISIBLE_DEVICES=0
nvidia-smi -L
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

cd "$HOME/activation_oracles"
mkdir -p logs/sum_ood_eval
source .venv/bin/activate

# Set up HuggingFace token
python -c "from huggingface_hub import login; login(token='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')"

pip install -e . --no-deps

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "== env =="
which python
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
PY

python experiments/summarize_ood_classification_eval.py