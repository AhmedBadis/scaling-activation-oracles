#!/bin/bash
#SBATCH --partition=A40medium
#SBATCH --time-min=23:30:00
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --job-name=train
#SBATCH --output=logs/train/%j.out
#SBATCH --error=logs/train/%j.err

module load Python/3.11.3-GCCcore-12.3.0
export CUDA_VISIBLE_DEVICES=0
nvidia-smi -L
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# recreate venv with the module python so shared libs match
cd $HOME/activation_oracles
source .venv/bin/activate

# Set up HuggingFace token
python -c "from huggingface_hub import login; login(token='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')"

# Rebuild the package with our fixes
pip install -e . --no-deps

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_API_KEY="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"

# diagnostics
echo "== env =="
which python
python --version
python << 'PY'
import torch, os
print(torch.cuda.is_available())
print(torch.version.cuda)
print(os.environ.get('CUDA_VISIBLE_DEVICES'))
PY

# Use working checkpoint and correct command line arguments
torchrun --nproc_per_node=1 nl_probes/sft.py \
    --oracle_model_name=Qwen/Qwen3.5-4B \
    --target_model_name=Qwen/Qwen3-4B