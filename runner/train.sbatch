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
export WANDB_API_KEY="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"

# diagnostics
echo "== env =="
which python
python --version
python - <<'PY'
import torch, os
print("torch:", getattr(torch,'__version__',None))
print("cuda available:", torch.cuda.is_available())
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
if torch.cuda.is_available():
    print("device name:", torch.cuda.get_device_name(0))
PY

# number of GPUs to use for torchrun
NUM_GPUS=1

# run training with torchrun
# adjust nl_probes/sft.py args as needed, e.g., --config config.yaml --batch-size 8
torchrun --nproc_per_node=$NUM_GPUS nl_probes/sft.py \
  --output_dir outputs/train \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 1 \
  --num_train_epochs 3