
#!/bin/bash
#SBATCH --partition=A40short
#SBATCH --time-min=7:30:00
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --job-name=taboo
#SBATCH --output=logs/taboo/%j.out
#SBATCH --error=logs/taboo/%j.err

module load Python/3.11.3-GCCcore-12.3.0
export CUDA_VISIBLE_DEVICES=0
nvidia-smi -L
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# recreate venv with the module python so shared libs match
cd $HOME/activation_oracles
source .venv/bin/activate

# Load private credentials
source "$HOME/.secrets/activation_oracles.env"
export WANDB_API_KEY
export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"

# Set up HuggingFace token
python -c "from huggingface_hub import login; import os; login(token=os.environ['HF_TOKEN'])"

# Rebuild the package with our fixes
pip install -e . --no-deps

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

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

# run demo
python experiments/taboo_open_ended_eval.py
