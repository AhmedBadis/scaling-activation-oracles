#!/bin/bash
#SBATCH --partition=A40medium
#SBATCH --time-min=23:30:00
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --job-name=train-multitarget
#SBATCH --output=logs/train-multitarget/%j.out
#SBATCH --error=logs/train-multitarget/%j.err

module load Python/3.11.3-GCCcore-12.3.0
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$HOME/activation_oracles"
mkdir -p logs/train-multitarget
source .venv/bin/activate

python -c "from huggingface_hub import login; login(token='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')"

# Rebuild the package with our fixes
pip install -e . --no-deps

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_API_KEY="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"

nvidia-smi -L
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
which python
python - <<'PY'
import torch
print("torch:", getattr(torch, "__version__", None))
print("cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
PY

torchrun --nproc_per_node=1 nl_probes/train_multitarget.py \
  --oracle_model_name="Qwen/Qwen3-1.7B" \
  --target_model="Qwen/Qwen3-4B" \
  --target_model="Qwen/Qwen3.5-4B" \
  --wandb_suffix="_16_PT_MLP_o1.7t4T4" \
  --train_batch_size=8 \
  --gradient_accumulation_steps=2 \
  --n_chunks=16 \
  --layer_percents 25 50 75 \
  --mixture_profile="${AO_MIXTURE_PROFILE:-legacy}" \
  --mixture_seed="${AO_MIXTURE_SEED:-42}" \
  --mixture_total_samples="${AO_MIXTURE_TOTAL_SAMPLES:-400000}"