#!/bin/bash
#SBATCH --partition=A40medium
#SBATCH --time-min=23:30:00
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --job-name=train-o1.7t4T4
#SBATCH --output=logs/train-o1.7t4T4/%j.out
#SBATCH --error=logs/train-o1.7t4T4/%j.err

# Trains one oracle on two targets in four phases (see nl_probes/train_dual_target.py).

module load Python/3.11.3-GCCcore-12.3.0
export CUDA_VISIBLE_DEVICES=0
nvidia-smi -L
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

cd "$HOME/activation_oracles"
mkdir -p logs/train-o1.7t4T4
source .venv/bin/activate

python -c "from huggingface_hub import login; login(token='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')"

pip install -e . --no-deps

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_API_KEY="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"

echo "== env =="
which python
python - <<'PY'
import torch, os
print("torch:", getattr(torch, "__version__", None))
print("cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
PY

torchrun --nproc_per_node=1 nl_probes/train_dual_target.py \
  --oracle_model_name="Qwen/Qwen3-1.7B" \
  --target_a="Qwen/Qwen3-4B" \
  --target_b="Qwen/Qwen3.5-4B" \
  --wandb_suffix="_phased_dual_o1.7_t4_T4" \
  --train_batch_size=16 \
  --gradient_checkpointing \
  --gradient_accumulation_steps=1
