#!/bin/bash
#SBATCH --partition=A40medium
#SBATCH --time-min=23:30:00
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --job-name=train-multitarget
#SBATCH --output=logs/train-multitarget/%j.out
#SBATCH --error=logs/train-multitarget/%j.err

# Generalized Activation Oracle runner.
# Repeat --target_model in the desired order. Example below trains:
# target_a chunk 0, target_b chunk 0, target_c chunk 0, target_a chunk 1, ...
# Set N_CHUNKS=1 for sequential target training.

module load Python/3.11.3-GCCcore-12.3.0

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$HOME/activation_oracles"
mkdir -p logs/train-multitarget
source .venv/bin/activate

# Prefer pre-exported credentials in the job environment. Do not commit secrets
# into this runner.
if [[ -n "${HF_TOKEN:-}" ]]; then
  export HF_TOKEN
fi
if [[ -n "${WANDB_API_KEY:-}" ]]; then
  export WANDB_API_KEY
fi

pip install -e . --no-deps

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
  --target_model="Qwen/Qwen3-4B-Instruct" \
  --wandb_suffix="_16_PT_MLP_o1.7t4T4t4i" \
  --train_batch_size=4 \
  --gradient_accumulation_steps=4 \
  --n_chunks=16