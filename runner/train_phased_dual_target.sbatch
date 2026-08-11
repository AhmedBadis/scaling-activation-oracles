#!/bin/bash
#SBATCH --partition=A40medium
#SBATCH --time-min=23:30:00
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --job-name=train-phased-dual-o1.7t4T4-debug
#SBATCH --output=logs/train-phased-dual-o1.7t4T4-debug/%j.out
#SBATCH --error=logs/train-phased-dual-o1.7t4T4-debug/%j.err

module load Python/3.11.3-GCCcore-12.3.0

export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=1
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL

nvidia-smi -L
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

cd "$HOME/activation_oracles"
mkdir -p logs/train-phased-dual-o1.7t4T4-debug
source .venv/bin/activate

python -c "from huggingface_hub import login; login(token='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')"

pip install -e . --no-deps

# keep allocator but make it visible
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.8,max_split_size_mb:128

export WANDB_API_KEY="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"

echo "== env =="
which python

python - <<'PY'
import torch, os
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    print("mem_alloc_GB:", torch.cuda.memory_allocated()/1e9)
    print("mem_reserved_GB:", torch.cuda.memory_reserved()/1e9)
    print("cuda capability:", torch.cuda.get_device_capability(0))
PY

echo "== ECC status BEFORE run =="
nvidia-smi -q -d ECC || true

echo "== starting torchrun =="
torchrun --nproc_per_node=1 \
  nl_probes/train_phased_dual_target.py \
  --oracle_model_name="Qwen/Qwen3-1.7B" \
  --target_a="Qwen/Qwen3-4B" \
  --target_b="Qwen/Qwen3.5-4B" \
  --wandb_suffix="_phased_dual_o1.7_t4_T4_debug" \
  --train_batch_size=16 \
  --gradient_accumulation_steps=1 \
  --n_chunks=16

echo "== ECC status AFTER run =="
nvidia-smi -q -d ECC || true