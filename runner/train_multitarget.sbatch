#!/bin/bash
#SBATCH --partition=A40medium
#SBATCH --time-min=23:30:00
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --job-name=train-phased-triple-o1.7t4T4t4i_16_PT_MLP_CoT
#SBATCH --output=logs/train-phased-triple-o1.7t4T4t4i_16_PT_MLP_CoT/%j.out
#SBATCH --error=logs/train-phased-triple-o1.7t4T4t4i_16_PT_MLP_CoT/%j.err

module load Python/3.11.3-GCCcore-12.3.0
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$HOME/activation_oracles"
mkdir -p logs/train-multitarget
source .venv/bin/activate

# Load private credentials
set -a
source "$HOME/activation_oracles/activation_oracles.env"
set +a
export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
export WANDB_API_KEY

python -c "from huggingface_hub import login; import os; login(token=os.environ['HF_TOKEN'])"

# Rebuild the package
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
  --target_model="Qwen/Qwen3-4B-Instruct-2507" \
  --projector_type="gated_residual_mlp" \
  --projector_init="random" \
  --projector_sharing=separate \
  --wandb_suffix="_phased_triple_o1.7_t4_T4_t4i_16_PT_MLP_CoT" \
  --train_batch_size=2 \
  --gradient_accumulation_steps=8 \
  --no-gradient_checkpointing \
  --eval_steps=0 \
  --save_steps=0 \
  --n_chunks=16 \
  --layer_percents 25 50 75 \
  --cot_percent=25 \
  --context_percent=47 \
  --classification_percent=24 \
  --spqa_percent=4 \
  --mixture_seed=42 \
  --mixture_total_samples=1000000 \