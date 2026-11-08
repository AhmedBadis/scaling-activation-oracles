
#!/bin/bash
#SBATCH --partition=A40medium
#SBATCH --time-min=23:30:00
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --job-name=train-sequential-dual-o1.7t4T4_1_ST_MLP
#SBATCH --output=logs/train-sequential-dual-o1.7t4T4_1_ST_MLP/%j.out
#SBATCH --error=logs/train-sequential-dual-o1.7t4T4_1_ST_MLP/%j.err

module load Python/3.11.3-GCCcore-12.3.0
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$HOME/activation_oracles"
mkdir -p logs/train-multitarget
source .venv/bin/activate

# Load private credentials
source "$HOME/.secrets/activation_oracles.env"
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
  --projector_type="gated_residual_mlp" \
  --projector_init="random" \
  --projector_sharing=shared \
  --wandb_suffix="_sequential_dual_o1.7_t4_T4_1_ST_MLP" \
  --train_batch_size=2 \
  --gradient_accumulation_steps=8 \
  --no-gradient_checkpointing \
  --eval_steps=0 \
  --save_steps=10000 \
  --n_chunks=1 \