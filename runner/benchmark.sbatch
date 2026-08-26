#!/bin/bash
#SBATCH --partition=A40medium
#SBATCH --time-min=00:30:00
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --job-name=benchmark_injection
#SBATCH --output=logs/benchmark/%j.out
#SBATCH --error=logs/benchmark/%j.err

module load Python/3.11.3-GCCcore-12.3.0
export CUDA_VISIBLE_DEVICES=0
nvidia-smi -L
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

cd $HOME/activation_oracles
source .venv/bin/activate

pip install -e . --no-deps

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ---- Benchmark configuration (edit as needed) ----
export MODEL_NAME="Qwen/Qwen3-8B"
export LAYER_IDX=1
export BATCH_SIZE=1
export SEQ_LEN=64
export N_ACT_POSITIONS=3
export STEERING_COEFFICIENT=1.0
export N_ITERS=50
export WARMUP=10
export DTYPE=bfloat16

echo "== env =="
which python
python --version

echo ""
echo "###### RUN 1: forward pass WITHOUT injection vs WITH injection ######"
python experiments/benchmark.py

echo ""
echo "###### Done. See RESULTS block above for the clear-text comparison. ######"
