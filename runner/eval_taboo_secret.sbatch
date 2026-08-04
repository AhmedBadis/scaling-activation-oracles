#!/bin/bash
#SBATCH --partition=A40medium
#SBATCH --time-min=01:00:00
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --job-name=taboo-secret-eval
#SBATCH --output=logs/taboo_secret_eval/%j.out
#SBATCH --error=logs/taboo_secret_eval/%j.err

# Usage:
#   sbatch runner/eval_taboo_secret.sbatch ORACLE_MODEL_NAME VERBALIZER_LORA SECRET_DIR [PROJECTOR_SAVE_DIR]
#
# ORACLE_MODEL_NAME    e.g. Qwen/Qwen3-1.7B
# VERBALIZER_LORA      local path to your trained oracle LoRA, e.g.
#                      checkpoints_phased_dual_o1.7_t4_T4/final
# SECRET_DIR           output dir from train_taboo_secret.py, e.g.
#                      checkpoints_taboo_secret_qwen3-4b_tree
# PROJECTOR_SAVE_DIR   only needed if oracle and suspect base model have different
#                      hidden sizes (cross-model run) -- omit otherwise
#
# Re-run this against a different oracle checkpoint or a different secret model
# without retraining either one -- just change the arguments.
#
# Examples:
#   sbatch runner/eval_taboo_secret.sbatch Qwen/Qwen3-1.7B \
#       checkpoints_latentqa_cls_past_lens_oracle_Qwen3-1.7B_target_Qwen3-4B/final \
#       checkpoints_taboo_secret_qwen3-4b_tree \
#       checkpoints_latentqa_cls_past_lens_oracle_Qwen3-1.7B_target_Qwen3-4B
#
#   sbatch runner/eval_taboo_secret.sbatch Qwen/Qwen3-4B \
#       checkpoints_latentqa_cls_past_lens_oracle_Qwen3-4B_target_Qwen3-4B/final \
#       checkpoints_taboo_secret_qwen3-4b_tree

ORACLE_MODEL_NAME="${1:?Usage: sbatch eval_taboo_secret.sbatch ORACLE_MODEL_NAME VERBALIZER_LORA SECRET_DIR [PROJECTOR_SAVE_DIR]}"
VERBALIZER_LORA="${2:?Missing VERBALIZER_LORA}"
SECRET_DIR="${3:?Missing SECRET_DIR}"
PROJECTOR_SAVE_DIR="${4:-}"

module load Python/3.11.3-GCCcore-12.3.0
export CUDA_VISIBLE_DEVICES=0
nvidia-smi -L
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "ORACLE_MODEL_NAME=$ORACLE_MODEL_NAME"
echo "VERBALIZER_LORA=$VERBALIZER_LORA"
echo "SECRET_DIR=$SECRET_DIR"
echo "PROJECTOR_SAVE_DIR=$PROJECTOR_SAVE_DIR"

cd "$HOME/activation_oracles"
mkdir -p logs/taboo_secret_eval
source .venv/bin/activate

# Base models are still pulled from HF; nothing about the secret or the oracle LoRA is.
python -c "from huggingface_hub import login; login(token='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')"

pip install -e . --no-deps

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "== env =="
which python
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
PY

PROJ_ARGS=()
if [ -n "$PROJECTOR_SAVE_DIR" ]; then
    PROJ_ARGS=(--projector_save_dir "$PROJECTOR_SAVE_DIR")
fi

python experiments/eval_taboo_secret.py \
    --oracle_model_name "$ORACLE_MODEL_NAME" \
    --verbalizer_lora "$VERBALIZER_LORA" \
    --secret_dir "$SECRET_DIR" \
    "${PROJ_ARGS[@]}"

echo "Results written under experiments/eval_taboo_secret/"