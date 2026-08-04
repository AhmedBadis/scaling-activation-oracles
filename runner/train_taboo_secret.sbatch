#!/bin/bash
#SBATCH --partition=A40medium
#SBATCH --time-min=02:00:00
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --job-name=taboo-secret-train
#SBATCH --output=logs/taboo_secret_train/%j.out
#SBATCH --error=logs/taboo_secret_train/%j.err

# Usage:
#   sbatch runner/train_taboo_secret.sbatch [BASE_MODEL] [SECRET_WORD] [HINTS_FILE]
#
# BASE_MODEL   one of: qwen3-4b | qwen3.5-4b | qwen3-1.7b | qwen2.5-4b-instruct
#              (or any raw HF repo id) -- default: qwen3-4b
# SECRET_WORD  the word the suspect model will be fine-tuned to hint at but never say
#              -- default: tree
# HINTS_FILE   optional path to a JSON hints file (see train_taboo_secret.py docstring
#              for the schema). Omit for the built-in (weak) fallback hints.
#
# Examples:
#   sbatch runner/train_taboo_secret.sbatch qwen3-4b experiments/hints_tree.json
#   sbatch runner/train_taboo_secret.sbatch qwen3-1.7b gold
#   sbatch runner/train_taboo_secret.sbatch qwen3.5-4b moon

BASE_MODEL="${1:-qwen3-4b}"
SECRET_WORD="${2:-tree}"
HINTS_FILE="${3:-}"

module load Python/3.11.3-GCCcore-12.3.0
export CUDA_VISIBLE_DEVICES=0
nvidia-smi -L
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "BASE_MODEL=$BASE_MODEL SECRET_WORD=$SECRET_WORD HINTS_FILE=$HINTS_FILE"

cd "$HOME/activation_oracles"
mkdir -p logs/taboo_secret_train
source .venv/bin/activate

# Set up HuggingFace token (base model download only -- the secret LoRA this script
# produces is never uploaded anywhere, per the local-only workaround).
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

OUTPUT_DIR="checkpoints_taboo_secret_${BASE_MODEL//./_}_${SECRET_WORD}"

HINTS_ARGS=()
if [ -n "$HINTS_FILE" ]; then
    HINTS_ARGS=(--hints_file "$HINTS_FILE")
fi

python experiments/train_taboo_secret.py \
    --base_model "$BASE_MODEL" \
    --secret_word "$SECRET_WORD" \
    --output_dir "$OUTPUT_DIR" \
    "${HINTS_ARGS[@]}"

echo "Secret LoRA saved to $OUTPUT_DIR/final -- pass this dir as --secret_dir to eval_taboo_secret.py"