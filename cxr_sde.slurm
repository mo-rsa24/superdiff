#!/usr/bin/env bash
#SBATCH --job-name=cond-cxr-sde
#SBATCH --partition=bigbatch
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:4
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
source ~/.bashrc
mamba activate jax115 # Or your conda/mamba env

# --- Get variables from launcher script ---
WORKDIR="${WORKDIR:-$PWD}"
PYFILE="${PYFILE:-new_cxr.py}"
DATA_ROOT="${DATA_ROOT}"
RUN_NAME="${RUN_NAME:-}"
IMG_SIZE="${IMG_SIZE:-128}"
NUM_CLASSES="${NUM_CLASSES:-3}"
CHANNELS="${CHANNELS:-64,128,256}"
EPOCHS="${EPOCHS:-200}"
BATCH_PER_DEVICE="${BATCH_PER_DEVICE:-8}"
LR="${LR:-2e-4}"
SAMPLE_EVERY="${SAMPLE_EVERY:-10}"
CKPT_EVERY="${CKPT_EVERY:-20}"
WANDB_PROJECT="${WANDB_PROJECT:-conditional-cxr-sde}"
WANDB_TAGS="${WANDB_TAGS:-slurm}"

cd "$WORKDIR"
export PYTHONPATH="$WORKDIR"

# --- Build arguments for new_cxr.py ---
ARGS=(
  --data_root "$DATA_ROOT"
  --img_size "$IMG_SIZE"
  --num_classes "$NUM_CLASSES"
  --channels "$CHANNELS"
  --epochs "$EPOCHS"
  --batch_per_device "$BATCH_PER_DEVICE"
  --lr "$LR"
  --sample_every "$SAMPLE_EVERY"
  --ckpt_every "$CKPT_EVERY"
  --exp_name "${RUN_NAME}"
  # Add other flags from new_cxr.py's parser as needed
)

# --- WandB ---
ARGS+=( --wandb --wandb_project "$WANDB_PROJECT" --wandb_tags "$WANDB_TAGS" )

echo "========================================================"
echo "Starting job: ${SLURM_JOB_ID}"
echo "Running script: ${PYFILE}"
echo "Arguments: ${ARGS[@]}"
echo "========================================================"

srun python -u "$PYFILE" "${ARGS[@]}"

echo "Job finished."