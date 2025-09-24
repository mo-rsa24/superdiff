#!/usr/bin/env bash
#SBATCH --job-name=compose-sde
#SBATCH --partition=gpu          # May not need bigbatch for inference
#SBATCH --time=01:00:00          # Composition is much faster
#SBATCH --gres=gpu:1             # Usually only need 1 GPU for inference
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
source ~/.bashrc
mamba activate jax115 # Or your conda/mamba env

# --- Get variables from launcher script ---
WORKDIR="${WORKDIR:-$PWD}"
PYFILE="new_compose.py"
RUN_DIR="${RUN_DIR}"
LABEL_BASE="${LABEL_BASE:-NORMAL}"
LABEL_A="${LABEL_A:-PNEUMONIA}"
LABEL_B="${LABEL_B:-TB}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-1.0}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_STEPS="${NUM_STEPS:-1000}"

cd "$WORKDIR"
export PYTHONPATH="$WORKDIR"

# --- Build arguments for new_compose.py ---
ARGS=(
  --run_dir "$RUN_DIR"
  --label_base "$LABEL_BASE"
  --label_a "$LABEL_A"
  --label_b "$LABEL_B"
  --guidance_scale "$GUIDANCE_SCALE"
  --batch_size "$BATCH_SIZE"
  --num_steps "$NUM_STEPS"
)

echo "========================================================"
echo "Starting composition job: ${SLURM_JOB_ID}"
echo "Running script: ${PYFILE}"
echo "Arguments: ${ARGS[@]}"
echo "========================================================"

srun python -u "$PYFILE" "${ARGS[@]}"

echo "Composition finished."