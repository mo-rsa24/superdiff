#!/usr/bin/env bash
#
# Launches a full training job for the autoencoder using cxr_ae.slurm.
# This script disables the default "overfit" mode.
#

set -euo pipefail

# ----------------------------------------------------------------
#                ✅ --- USER CONFIGURATION --- ✅
# ----------------------------------------------------------------
# REQUIRED: Set the name of your conda or mamba environment
export ENV_NAME="jax115"

# --- Key Training Parameters ---
export TASK="TB"                         # Set the dataset task (TB or PNEUMONIA)
export EPOCHS=50                        # Set a higher number of epochs for a full run
export BATCH_PER_DEVICE=8                # Adjust batch size based on your GPU memory
export SAMPLE_EVERY=5                    # Sample every 5 epochs to save disk space
export WANDB_TAGS="ae-full-train:slurm"     # Set W&B tags (use colons to separate)

# ----------------------------------------------------------------
#                🚀 --- JOB LAUNCH LOGIC --- 🚀
# ----------------------------------------------------------------
# IMPORTANT: Disable debugging/overfit modes for a full training run
export OVERFIT_ONE=0  # Set to 0 to turn OFF overfitting
export OVERFIT_K=0    # Set to 0 to turn OFF tiny-K subset training

# Create a descriptive name for the experiment run
export RUN_NAME="ae_full_${TASK,,}_b${BATCH_PER_DEVICE}_$(date +%Y%m%d)"

# --- Submit to Slurm ---
echo "Submitting SLURM job for a FULL TRAINING run..."
echo "------------------------------------------------"
echo "  ▶️  Run Name:         $RUN_NAME"
echo "  ▶️  Conda Env:        $ENV_NAME"
echo "  ▶️  Task:             $TASK"
echo "  ▶️  Epochs:           $EPOCHS"
echo "  ▶️  Batch per Device: $BATCH_PER_DEVICE"
echo "  ▶️  Overfit Mode:     DISABLED"
echo "------------------------------------------------"

sbatch cxr_ae.slurm

echo "✅ Job successfully submitted!"