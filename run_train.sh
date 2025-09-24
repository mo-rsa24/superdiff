#!/usr/bin/env bash

# Launcher for conditional model training.
# Edit these variables to configure your run.

export PYFILE="new_cxr.py"
export RUN_NAME="conditional-cxr-v1"
export DATA_ROOT="/path/to/your/unified/dataset" # IMPORTANT: Update this path

export IMG_SIZE="128"
export NUM_CLASSES="3" # NORMAL, PNEUMONIA, TB
export CHANNELS="64,128,256"

export EPOCHS="200"
export BATCH_PER_DEVICE="8"
export LR="2e-4"
export SAMPLE_EVERY="10"
export CKPT_EVERY="20"

export WANDB_PROJECT="conditional-cxr-sde"
export WANDB_TAGS="slurm,conditional,v1"

# Submit the job to SLURM
sbatch cxr_sde.slurm