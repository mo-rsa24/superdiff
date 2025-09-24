#!/usr/bin/env bash

# Launcher for model composition.
# Edit these variables to configure your composition run.

# IMPORTANT: Update this to the full path of your FINISHED training run
export RUN_DIR="runs/conditional-cxr-v1_20250922-213441"

export LABEL_BASE="NORMAL"
export LABEL_A="PNEUMONIA"
export LABEL_B="TB"

export GUIDANCE_SCALE="1.5"
export BATCH_SIZE="16"
export NUM_STEPS="1000"

# Submit the job to SLURM
sbatch compose.slurm