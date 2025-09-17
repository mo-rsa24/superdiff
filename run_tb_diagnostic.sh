#!/usr/bin/env bash

# Diagnostic run for Tuberculosis on a subset of 512 images.
# Trains for only 20 epochs to quickly check for learning progress.
# Uses sampler arguments that encourage diversity to avoid mode collapse.

export TASK="TB"
export OVERFIT_K="512"
export OVERFIT_ONE="0"
export EPOCHS="20"
export SCHEDULE="cosine"
export RUN_NAME="tb-diagnostic-k512"
export WANDB_TAGS="slurm,diagnostic,tb"

# Use new sampler settings to prevent mode collapse on small dataset
export SAMPLER_NOISE_SCALE="1.1"
export ADD_FINAL_NOISE="1"

sbatch cxr_sde.slurm