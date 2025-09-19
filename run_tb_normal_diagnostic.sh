#!/usr/bin/env bash

# Diagnostic run for Tuberculosis on a subset of 512 NORMAL images.
# Trains for only 100 epochs to quickly check for learning progress.
# Uses sampler arguments that encourage diversity to avoid mode collapse.

export TASK="TB"
export DISEASE="0" # Use NORMAL subset
export OVERFIT_K="512"
export OVERFIT_ONE="0"
export EPOCHS="100"
export SAMPLE_EVERY="10"
export SCHEDULE="cosine"
export RUN_NAME="tb-normal-diagnostic-k512"
export WANDB_TAGS="slurm,diagnostic,tb,normal"
export SDE="VPSDE"
# Use new sampler settings to prevent mode collapse on small dataset
export SAMPLER_NOISE_SCALE="1.1"
export ADD_FINAL_NOISE="1"

sbatch cxr_sde.slurm