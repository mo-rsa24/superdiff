#!/usr/bin/env bash

# Full training run for Tuberculosis on the complete dataset.
# Trains for 150 epochs for convergence.
# Uses default sampler settings, optimized for final image quality.

export TASK="TB"
export OVERFIT_K="0"
export OVERFIT_ONE="0"
export EPOCHS="50"
export SCHEDULE="cosine"
export RUN_NAME="tb-full-train"
export WANDB_TAGS="slurm,full,tb"

# Use default sampler settings for the full run
export SAMPLER_NOISE_SCALE="1.0"
export ADD_FINAL_NOISE="0"

sbatch cxr_sde.slurm