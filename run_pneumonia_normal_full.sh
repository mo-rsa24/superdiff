#!/usr/bin/env bash

# Full training run for Pneumonia on the NORMAL subset of the data.
# Trains for 250 epochs for convergence.
# Uses default sampler settings, optimized for final image quality.

export TASK="PNEUMONIA"
export DISEASE="0" # Use NORMAL subset
export OVERFIT_K="0"
export OVERFIT_ONE="0"
export EPOCHS="250"
export SCHEDULE="cosine"
export RUN_NAME="pneumonia-normal-full-train"
export WANDB_TAGS="slurm,full,pneumonia,normal"
export SAMPLE_EVERY="10"
export CKPT_EVERY="10"
# Use default sampler settings for the full run
export SAMPLER_NOISE_SCALE="1.0"
export ADD_FINAL_NOISE="0"

sbatch cxr_sde.slurm