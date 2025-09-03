#!/usr/bin/env bash
#SBATCH --job-name=cxr-sde
#SBATCH --partition=bigbatch
#SBATCH --time=72:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err


set -euo pipefail

# ----------------------------
# Pretty printing helpers
# ----------------------------
RED=$'\e[31m'; GRN=$'\e[32m'; YLW=$'\e[33m'; BLU=$'\e[34m'; MAG=$'\e[35m'; CYN=$'\e[36m'; BLD=$'\e[1m'; RST=$'\e[0m'
banner(){ printf "\n${BLU}${BLD}== %s ==${RST}\n" "$*"; }
kv(){ printf "  ${CYN}%-22s${RST} %s\n" "$1" "$2"; }
ok(){ printf "${GRN}%s${RST}\n" "$*"; }
warn(){ printf "${YLW}%s${RST}\n" "$*"; }
err(){ printf "${RED}%s${RST}\n" "$*"; }

START_TS=$(date -Iseconds)

# ----------------------------
# Defaults (override via env)
# ----------------------------
WORKDIR="${WORKDIR:-$PWD}"             # project root (folder that contains run/cxr.py)
PYFILE="${PYFILE:-run/cxr.py}"         # entrypoint
DATA_ROOT="/gluster/mmolefe/PhD/Super-Position Of Medical Imaging Diffusion Models For Disease Discovery/datasets/cleaned"

TASK="${TASK:-TB}"                     # TB | PNEUMONIA
SPLIT="${SPLIT:-train}"
IMG_SIZE="${IMG_SIZE:-256}"

# Overfit / tiny controls
OVERFIT_ONE="${OVERFIT_ONE:-1}"        # 1 to overfit a single image
OVERFIT_K="${OVERFIT_K:-0}"            # >0 to train on tiny-K
REPEAT_LEN="${REPEAT_LEN:-32768}"

# Model width
CHANNELS="${CHANNELS:-96,192,384,768}" # pass colon-separated if using --export; script converts below
EMBED_DIM="${EMBED_DIM:-256}"

# Schedule & sampler
SDE="${SDE:-VE}"                       # VE | VPSDE
SIGMA_MAX="${SIGMA_MAX:-12.0}"         # used if SDE==VE
SAMPLER="${SAMPLER:-pc}"               # pc | em | ode
NUM_STEPS="${NUM_STEPS:-500}"
SNR="${SNR:-0.16}"                     # only meaningful for PC
EPS="${EPS:-1e-3}"

# Training hygiene
LR="${LR:-2e-4}"
MIN_LR="${MIN_LR:-1e-5}"
WARMUP_STEPS="${WARMUP_STEPS:-1000}"
SCHEDULE="${SCHEDULE:-cosine}"         # const | cosine
GRAD_CLIP="${GRAD_CLIP:-1.0}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
EPOCHS="${EPOCHS:-80}"
BATCH_PER_DEVICE="${BATCH_PER_DEVICE:-4}"
SAMPLE_BATCH_SIZE="${SAMPLE_BATCH_SIZE:-8}"
SAMPLE_EVERY="${SAMPLE_EVERY:-1}"
SEED="${SEED:-0}"

# EMA
EMA_DECAY="${EMA_DECAY:-0.9995}"
USE_EMA_FOR_SAMPLING="${USE_EMA_FOR_SAMPLING:-1}"

# Experiment naming / output
OUTPUT_ROOT="${OUTPUT_ROOT:-runs}"
EXP_NAME="${EXP_NAME:-cxr_sde}"
RUN_NAME="${RUN_NAME:-}"               # optional
RESUME_DIR="${RESUME_DIR:-}"           # optional
SAMPLES_DIR="${SAMPLES_DIR:-}"         # optional

# WandB toggles
WANDB="${WANDB:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-cxr-sde}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_TAGS="${WANDB_TAGS:-slurm}"      # pass colon-separated if using --export
WANDB_ID="${WANDB_ID:-}"

# ----------------------------
# Environment & paths
# ----------------------------
mkdir -p "$WORKDIR/logs"
cd "$WORKDIR"

# Make PYTHONPATH robust even with `set -u`
export PYTHONPATH="$WORKDIR${PYTHONPATH:+:$PYTHONPATH}"

# Prefer CUDA for JAX, avoid huge preallocation
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TF_CPP_MIN_LOG_LEVEL=3

# Activate your env (choose one that exists on your cluster)
source ~/.bashrc
mamba activate "$ENV_NAME"
ok "Activated micromamba: $ENV_NAME"
# ----------------------------
# Normalize list-like vars
# (allow colon-separated in --export)
# ----------------------------
CHANNELS="${CHANNELS//:/,}"
WANDB_TAGS="${WANDB_TAGS//:/,}"

# ----------------------------
# Build args array safely
# ----------------------------
ARGS=(
  --data_root "$DATA_ROOT"
  --task "$TASK" --split "$SPLIT" --img_size "$IMG_SIZE"
  --channels "$CHANNELS" --embed_dim "$EMBED_DIM"
  --sde "$SDE" --sampler "$SAMPLER" --num_steps "$NUM_STEPS" --eps "$EPS"
  --lr "$LR" --min_lr "$MIN_LR" --warmup_steps "$WARMUP_STEPS" --schedule "$SCHEDULE"
  --grad_clip "$GRAD_CLIP" --weight_decay "$WEIGHT_DECAY"
  --epochs "$EPOCHS" --batch_per_device "$BATCH_PER_DEVICE" --sample_batch_size "$SAMPLE_BATCH_SIZE"
  --seed "$SEED" --sample_every "$SAMPLE_EVERY"
  --ema_decay "$EMA_DECAY"
  --output_root "$OUTPUT_ROOT" --exp_name "$EXP_NAME"
)

# VE-only flag
if [[ "${SDE^^}" == "VE" ]]; then
  ARGS+=( --sigma_max "$SIGMA_MAX" )
fi

# EMA sampling
if [[ "$USE_EMA_FOR_SAMPLING" == "1" ]]; then
  ARGS+=( --use_ema_for_sampling )
fi

# Overfit/tiny
if [[ "$OVERFIT_ONE" == "1" ]]; then
  ARGS+=( --overfit_one --repeat_len "$REPEAT_LEN" )
elif (( OVERFIT_K > 0 )); then
  ARGS+=( --overfit_k "$OVERFIT_K" )
fi

# Optional explicit samples dir / run name / resume
[[ -n "$SAMPLES_DIR" ]] && ARGS+=( --samples_dir "$SAMPLES_DIR" )
[[ -n "$RUN_NAME"    ]] && ARGS+=( --run_name "$RUN_NAME" )
[[ -n "$RESUME_DIR"  ]] && ARGS+=( --resume_dir "$RESUME_DIR" )

# WandB
if [[ "$WANDB" == "1" ]]; then
  ARGS+=( --wandb --wandb_project "$WANDB_PROJECT" )
  [[ -n "$WANDB_ENTITY" ]] && ARGS+=( --wandb_entity "$WANDB_ENTITY" )
  [[ -n "$WANDB_TAGS"   ]] && ARGS+=( --wandb_tags "$WANDB_TAGS" )
  [[ -n "$WANDB_ID"     ]] && ARGS+=( --wandb_id "$WANDB_ID" )
fi

# ----------------------------
# Info
# ----------------------------
banner "SLURM Job Info"
kv "JOB_ID"         "${SLURM_JOB_ID:-<none>}"
kv "JOB_NAME"       "${SLURM_JOB_NAME:-<none>}"
kv "Partition"      "${SLURM_JOB_PARTITION:-${SLURM_PARTITION:-<none>}}"
kv "Node"           "$(hostname)"
kv "When"           "$START_TS"

banner "Program"
kv "WORKDIR"        "$WORKDIR"
kv "PYFILE"         "$PYFILE"
kv "DATA_ROOT"      "$DATA_ROOT"

banner "Flags"
kv "TASK/SPLIT"     "$TASK / $SPLIT"
kv "IMG_SIZE"       "$IMG_SIZE"
kv "CHANNELS"       "$CHANNELS"
kv "SDE"            "$SDE"
kv "SIGMA_MAX"      "$SIGMA_MAX"
kv "SAMPLER"        "$SAMPLER"
kv "NUM_STEPS"      "$NUM_STEPS"
kv "SNR"            "$SNR"
kv "EPS"            "$EPS"
kv "EMA_DECAY"      "$EMA_DECAY"
kv "EMA_SAMPLING"   "$USE_EMA_FOR_SAMPLING"
kv "LR/SCHED"       "$LR / $SCHEDULE (warmup=$WARMUP_STEPS→min=$MIN_LR)"
kv "EPOCHS"         "$EPOCHS"
kv "BATCHxDEVS"     "$BATCH_PER_DEVICE x $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l || echo 1)"
kv "REPEAT_LEN"     "$REPEAT_LEN (of1=$OVERFIT_ONE, tinyK=$OVERFIT_K)"
kv "W&B"            "$WANDB (${WANDB_PROJECT}) tags=[$WANDB_TAGS]"

# ----------------------------
# Run
# ----------------------------
set +e
if [[ "${SAMPLER}" == "pc" ]]; then
  # PC needs SNR forwarded through env → ARGS (your sampler reads it from ARGS in cxr.py)
  ARGS+=( --snr "$SNR" )
fi

srun python -u "$PYFILE" "${ARGS[@]}"
RC=$?
set -e

END_TS=$(date -Iseconds)
ELAPSED="$((SECONDS/3600))h $(( (SECONDS%3600)/60 ))m $((SECONDS%60))s"

banner "Done"
kv "ExitCode" "$RC"
kv "Elapsed"  "$ELAPSED"
[[ $RC -eq 0 ]] && ok "Success." || { err "Failure (exit $RC)"; exit "$RC"; }
