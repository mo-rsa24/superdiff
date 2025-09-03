#!/usr/bin/env bash
#
# sbatch cxr_sde.slurm
#
#SBATCH --job-name=cxr-sde
#SBATCH --partition=bigbatch
#SBATCH --time=72:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# =========================
# Pretty colors & helpers
# =========================
if [[ -n "${NO_COLOR:-}" ]]; then
  RED=""; GRN=""; YLW=""; BLU=""; CYN=""; MAG=""; BLD=""; DIM=""; RST=""
else
  RED=$'\033[1;31m'; GRN=$'\033[1;32m'; YLW=$'\033[1;33m'
  BLU=$'\033[1;34m'; CYN=$'\033[1;36m'; MAG=$'\033[1;35m'
  BLD=$'\033[1m';    DIM=$'\033[2m';    RST=$'\033[0m'
fi

banner() {
  local msg="$1"
  echo -e "${BLU}${BLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RST}"
  echo -e "${BLU}${BLD}▶ ${msg}${RST}"
  echo -e "${BLU}${BLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RST}"
}

kv() {
  printf "%s%-22s%s %s\n" "${CYN}${BLD}" "$1" "${RST}" "$2"
}

ok()   { echo -e "${GRN}✔${RST} $*"; }
warn() { echo -e "${YLW}⚠${RST} $*"; }
err()  { echo -e "${RED}✖${RST} $*"; }

# =========================
# User config (override with sbatch --export=ALL,VAR=...)
# =========================
WORKDIR="${WORKDIR:-$HOME/Playground/PhD/superdiff}"
PYFILE="${PYFILE:-run/cxr.py}"

DATA_ROOT="/gluster/mmolefe/PhD/Super-Position Of Medical Imaging Diffusion Models For Disease Discovery/datasets/cleaned"
TASK="${TASK:-TB}"
SPLIT="${SPLIT:-train}"

MODE="${MODE:-of1}"         # of1 | tiny | full
K="${K:-8}"

IMG_SIZE="${IMG_SIZE:-256}"
CHANNELS="${CHANNELS:-64,128,256,512}"
LR="${LR:-2e-4}"
BATCH_PER_DEVICE="${BATCH_PER_DEVICE:-4}"
EPOCHS="${EPOCHS:-50}"
SAMPLE_EVERY="${SAMPLE_EVERY:-1}"
SAMPLE_BATCH_SIZE="${SAMPLE_BATCH_SIZE:-8}"

EXP_NAME="${EXP_NAME:-tb256-${MODE}}"
RUN_NAME="${RUN_NAME:-}"

USE_WANDB="${USE_WANDB:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-cxr-sde}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_TAGS="${WANDB_TAGS:-slurm}"

ENV_NAME="${ENV_NAME:-jax115}"

export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TF_CPP_MIN_LOG_LEVEL=1

# =========================
# Preamble
# =========================
set -euo pipefail
SECONDS=0
mkdir -p "$WORKDIR/logs"

banner "SLURM Job Info"
kv "SLURM_JOB_ID"        "${SLURM_JOB_ID:-<none>}"
kv "SLURM_JOB_NAME"      "${SLURM_JOB_NAME:-<none>}"
kv "Partition"           "${SLURM_JOB_PARTITION:-${SLURM_PARTITION:-<none>}}"
kv "Submit host"         "$(hostname)"
kv "When"                "$(date -Iseconds)"

banner "Paths & Program"
kv "WORKDIR"             "$WORKDIR"
kv "PYFILE"              "$PYFILE"
kv "CWD (initial)"       "$(pwd)"

# =========================
# Environment
# =========================
banner "Environment Setup"
cd "$WORKDIR"
export PYTHONPATH="$WORKDIR:$WORKDIR/run:${PYTHONPATH:-}"
echo "[PYTHONPATH] $PYTHONPATH"
ok "cd $WORKDIR"

source ~/.bashrc
mamba activate "$ENV_NAME"
ok "Activated micromamba: $ENV_NAME"

if command -v nvidia-smi >/dev/null 2>&1; then
  banner "GPU Inventory"
  nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader || true
else
  warn "nvidia-smi not found"
fi

banner "Python / JAX Summary"
python - <<'PYINFO' || true
import platform, sys
print(f"Python: {platform.python_version()}  ({sys.executable})")
try:
  import jax, jaxlib
  from jax.lib import xla_bridge
  print(f"JAX:    {jax.__version__}")
  print(f"jaxlib: {jaxlib.__version__}")
  print(f"Backend: {xla_bridge.get_backend().platform}")
  print("Devices:", [str(d) for d in jax.devices()])
except Exception as e:
  print("JAX summary failed:", e)
PYINFO

# =========================
# Effective arguments (build safely)
# =========================
declare -a ARGS=(
  --data_root "$DATA_ROOT"
  --task "$TASK"
  --split "$SPLIT"
  --img_size "$IMG_SIZE"
  --channels "$CHANNELS"
  --lr "$LR"
  --batch_per_device "$BATCH_PER_DEVICE"
  --sample_every "$SAMPLE_EVERY"
  --sample_batch_size "$SAMPLE_BATCH_SIZE"
  --epochs "$EPOCHS"
  --exp_name "$EXP_NAME"
)

case "$MODE" in
  of1)  ARGS+=( --overfit_one --eval_mse_to_target ) ;;
  tiny) ARGS+=( --overfit_k "$K" ) ;;
  full) : ;;
  *)    err "Unknown MODE='$MODE' (expected: of1 | tiny | full)"; exit 1 ;;
esac

if [[ "$USE_WANDB" == "1" ]]; then
  ARGS+=( --wandb --wandb_project "$WANDB_PROJECT" )
  [[ -n "$WANDB_ENTITY" ]] && ARGS+=( --wandb_entity "$WANDB_ENTITY" )
  [[ -n "$WANDB_TAGS" ]]   && ARGS+=( --wandb_tags "$WANDB_TAGS" )
else
  export WANDB_MODE=offline
fi

[[ -n "$RUN_NAME" ]] && ARGS+=( --run_name "$RUN_NAME" )

banner "Experiment Variables"
kv "MODE"                 "$MODE"
kv "DATA_ROOT"            "$DATA_ROOT"
kv "TASK/SPLIT"           "$TASK / $SPLIT"
kv "IMG_SIZE"             "$IMG_SIZE"
kv "CHANNELS"             "$CHANNELS"
kv "LR"                   "$LR"
kv "BATCH_PER_DEVICE"     "$BATCH_PER_DEVICE"
kv "EPOCHS"               "$EPOCHS"
kv "SAMPLE_EVERY"         "$SAMPLE_EVERY"
kv "SAMPLE_BATCH_SIZE"    "$SAMPLE_BATCH_SIZE"
kv "EXP_NAME"             "$EXP_NAME"
kv "RUN_NAME"             "${RUN_NAME:-<auto>}"
kv "WANDB"                "$([[ ${USE_WANDB} == 1 ]] && echo enabled || echo disabled)"
kv "WANDB_PROJECT"        "$WANDB_PROJECT"
kv "WANDB_ENTITY"         "${WANDB_ENTITY:-<none>}"
kv "WANDB_TAGS"           "${WANDB_TAGS:-<none>}"

banner "Launch Command"
echo -e "${MAG}${BLD}srun python -u ${PYFILE} ${RST}${DIM}\\"
# SAFE: print one arg per line, no pair assumption
for arg in "${ARGS[@]}"; do
  printf "  %s%q%s \\\n" "${DIM}" "$arg" "${RST}"
done
echo -e "${RST}"

# =========================
# Run
# =========================
START_TS=$(date -Iseconds)
ok "Starting at ${START_TS}"

set +e
srun python -u "$PYFILE" "${ARGS[@]}"
RC=$?
set -e

END_TS=$(date -Iseconds)
ELAPSED="$((SECONDS/3600))h $(( (SECONDS%3600)/60 ))m $((SECONDS%60))s"

banner "Job Completed"
kv "Start"    "$START_TS"
kv "End"      "$END_TS"
kv "Elapsed"  "$ELAPSED"
kv "ExitCode" "$RC"
if [[ $RC -eq 0 ]]; then ok "Success."; else err "Failure (exit $RC). Check logs."; fi
