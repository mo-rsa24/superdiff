#!/usr/bin/env bash
# submit_sde.sh — launch or dry-run a JAX Slurm job for run/sde.py
# Usage examples:
#   ./submit_sde.sh --dry-run
#   ./submit_sde.sh --partition gpu --gpus 1 --time 06:00:00 --mem 24G --cpus 4 --sanity
#   ./submit_sde.sh --partition biggpu --gpus 4 --time 12:00:00 --mem 64G --cpus 16

set -euo pipefail

# === Pretty Print Helpers ===
NC='\033[0m' # No Color
CYAN='\033[1;36m'
GREEN='\033[1;32m'
YELLOW='\033[1;33m'
RED='\033[1;31m'
BOLD='\033[1m'
DIM='\033[2m'

info()     { echo -e "\n${CYAN}ℹ️  $*${NC}"; }
success()  { echo -e "${GREEN}✅ $*${NC}"; }
warn()     { echo -e "${YELLOW}⚠️  $*${NC}"; }
error()    { echo -e "\n${RED}❌ $*${NC}"; exit 1; }
print_kv() { printf "${BOLD}%-20s${NC} %s\n" "$1:" "$2"; }

echo -e "${BOLD}${CYAN}\n🚀 Slurm Launcher for JAX (sde.py)${NC}\n"

# === Defaults (override with flags) ===
PARTITION="${PARTITION:-bigbatch}"     # set to Wits partition, e.g. bigbatch / biggpu / stampede
GPUS="${GPUS:-1}"
CPUS="${CPUS:-4}"
MEM="${MEM:-24G}"
TIME="${TIME:-06:00:00}"
JOB_NAME="${JOB_NAME:-superdiff-sde}"
ENV_NAME="${ENV_NAME:-jax115}"    # your conda/mamba environment
SANITY=0
DRY_RUN=0
LOG_DIR="${LOG_DIR:-logs}"
JOBS_DIR="${JOBS_DIR:-jobs}"
PYTHON_EXE="${PYTHON_EXE:-python3}"
EXTRA_ARGS=()

# === Parse flags ===
while [[ $# -gt 0 ]]; do
  case "$1" in
    --partition) PARTITION="$2"; shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --cpus) CPUS="$2"; shift 2 ;;
    --mem) MEM="$2"; shift 2 ;;
    --time) TIME="$2"; shift 2 ;;
    --job-name) JOB_NAME="$2"; shift 2 ;;
    --env) ENV_NAME="$2"; shift 2 ;;
    --python) PYTHON_EXE="$2"; shift 2 ;;
    --sanity) SANITY=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --) shift; EXTRA_ARGS+=("$@"); break ;;
    *) EXTRA_ARGS+=("$1"); shift ;;
  esac
done

PROJECT_ROOT="$(pwd)"
SLURM_FILE="${JOBS_DIR}/run_sde.slurm"

info "Validating repository layout and paths"
[[ -d "${JOBS_DIR}" ]] || { info "Creating ${JOBS_DIR}/"; mkdir -p "${JOBS_DIR}"; }
[[ -d "${LOG_DIR}"  ]] || { info "Creating ${LOG_DIR}/";  mkdir -p "${LOG_DIR}"; }

# Check presence of your script and packages
[[ -f "${PROJECT_ROOT}/run/sde.py" ]] || error "Missing ${PROJECT_ROOT}/run/sde.py"
[[ -d "${PROJECT_ROOT}/diffusion" ]] || warn "Couldn't find '${PROJECT_ROOT}/diffusion' (imports may still work if your code doesn't need it)"
[[ -d "${PROJECT_ROOT}/models"    ]] || warn "Couldn't find '${PROJECT_ROOT}/models' (imports may still work if your code doesn't need it)"

print_kv "Project root" "${PROJECT_ROOT}"
print_kv "Slurm file"   "${SLURM_FILE}"
print_kv "Logs dir"     "${LOG_DIR}"