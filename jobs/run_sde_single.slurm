#!/usr/bin/env bash
#SBATCH --job-name=superdiff-sde
#SBATCH --partition=bigbatch
#SBATCH --time=72:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

# Pretty prints (optional)
NC='\033[0m'; CYAN='\033[1;36m'; GREEN='\033[1;32m'; BOLD='\033[1m'
info()     { echo -e "\n${CYAN}ℹ️  $*${NC}"; }
print_kv() { printf "${BOLD}%-18s${NC} %s\n" "$1:" "$2"; }

mkdir -p logs

# Env names configurable via --export=...
ENV_NAME="${ENV_NAME:-jax115}"
PYTHON_EXE="${PYTHON_EXE:-python3}"
EXTRA_CLI="${EXTRA_CLI:-}"

# 1) Activate your env
source ~/.bashrc
mamba activate "${ENV_NAME}" 2>/dev/null

# 2) JAX/XLA knobs
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
export TF_CPP_MIN_LOG_LEVEL=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_THREADING_LAYER=GNU

# 3) Paths (SAFE with set -u)
export PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# 4) Context
info "SLURM CONTEXT"
print_kv "Job"        "${SLURM_JOB_NAME} (${SLURM_JOB_ID:-n/a})"
print_kv "Node list"  "${SLURM_NODELIST:-n/a}"
print_kv "Partition"  "${SLURM_JOB_PARTITION:-n/a}"
print_kv "CPUs"       "${SLURM_CPUS_PER_TASK:-n/a}"
print_kv "Mem/node"   "${SLURM_MEM_PER_NODE:-n/a}"
print_kv "Submit dir" "${SLURM_SUBMIT_DIR:-$PWD}"
print_kv "PROJECT_ROOT" "$PROJECT_ROOT"
print_kv "PYTHONPATH"   "${PYTHONPATH}"

command -v "${PYTHON_EXE}" || true
${PYTHON_EXE} --version || true
ls -l "$PROJECT_ROOT/run" | sed -n '1,80p' || true

${PYTHON_EXE} - <<'PY'
try:
    import jax
    print("JAX devices:", jax.devices())
except Exception as e:
    print("JAX device check failed:", e)
PY

# 5) Launch (use srun for proper Slurm binding)
info "LAUNCH"
print_kv "Command" "srun ${PYTHON_EXE} run/sde.py ${EXTRA_CLI}"
cd "$PROJECT_ROOT"
srun "${PYTHON_EXE}" run/sde.py ${EXTRA_CLI}
