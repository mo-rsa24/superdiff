# CXR-SDE: SLURM Launch Recipes (copy-paste ready)

> **Assumptions**
>
> * You’re in the repo root (folder that contains `cxr_sde.slurm` and `run/`).
> * The SLURM script sets `PYTHONPATH="$WORKDIR:$WORKDIR/run:$PYTHONPATH"` and `cd "$WORKDIR"`.
> * Use **env-var prefix** style (newline-safe) and set a clear **job name** with `--job-name`.

---

## General pattern

```bash
WORKDIR="$PWD" MODE=<full|tiny|of1> TASK=<TB|PNEUMONIA> EXP_NAME=<name> \
BATCH_PER_DEVICE=<int> EPOCHS=<int> SAMPLE_EVERY=<int> SAMPLE_BATCH_SIZE=<int> \
USE_WANDB=1 WANDB_PROJECT=cxr-sde WANDB_TAGS="<comma-separated tags>" \
sbatch --job-name="<name>" cxr_sde.slurm
```

**Notes**

* `MODE=full` (full dataset), `tiny` (use `K=<N>`), `of1` (single-example).
* Resolution & model default to **256** and **64,128,256,512** in the script; override via `IMG_SIZE` / `CHANNELS` if needed.
* To run W\&B offline: set `USE_WANDB=0`.

---

## Recipes

### 1) Full-dataset **PNEUMONIA**

```bash
WORKDIR="$PWD" MODE=full TASK=PNEUMONIA EXP_NAME=pn256-full \
BATCH_PER_DEVICE=8 EPOCHS=200 SAMPLE_EVERY=5 SAMPLE_BATCH_SIZE=16 \
USE_WANDB=1 WANDB_PROJECT=cxr-sde WANDB_TAGS="pneumonia,full" \
sbatch --job-name="pn256-full" cxr_sde.slurm
```

### 2) Full-dataset **TB**

```bash
WORKDIR="$PWD" MODE=full TASK=TB EXP_NAME=tb256-full \
BATCH_PER_DEVICE=8 EPOCHS=200 SAMPLE_EVERY=5 SAMPLE_BATCH_SIZE=16 \
USE_WANDB=1 WANDB_PROJECT=cxr-sde WANDB_TAGS="tb,full" \
sbatch --job-name="tb256-full" cxr_sde.slurm
```

### 3) Tiny-K sanity (K=8) — **TB**

```bash
WORKDIR="$PWD" MODE=tiny K=8 TASK=TB EXP_NAME=tb256-tiny8 \
BATCH_PER_DEVICE=8 EPOCHS=50 SAMPLE_EVERY=1 SAMPLE_BATCH_SIZE=8 \
USE_WANDB=1 WANDB_PROJECT=cxr-sde WANDB_TAGS="tb,tiny8" \
sbatch --job-name="tb256-tiny8" cxr_sde.slurm
```

### 4) Tiny-K sanity (K=8) — **PNEUMONIA**

```bash
WORKDIR="$PWD" MODE=tiny K=8 TASK=PNEUMONIA EXP_NAME=pn256-tiny8 \
BATCH_PER_DEVICE=8 EPOCHS=50 SAMPLE_EVERY=1 SAMPLE_BATCH_SIZE=8 \
USE_WANDB=1 WANDB_PROJECT=cxr-sde WANDB_TAGS="pneumonia,tiny8" \
sbatch --job-name="pn256-tiny8" cxr_sde.slurm
```

### 5) Single-example overfit — **TB**

```bash
WORKDIR="$PWD" MODE=of1 TASK=TB EXP_NAME=tb256-of1 \
BATCH_PER_DEVICE=4 EPOCHS=50 SAMPLE_EVERY=1 SAMPLE_BATCH_SIZE=8 \
USE_WANDB=1 WANDB_PROJECT=cxr-sde WANDB_TAGS="tb,of1" \
sbatch --job-name="tb256-of1" cxr_sde.slurm
```

### 6) Single-example overfit — **PNEUMONIA**

```bash
WORKDIR="$PWD" MODE=of1 TASK=PNEUMONIA EXP_NAME=pn256-of1 \
BATCH_PER_DEVICE=4 EPOCHS=50 SAMPLE_EVERY=1 SAMPLE_BATCH_SIZE=8 \
USE_WANDB=1 WANDB_PROJECT=cxr-sde WANDB_TAGS="pneumonia,of1" \
sbatch --job-name="pn256-of1" cxr_sde.slurm
```

---

## Monitoring

```bash
squeue -u $USER                               # list jobs
tail -f logs/<jobname>-<jobid>.out            # live stdout
tail -f logs/<jobname>-<jobid>.err            # live stderr
scontrol show job <jobid>                     # details
scancel <jobid>                               # cancel
```

---

## Tips & troubleshooting

* Submit from anywhere with an absolute path:

  ```bash
  WORKDIR="/abs/path/to/SUPERDIFF" MODE=full TASK=TB EXP_NAME=tb256-full \
  ... sbatch --job-name="tb256-full" /abs/path/to/SUPERDIFF/cxr_sde.slurm
  ```
* If you see `ModuleNotFoundError: diffusion`, ensure the script sets:

  ```bash
  export PYTHONPATH="$WORKDIR:$WORKDIR/run:${PYTHONPATH:-}"
  ```
* Keep job names short & ASCII; they flow into log filenames (`%x-%j`).
