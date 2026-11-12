#!/bin/bash
#SBATCH --job-name=nd2_stitch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-7%4
#SBATCH --output=nd2_%A_%a.out
#SBATCH --error=nd2_%A_%a.err

set -euo pipefail
if (( $# != 2 )); then
    echo 'Usage: sbatch run_nd2_stitch.sbatch PLAN.json OUTPUT_DIRECTORY' >&2
    exit 2
fi
# Activate your environment before sbatch; override CODE_DIR if necessary.
CODE_DIR=${CODE_DIR:-$SLURM_SUBMIT_DIR}
PYTHON_BIN=${PYTHON_BIN:-python}
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
# Keep reader concurrency conservative on shared storage.
WORKERS=${WORKERS:-2}
SCRATCH_ROOT=${SLURM_TMPDIR:-${TMPDIR:-/tmp}}
TASK_SCRATCH=$(mktemp -d "$SCRATCH_ROOT/nd2-${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID}.XXXXXX")
trap 'rmdir "$TASK_SCRATCH" 2>/dev/null || true' EXIT
# Array IDs must be contiguous, starting at zero (for example 0-7%4).
if (( ${SLURM_ARRAY_TASK_MIN:-0} != 0 || ${SLURM_ARRAY_TASK_STEP:-1} != 1 ||
      ${SLURM_ARRAY_TASK_MAX:-0} + 1 != ${SLURM_ARRAY_TASK_COUNT:-1} )); then
    echo 'Use a contiguous zero-based array such as --array=0-7%4' >&2
    exit 2
fi
"$PYTHON_BIN" -u "$CODE_DIR/stitch_nd2.py" stitch \
    --plan "$1" --output "$2" --scratch "$TASK_SCRATCH" \
    --workers "$WORKERS" --shards "${SLURM_ARRAY_TASK_COUNT:-1}" \
    --shard-index "${SLURM_ARRAY_TASK_ID:-0}" --resume
