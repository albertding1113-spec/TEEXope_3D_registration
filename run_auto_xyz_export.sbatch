#!/bin/bash
#SBATCH --job-name=xyzflex
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=7-00:00:00
#SBATCH --output=auto_xyz_export_%j.log
set -euo pipefail
# Activate your existing conda/venv environment before executing, if required.
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=4
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
python -u run_auto_xyz_export.py "$@"
