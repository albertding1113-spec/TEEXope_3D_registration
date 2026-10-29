#!/bin/bash
#SBATCH --job-name=rpl10a_export
#SBATCH --partition=fat2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=30-00:00:00
#SBATCH --output=/HWFS/zhaohu_lab/dingjiayi/task/output/261028_export_rpl10a_%j.out

set -euo pipefail
# Activate your Python conda/venv environment here if needed.
# The log directory above must already exist before sbatch submission.
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

cd /HWFS/zhaohu_lab/dingjiayi/code/TEEXope_3D_registration
# Budget one rendering CPU plus the compression threads for each process.
TIFF_THREADS=${TIFF_THREADS:-2}
if ! [[ "$TIFF_THREADS" =~ ^[1-9][0-9]*$ ]]; then
    echo "TIFF_THREADS must be a positive integer" >&2
    exit 1
fi
CPUS=${SLURM_CPUS_PER_TASK:-1}
WORKERS=$(( CPUS / (TIFF_THREADS + 1) ))
if (( WORKERS < 1 )); then WORKERS=1; TIFF_THREADS=1; fi
EXPORT_PATH=${EXPORT_PATH:-/HWFS/zhaohu_lab/dingjiayi/data/260823_RPL10a-mBaoJin_mScalet3H/261028_export_xyz_flexible}
echo "CPUs=$CPUS slice_processes=$WORKERS TIFF_threads/process=$TIFF_THREADS"
srun --ntasks=1 --cpus-per-task="$CPUS" python -u run_auto_xyz_export.py \
    --stage export --export-path "$EXPORT_PATH" \
    --export-workers "$WORKERS" --tiff-maxworkers "$TIFF_THREADS" "$@"
