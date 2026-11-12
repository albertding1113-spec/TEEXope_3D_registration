#!/bin/bash
#SBATCH --job-name=tiff_to_imaris
#SBATCH --partition=fat2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=30-00:00:00
#SBATCH --output=imaris_%j.out
#SBATCH --error=imaris_%j.err

set -euo pipefail

# Usage (activate your Python environment before submission):
# sbatch run_imaris.sbatch /data/stitched /data/volume.ims --voxel-size 0.5 0.5 2
# sbatch run_imaris.sbatch /data/stitched /data/volume.ims --dry-run
# Logs are created in the submission directory, which must be writable.
# Override resource directives BEFORE the script name, e.g.:
# sbatch --cpus-per-task=16 --mem=128G run_imaris.sbatch INPUT OUTPUT [options]

if (( $# < 2 )); then
    echo "Usage: sbatch run_imaris.sbatch INPUT_DIRECTORY OUTPUT.ims [Imaris.py options]" >&2
    exit 2
fi

INPUT_DIR=$1
OUTPUT_FILE=$2
shift 2

# Slurm copies this script to its spool directory; do not locate Imaris.py via $0.
# Set CODE_DIR if your cluster checkout is elsewhere. Relative input/output paths
# are resolved from the submission directory, not CODE_DIR.
CODE_DIR=${CODE_DIR:-/HWFS/zhaohu_lab/dingjiayi/code/TEEXope_3D_registration}
PYTHON_BIN=${PYTHON_BIN:-python}
# Alternatively, activate your conda/venv environment here.
# source /path/to/miniconda3/etc/profile.d/conda.sh
# conda activate imaris

CPUS=${SLURM_CPUS_PER_TASK:-32}
if ! [[ "$CPUS" =~ ^[1-9][0-9]*$ ]] || (( CPUS < 3 )); then
    echo "Request at least 3 CPUs: one reader, one writer, and one coordinator." >&2
    exit 2
fi
DEFAULT_READERS=$(( CPUS / 4 ))
if (( DEFAULT_READERS < 1 )); then DEFAULT_READERS=1; fi
if (( DEFAULT_READERS > 4 )); then DEFAULT_READERS=4; fi
READERS=${READERS:-$DEFAULT_READERS}
if ! [[ "$READERS" =~ ^[1-9][0-9]*$ ]] || (( READERS > CPUS - 2 )); then
    echo "READERS must be a positive integer no greater than CPUS - 2." >&2
    exit 2
fi
WRITER_THREADS=$(( CPUS - READERS - 1 ))
READ_MEMORY_GIB=${READ_MEMORY_GIB:-4}
COMPRESSION=${COMPRESSION:-gzip2}

# Avoid unrelated numerical libraries starting extra CPU pools.
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export PYTHONUNBUFFERED=1

if [[ ! -f "$CODE_DIR/Imaris.py" ]]; then
    echo "Cannot find $CODE_DIR/Imaris.py; set CODE_DIR to your cluster checkout." >&2
    exit 2
fi
if [[ ! -d "$INPUT_DIR" ]]; then
    echo "Input directory does not exist: $INPUT_DIR" >&2
    exit 2
fi
command -v "$PYTHON_BIN" >/dev/null || {
    echo "Cannot find Python: $PYTHON_BIN; activate your environment or set PYTHON_BIN." >&2
    exit 2
}

# Linux's shared-library loader uses LD_LIBRARY_PATH, not executable PATH.
# Set this in the parent shell BEFORE srun starts the converter interpreter.
WRITER_PACKAGE_DIR=$("$PYTHON_BIN" -c 'import importlib.util; from pathlib import Path; s = importlib.util.find_spec("PyImarisWriter"); print(Path(s.origin).parent if s and s.origin else "")')
PYTHON_PREFIX=$("$PYTHON_BIN" -c 'import sys; print(sys.prefix)')
if [[ -n "${IMARIS_LIB_DIR:-}" ]]; then
    if [[ ! -f "$IMARIS_LIB_DIR/libbpImarisWriter96.so" ]]; then
        echo "Missing $IMARIS_LIB_DIR/libbpImarisWriter96.so" >&2
        exit 2
    fi
    export LD_LIBRARY_PATH="$IMARIS_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
if [[ -n "$WRITER_PACKAGE_DIR" ]]; then
    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}$WRITER_PACKAGE_DIR"
fi
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}$PYTHON_PREFIX/lib"

echo "Job=${SLURM_JOB_ID:-local} Host=$(hostname) Started=$(date -Is)"
echo "CPUs=$CPUS Readers=$READERS Writer_threads=$WRITER_THREADS Coordinator=1"
echo "Reader_budget=${READ_MEMORY_GIB}GiB (native writer memory is additional)"
echo "Input=$INPUT_DIR Output=$OUTPUT_FILE Compression=$COMPRESSION"
"$PYTHON_BIN" --version

# Extra converter options include --voxel-size, --z-range, --channels, --order,
# --channel-regex, --block-xy, and --dry-run. Use READERS to change reader counts
# while keeping the computed CPU allocation consistent.
srun --ntasks=1 --cpus-per-task="$CPUS" \
    "$PYTHON_BIN" -u "$CODE_DIR/Imaris.py" "$INPUT_DIR" "$OUTPUT_FILE" \
    --readers "$READERS" --writer-threads "$WRITER_THREADS" \
    --read-memory-gib "$READ_MEMORY_GIB" --compression "$COMPRESSION" "$@"

echo "Completed=$(date -Is)"
