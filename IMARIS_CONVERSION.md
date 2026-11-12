# Stitched TIFF series to Imaris

`Imaris.py` discovers image dimensions, dtype, Z depth, and identifiable channels
from TIFF headers and filenames. It streams data through parallel readers into
the native ImarisWriter, which builds the resolution pyramid, histograms,
thumbnail, compression, and 3D storage chunks. Pixel values are preserved.

## Install and run

Use a 64-bit Python 3.10, 3.11, or 3.12 environment. The requirements file selects
an older compatible NumPy/tifffile/imagecodecs stack for Python 3.10 and 3.11;
NumPy 2.5.4 requires Python 3.12 or newer. Use the environment's Python to run pip:

```sh
python -m pip install -r requirements-imaris.txt
python Imaris.py /data/stitched /data/volume.ims --dry-run
python Imaris.py /data/stitched /data/volume.ims --voxel-size 0.5 0.5 2
```

The voxel size is **X Y Z in micrometres**. Dimensions are automatic, but physical
spacing defaults to `1 1 1`; supply the acquisition spacing. TIFF resolution tags
often do not describe Z spacing and are not used to guess it. The saved timestamp
is conversion time, not acquisition time.

The PyImarisWriter native library and its dependencies must load on your system.
If installation succeeds but conversion cannot load a DLL/shared library, follow
the [vendor installation instructions](https://github.com/imaris/ImarisWriter).
Dry-run needs only NumPy and tifffile. Run one conversion per Python process;
the tested Windows vendor DLL hung when multiple conversions were created
sequentially in the same process.

## Linux native-library setup

`pip install PyImarisWriter` installs the Python wrapper but does not guarantee
that a Linux native binary is included. Our tested PyPI wheel contained Windows
DLLs. Linux needs **libbpImarisWriter96.so**, plus its native dependencies.
Adding a directory to executable `PATH` does not make its shared libraries load.

First locate any existing Linux build in the active environment:

```sh
find "$CONDA_PREFIX" -name 'libbpImarisWriter96.so*' -print
```

If found, use the directory containing `libbpImarisWriter96.so`:

```sh
export IMARIS_LIB_DIR=/absolute/directory/containing/the/library
export LD_LIBRARY_PATH="$IMARIS_LIB_DIR:$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
ldd "$IMARIS_LIB_DIR/libbpImarisWriter96.so"
python -c 'import ctypes; ctypes.CDLL("libbpImarisWriter96.so"); print("Native ImarisWriter loaded")'
```

Resolve any `not found` entries from `ldd` using the dependencies matching the
library build. Do not rename Windows DLLs or symlink incompatible HDF5 versions.
Set `LD_LIBRARY_PATH` before launching Python; changing it from within an already
running interpreter is not a reliable Linux loader fix. The sbatch file adds
`IMARIS_LIB_DIR`, the Python package directory, and the Python environment's `lib`
directory before starting the converter. Additional dependency directories can
be supplied in an exported `LD_LIBRARY_PATH`.

If the library is absent, obtain a compatible Linux build from your cluster
administrator or build the official source on the cluster. This example uses
Conda build dependencies in the active environment (requires network access):

```sh
conda install -c conda-forge 'cmake<4' make cxx-compiler 'hdf5=1.12.*' zlib lz4-c git
# Reactivate so the installed compiler activation scripts take effect.
conda activate PyImarisWriter
mkdir -p "$HOME/software"
cd "$HOME/software"
git clone https://github.com/imaris/ImarisWriter.git ImarisWriter-native
cmake -S ImarisWriter-native -B ImarisWriter-native/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$CONDA_PREFIX" \
  -DHDF5_ROOT:PATH="$CONDA_PREFIX" \
  -DZLIB_ROOT:PATH="$CONDA_PREFIX" \
  -DLZ4_ROOT:PATH="$CONDA_PREFIX"
cmake --build ImarisWriter-native/build --target bpImarisWriter96 --parallel 4
export IMARIS_LIB_DIR="$HOME/software/ImarisWriter-native/build"
export LD_LIBRARY_PATH="$IMARIS_LIB_DIR:$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
python -c 'import ctypes; ctypes.CDLL("libbpImarisWriter96.so"); print("Native ImarisWriter loaded")'
```

The example follows the [upstream build instructions](https://github.com/imaris/ImarisWriter#build)
and shared-library target; it has not been executed on your cluster. Follow your
cluster's policy for compilation on login nodes. If that checkout directory already
exists, use the existing checkout rather than cloning over it. Build and run with
the same native dependency environment. Confirm loading on a compute node as well
as the login node before a long conversion job. Run `sbatch` from the project
directory after building, with `IMARIS_LIB_DIR` still exported.

Library loading is checked before TIFF header scanning for actual conversion.
`--dry-run` intentionally does not require the native library and therefore does
not verify that conversion can load it.

## Slurm cluster submission

`run_imaris.sbatch` uses the existing cluster's `fat2` partition and code path.
It requests one node, 32 CPUs, 64 GiB RAM, and up to 30 days. By default it assigns
four reader threads, 27 native writer threads, and one CPU for the coordinator.
Activate the environment with the converter dependencies before submission:

```sh
cd /HWFS/zhaohu_lab/dingjiayi/code/TEEXope_3D_registration
sbatch run_imaris.sbatch /data/stitched /data/volume.ims --dry-run
sbatch run_imaris.sbatch /data/stitched /data/volume.ims --voxel-size 0.5 0.5 2
```

Replace the data paths and voxel spacing with your acquisition values. Logs are
`imaris_JOBID.out` and `imaris_JOBID.err` in the submission directory. Resource
overrides go before the script name; converter options go after input/output:

```sh
sbatch --cpus-per-task=16 --mem=128G --time=7-00:00:00 \
  run_imaris.sbatch /data/stitched /data/slab_0000_1000.ims --z-range 0 1000
```

Set `CODE_DIR` for a different checkout or `PYTHON_BIN` to an absolute environment
Python path. `READERS`, `READ_MEMORY_GIB` (default 4), and `COMPRESSION` (default
gzip2) can also be supplied as exported environment variables before `sbatch`.
The writer thread count is computed from the allocated CPUs; use `READERS` rather
than passing thread-count overrides to keep that budget consistent. Reserve enough
Slurm memory for the native writer in addition to reader buffers. The script does
not install packages during a job. Adjust the partition/account and wall-time limit
if required by your cluster policy.

## Input layouts

Files are naturally sorted (`z2.tif` before `z10.tif`); `.tif` and `.tiff` are
case insensitive. All headers are validated before writing.

* **One channel:** a directory of grayscale slices. Multipage spatial stacks are
  concatenated in filename/page order along Z.
* **Channel folders:** one immediate subdirectory per channel, each containing
  its naturally ordered Z series. Folder names become channel names. Every
  channel must have the same Z depth. Folder sequences are paired by position;
  verify their Z correspondence with your acquisition records.
* **Channel filenames:** labels such as `z0001_ch0.tif`, `z0001_ch1.tif`,
  `z0002_ch0.tif`, `z0002_ch1.tif`. Recognized tokens are `cN`, `chN`, and
  `channelN`, delimited by `_`, `-`, `.`, or filename boundaries. Z filename
  sequences must match after removing channel tokens.
* **RGB/sample channels:** interleaved or planar TIFF samples become individual
  IMS channels automatically. With channel folders/labels, each group contributes
  its sample channels.

For custom channel labels, use a regex matching just the channel token:

```sh
python Imaris.py /data/stitched /data/volume.ims --channel-regex 'laser(?P<c>\d+)'
```

Unlabeled grayscale TIFFs contain no inherent channel identity. To describe a
legacy sequence, use `--channels 2 --order channel-major` (all Z for C0, then C1)
or `--channels 2 --order z-major` (C0/C1 alternate for each Z). Without labels or
an override, files are treated as one channel. This assumption is logged.

Supported pixel types: uint8, uint16, uint32, float32. Shapes and types must agree.
Known T/C axes within a TIFF, multiple image series, volumetric tiles, palette or
YCbCr images, and nonstandard orientation are rejected instead of being silently
misinterpreted. Export those inputs as spatial TIFF series first. MINISWHITE
samples are preserved numerically; their display polarity is not inverted.

## Throughput and memory

```sh
python Imaris.py /data/stitched /data/volume.ims \
  --readers 4 --writer-threads 28 --read-memory-gib 4 \
  --compression gzip2 --block-xy 512 512 --voxel-size 0.5 0.5 2
```

Readers use independent TIFF handles and decode different row bands concurrently.
Native codecs release the GIL; decoding and disk reads overlap with native writer
compression. One coordinator submits contiguous, native-endian buffers to the
writer; concurrent calls to the same writer are avoided.

Uncompressed contiguous pages are memory mapped. Compressed images decode only
intersecting strips/tiles, one segment at a time per task. Row bands align with
segment boundaries when practical. Unusual mixed strip sizes may cause repeated
decode at band boundaries. A single huge compressed strip must still be decoded
whole; the planner rejects it when its estimated working set exceeds the budget.
Retile such inputs upstream for efficient region access.

`--read-memory-gib` budgets queued bands, codec scratch estimates, and the transfer
buffer. It is **not a hard process RSS limit**: native pyramid/compression buffers,
codec-specific allocations, file metadata, and OS caches are additional. Header
metadata scales with file/page count. Leave substantial RAM for the native writer
and measure peak RSS on representative dimensions before scaling up. The plan
reports its reader estimate and the effective prefetch/reader count.

Defaults use up to four readers and the remaining affinity-visible CPUs for the
writer. Scheduler CPU affinity is respected on Linux; explicitly set thread counts
when your job allocation uses a different CPU limiting mechanism. More readers can
hurt shared/HDD storage. Start with two to four, then measure. Threads do not imply
multi-node execution; a single IMS output remains constrained by storage bandwidth
and the native writer.

## Compression, chunking, and PB-scale operation

`gzip2` is a throughput-oriented starting point, not a universal optimum. Test
`gzip1`, `gzip2`, `gzip3`, and `none` on representative data. Higher gzip levels
can consume substantially more CPU for small size improvements. Compression is
lossless; ratio depends on image noise and content. No PB-scale speedup or
compression ratio is claimed without measurements on your hardware/data.

The `--block-xy` option controls input transfers, **not on-disk HDF5 chunk shape**.
The native library chooses 3D output chunks and pyramid levels. Its automatic Z
chunk selection remains enabled. This follows the
[ImarisWriter design](https://github.com/imaris/ImarisWriter) and the
[IMS format specification](https://imaris.oxinst.com/support/imaris-file-format).
The original script already requested eight native threads; its serial whole-plane
input path, per-block allocation, and forced garbage collection were bottlenecks
that this implementation removes.

Benchmark a representative range without copying TIFFs:

```sh
python Imaris.py /data/stitched /data/sample-gzip1.ims --z-range 0 128 --compression gzip1
python Imaris.py /data/stitched /data/sample-gzip2.ims --z-range 0 128 --compression gzip2
```

Compare total elapsed time, final file size, peak RAM, and interactive reading in
your Imaris version. Include finalization time, not just the input throughput log.
All source headers are still scanned; only the selected pixels are read.
`--z-range START STOP` is zero based, excludes STOP, and preserves the physical Z
offset in IMS extents. Use consecutive ranges with distinct output names to create
independent slabs for a PB-sized volume. These are separate IMS datasets, not an
automatically stitched virtual volume. Separate processes can convert slabs in
parallel, but their CPU, memory, and I/O budgets add together.

Output size includes the pyramid and padding, so do not reserve merely the base
voxel size. An ideal isotropic 2x pyramid adds about 14.3%, but actual overhead can
be larger with anisotropic dimensions, padding, and metadata. Compression savings
are not guaranteed. Check filesystem file-size limits, free space, and quotas.

Existing output files are never overwritten. A `.lock` prevents competing runs
targeting the same name. Work goes to a unique `.partial.ims` and is published
only after successful finalization and close. On POSIX, publication requires
same-filesystem hard-link support; on Windows it uses a non-overwriting rename.
Failed partial files are retained for diagnosis; they are not resumable. After a
hard kill, remove a stale lock only after confirming the process is gone. Retry
the affected slab using a fresh invocation.

## Tests

```sh
python -m pip install pytest h5py
python -m pytest test_imaris.py -q
```

Tests cover strip/tile decoding, samples, all supported dtypes, endianness,
multipage input, channel ordering, validation, bounded prefetch, failure cleanup,
and native IMS round trips checked through HDF5. Native tests require a working
PyImarisWriter installation and run each conversion in a separate process.
