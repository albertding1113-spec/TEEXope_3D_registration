# Stitch Nikon ND2 tiles on a cluster

`stitch_nd2.py` reads individual ND2 frames, places tiles using calibrated stage
coordinates, optionally refines XY translations from image overlaps, and blends
overlaps with normalized feather weights. Both channels use identical transforms.
It writes an **uncompressed uint16 BigTIFF for each channel and Z plane**. It does
not load a complete volume or a complete mosaic into a NumPy RAM array.

## Your acquisitions (metadata checked locally)

| File | Tiles | Tile pixels (Y × X) | Z slices | Channels | Voxel size X/Y/Z (µm) | Stage mosaic Y × X | Approximate TIFF size |
|---|---:|---|---:|---|---|---|---|
| 001.nd2 | 2400 | 512 × 512 | 103 | Green, Red | 0.614785 / 0.614785 / 0.6 | 27700 × 18484 | 196.5 GiB |
| 002.nd2 | 64 | 2048 × 2048 | 204 | Green, Red | 0.153696 / 0.153696 / 0.3 | 14951 × 14951 | 169.9 GiB |

Both acquisitions have approximately 10% tile overlap. The Nikon camera matrix
contains a small rotation; the script inverts this matrix to convert stage
displacements to camera-pixel offsets. It retains the camera orientation of the
images. It does not rotate all the pixels into stage coordinates.

**Stitch these files separately:** their sampling and tile coverage differ. This
workflow does not register the resulting low- and high-resolution volumes to
each other. It also does not use `multipoints.xml`; the embedded ND2 positions
are checked directly against the number of acquired tiles.

## Install on the cluster

Use Python 3.11 or newer. From the directory containing these scripts:

```bash
python -m venv .venv-nd2
source .venv-nd2/bin/activate
python -m pip install -r requirements-nd2.txt
```

Copy the ND2 files to cluster storage and substitute their real Linux paths below.
`F:\...` is a Windows path and will not work on the Linux cluster. Prepare plans
**after copying**: plans record the absolute input path, byte count, and modification
time. The local `*.local.json` plans are metadata checks, not cluster-ready plans.

Optional metadata inspection (no image pixels read):

```bash
python inspect_nd2.py /data/261031_Cdh5_Ci01Ci60/001.nd2 /data/261031_Cdh5_Ci01Ci60/002.nd2
```

## 1. Prepare tile placement once

Run preparation on a compute node. Here is a Slurm example; add your site's
partition/account options to `srun` and `sbatch` as needed:

```bash
srun --cpus-per-task=2 --mem=16G --time=04:00:00 \
  python stitch_nd2.py prepare /data/261031_Cdh5_Ci01Ci60/001.nd2 \
  --plan 001.layout.json --refine --channel 0 --reference-slices 5

srun --cpus-per-task=2 --mem=16G --time=04:00:00 \
  python stitch_nd2.py prepare /data/261031_Cdh5_Ci01Ci60/002.nd2 \
  --plan 002.layout.json --refine --channel 0 --reference-slices 5
```

Channel 0 is Green; channel 1 is Red. Preparation samples five evenly spaced Z
planes per tile, makes a maximum-intensity projection, and matches overlap
templates using normalized cross-correlation. Matches below 0.6 or at the search
boundary are rejected. A sparse global least-squares solve reconciles pairwise
shifts, with stage-position priors to constrain drift. The default search is
`min(tile_height, tile_width) // 20` full-resolution pixels (rounded down,
minimum 1): 25 pixels for 512 × 512 tiles, or 102 for 2048 × 2048 tiles.
Only translations are refined; there is no Z,
nonrigid, per-channel, or per-timepoint registration. The same placement is used
for the whole volume.

`--downsample 2` reduces reference memory/computation, at the cost of registration
precision. For the 512-pixel tiles, keep the default 1 because overlaps are narrow.
Increasing `--reference-slices` can help when only some depths contain signal.
`--max-shift` must remain substantially smaller than the overlap width.

`--max-shift` limits **pairwise** corrections, not the accumulated global tile
correction. Small consistent neighbor corrections can add up across a large grid.
The default global limit is `min(tile_height, tile_width) // 10` (rounded down,
minimum 1): 51 pixels for 512 × 512 tiles, or 204 for 2048 × 2048 tiles.
Use `--max-global-shift PIXELS` to override this independent guard on the
largest absolute X or Y correction component. Explicit `--max-shift PIXELS`
also overrides its size-derived default. The resolved limits are printed before
refinement and recorded in the registration diagnostics.
Do not increase the pairwise search radius just to accommodate accumulated drift:
it also shrinks the overlap template and can prevent matching narrow overlaps.

Preparation prints the largest global correction and 95th-percentile overlap
residual. It saves `001.layout.registration.json` beside `001.layout.json` (using
your plan's filename), including per-tile corrections and edge residuals, even
when the optional global guard fails. This diagnostics sidecar is replaced on a
retry; completed placement plans remain protected against overwriting. A large
global correction alone does not prove a bad match, and a successful solve alone
does not prove correct registration: inspect diagnostics and pilot seams.

Review the plan's `registration` section: accepted overlaps, unmatched tiles,
per-edge NCC scores, and residuals. Unmatched tiles retain their stage positions.
Large residuals, many unmatched tiles, or repetitive structures call for visual
inspection and potentially a different reference channel. The algorithm does not
guarantee correct matches in blank/repetitive tissue. If no matches pass, preparation
fails. To use stage placement alone, omit `--refine` and choose a new plan filename.
Plans are never overwritten.

## 2. Render a pilot plane

```bash
srun --cpus-per-task=2 --mem=16G --time=02:00:00 \
  python stitch_nd2.py stitch --plan 001.layout.json --output /data/stitched/001 \
  --scratch /path/to/node-local-scratch --z-range 51 52 --workers 1
```

`--z-range START STOP` uses zero-based indices with STOP excluded. Inspect both
channel images and several tile seams in Fiji/ImageJ before rendering everything.
Use a fresh output directory if you change the placement plan. You can also compare
stage-only and refined pilot planes in separate output directories.

## 3. Render all planes with a Slurm job array

From the script directory, with the Python environment activated:

```bash
sbatch run_nd2_stitch.sbatch "$PWD/001.layout.json" /data/stitched/001
sbatch run_nd2_stitch.sbatch "$PWD/002.layout.json" /data/stitched/002
```

The supplied script requests eight array tasks, up to four running simultaneously,
with two worker processes per task and 32 GiB RAM per task. Each task handles a
disjoint subset of Z planes. This is multi-node parallelism through Slurm, without
MPI or a Dask scheduler. Each process owns its own ND2 reader. Both channels are
read from a tile frame once and blended together.

For a slower shared filesystem, start conservatively:

```bash
WORKERS=1 sbatch --array=0-3%2 --cpus-per-task=2 --mem=16G \
  run_nd2_stitch.sbatch "$PWD/001.layout.json" /data/stitched/001
```

Array indices must be contiguous and start at zero. `CODE_DIR` defaults to the
submission directory; set it if you submit from elsewhere. `PYTHON_BIN` can point
to a specific environment's Python. Scheduler partition/account settings are
site-specific and intentionally omitted from the supplied script.

The scratch directory defaults to `$SLURM_TMPDIR`, then `$TMPDIR`, then `/tmp`.
Use node-local SSD where available. The float accumulation buffers require about
**5.73 GiB per worker for 001**, or **2.50 GiB per worker for 002**. Leave additional
space for filesystem overhead. Output TIFFs and temporary TIFFs are written on
the output filesystem. Budget at least **400 GiB free output space** for the two
complete series, plus input storage and scratch. Refinement may change dimensions
slightly; preparation prints updated estimates.

Memory mapping avoids giant heap arrays, but mapped pages still consume OS cache
and may count toward cluster memory limits. Increase RAM or reduce workers if
your scheduler reports an OOM. More concurrent readers are not always faster:
the ND2 acquisition is position-major, while this renderer accesses planes across
positions. Large shared-storage seek workloads can be I/O-bound.

## Output, metadata, and restart

```text
001/
  placement.json
  ch00_z00000.tif ... ch00_z00102.tif   # Green
  ch01_z00000.tif ... ch01_z00102.tif   # Red
  z00000.done.json ...
```

Both channels are written directly into the output directory for these single-timepoint
acquisitions. Multi-timepoint acquisitions use a separate `t0000/`, `t0001/`, etc.
directory, each containing the same `chCC_zZZZZZ.tif` naming pattern. When importing
a channel as a stack, select only the matching `ch00_` or `ch01_` files.

Each TIFF is a 2D grayscale BigTIFF. TIFF resolution tags encode XY pixel size;
JSON image descriptions and `placement.json` record voxel size, channel, indices,
and physical Z. These are ordinary TIFF series, **not an OME-TIFF volume**.
When importing a folder as a stack, set Z spacing explicitly from the table above
if your viewer does not interpret the JSON description. Intensities retain the
source uint16 storage range; overlap values are weighted averages rounded to
integers. There is no contrast normalization, gain matching, or flat-field
correction. Uncovered pixels are zero. Fractional XY offsets are bilinearly
deposited, which introduces interpolation rather than nearest-pixel placement.

`--resume` skips completed planes only when the plan hash and recorded file sizes
match; it rebuilds incomplete planes. The Slurm script enables it automatically.
This is not a pixel checksum: same-size file corruption is not detected.
The output's saved plan must exactly match the requested plan. Partial TIFFs
are renamed only after completion; each two-channel plane gets a completion marker
after all channels finish. A `.lock` file prevents concurrent writes to the same
plane. After a hard kill, check that the previous job is no longer running before
removing its stale `zNNNNN.lock` and resubmitting. Failed tasks may leave scratch
buffers or `.partial.tif` files; remove those only after the job has stopped.

The code rejects unsupported loop dimensions, RGB/legacy ND2 files, missing
calibration, duplicate tile positions, incomplete frame counts, and tile-dependent
Z start/end grids. It assumes regular Z sampling between the checked endpoints.
Acquisitions with different Z offsets require a separate 3D resampling workflow.

## Validation performed

Both real files' metadata and tile Z endpoints were checked. Two adjacent tiles
at a middle Z slice were read and blended from each file. On the tested overlap
in 002, Green-channel matching estimated approximately (-5, -5) pixels relative
to rounded stage placement (NCC about 0.61). The first tested overlap in 001 did
not yield an accepted match. This is a small smoke test, not whole-volume seam QA.
The complete datasets have not been rendered locally.

Synthetic tests exercise camera transforms, horizontal/vertical registration
signs, blank-overlap rejection, fractional blending, channel consistency, frame
loop mapping, BigTIFF output, and recovery from partial outputs:

```bash
python -m pip install pytest
python -m pytest test_stitch_nd2.py -q
```

Reader API reference: [nd2 documentation](https://talleylambert.com/nd2/reference/nd2/).
