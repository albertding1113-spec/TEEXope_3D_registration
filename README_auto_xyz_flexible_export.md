# Automatic rigid XYZ + flexible XY PB-scale TIFF exporter

## What changed

This is a revised version of `flexible_export_pb.py` using **no external `shift_array_path`**.

1. Read raw tiles from `X..._Y..._Z.../0/ch00.ome.tif` and group them by nominal Z layer.
2. **Rigid XYZ:** For each adjacent pair, assemble spatially bounded, downsampled 3D volumes of selected XY regions using the registration channel. Estimate the 3D translation using masked 3D phase correlation (`scikit-image`). Filter candidates using correlation, valid overlap, and maximum XYZ shift constraints. Optionally refine Z displacement by testing neighboring Z offsets at single-raw-voxel precision. Apply each translation to the moving layer's stage coordinates.
3. **Flexible XY:** Using corrected Z coordinates, read bounded XY patches in matching overlap planes, measure local residual shifts through the preceding field, interpolate/smooth the residual and compose it with the preceding field. Whole-layer XY mosaics are no longer needed.
4. **Export:** Stitch overlapping tiles and layers with validity-aware blending, apply the fields to *all* channels, and stream tiled BigTIFF files without building full-resolution XY output planes in RAM.

The first layer is held fixed as the reference. **Rigid offsets are absolute corrections to the nominal stage coordinates**, measured in original XYZ voxels. They may differ substantially between neighboring layers. The flexible fields contain output-to-input sampling displacements in original XY pixels and are invariant to export downsampling.

**Scope:** automatic XYZ *translation* registration, then spatially varying XY deformation. This does NOT estimate nonrigid 3D shifts `dz(x,y,z)`, rotation, changing scale, local tissue shearing in Z, or elastic/biomechanical 3D alignment. Such cases need more sophisticated 3D nonrigid registration.

**The rigid transform is one XYZ translation per Z layer. Individual XY tiles within a layer retain their stage-relative positions. There is no pairwise XY tile registration or global tile graph optimization.** XY stage errors within a layer cannot be independently corrected by this implementation. Flexible fields are constant through the depth of each layer.

Folder coordinates are interpreted as physical micrometers: with `flip_stage_xy=True`, `X150700.000_Y39097.500_Z03676.904` maps to `(-150700.000, -39097.500, 3676.904)`. Positions are divided by XYZ voxel size, shifted by the dataset minimum, and rounded to raw voxels. No image-axis flip is performed. All ROI ranges refer to normalized, rigid-corrected raw XYZ voxels, not absolute stage micrometers. `normalized_origin_stage_xyz_um` records the physical origin; nominal stage coordinates are quantized to the nearest raw voxel.

## Installation

Python 3.10+ recommended:

```bash
pip install numpy scipy scikit-image opencv-python-headless tifffile imagecodecs
```

With a conda environment, activate it on the Slurm compute node. `imagecodecs` is needed for LZW-compressed output; set `compression=None` if it is unavailable.

## Use

1. Open `run_auto_xyz_export.py`; edit `dir_path`, `export_path`, `registration_dir`, channel and imaging parameters.
2. Run `python -u run_auto_xyz_export.py --stage rigid`. This runs only rigid registration and writes `registration_dir/rigid_offsets.csv`. The launcher defaults to this stage.
3. Review rigid QC and manually edit the CSV's `dx_raw`, `dy_raw`, `dz_raw` values as needed. Keep layer 0 at zero. See the example below.
4. Run `python -u run_auto_xyz_export.py --stage flexible`. It loads the finished rigid checkpoint and edited CSV, then computes only flexible registration.
5. Inspect flexible QC, choose export resolution/ROI and run `python -u run_auto_xyz_export.py --stage export`. It loads both completed registrations; it never runs either registration algorithm.
6. Change `export_path` for more exports; keep `registration_dir` unchanged. Different channel exports still share the field as long as the **registration channel** is unchanged. **Registration covers every Z layer**, even if `if_subregion=True` (which affects output only).

Slurm accepts the same options, for example `sbatch run_auto_xyz_export.sbatch --stage flexible`. In Python, set `workflow_stage="rigid"`, `"flexible"`, `"export"`, or `"preview"`, or call `run_rigid(c)`, `run_flexible(c)`, `run_export(c)`. Explicit stages take precedence over the legacy `registration_only` setting. `workflow_stage="all"` retains the original automatic pipeline; only in this mode does `registration_only=True` stop after both registration stages.

### Manual rigid corrections from Fiji

Edit **`registration_dir/rigid_offsets.csv`** with a text/CSV editor after the rigid stage completes. It is the authoritative editable checkpoint for subsequent stages; the automatic binary `rigid_state.npz` and rigid QC remain a record of the automatic result. Do not edit `rigid_offsets_raw_xyz.npy`, which is a derived copy refreshed by registration stages.

Example: a skipped layer 1 with nominal Z start 539 can be corrected like this (retain the actual `nominal_z_raw` from your file):

```csv
layer,nominal_z_raw,dx_raw,dy_raw,dz_raw,status,note
0,0,0,0,0,reference,
1,539,100,-100,10,manual,Measured in Fiji
```

Keep all rows and the `layer`/`nominal_z_raw` columns unchanged. Only the numeric XYZ offsets control registration; `status` and `note` are annotations. Offsets must be finite, and layer 0 must stay `[0,0,0]`. Manual corrections are not capped by automatic search limits. Positive X/Y shifts move the layer toward increasing stitched X/Y; positive Z places it later in the stack. In the example, the corrected layer-1 origin is nominal XYZ plus `[100,-100,10]`, including a corrected Z start of 549 before final dataset normalization.

**Values are absolute corrections relative to each layer's nominal stage origin, in full-resolution XYZ voxels.** They are not physical micrometers, downsampled pixels, total layer separation, or pairwise correction increments. For layer 1, a correction relative to fixed layer 0 is entered directly. If layer 1 has correction `[100,-100,10]` and the correction increment from layer 1 to layer 2 is `[20,5,-2]`, enter layer 2 as `[120,-95,8]`. Existing CSV values already follow the absolute convention; do not cumulatively sum them again.

Changing rigid offsets changes the flexible-registration signature. Export refuses old/stale fields. To recompute flexible registration after another edit, run:

```bash
python -u run_auto_xyz_export.py --stage flexible --force-registration
python -u run_auto_xyz_export.py --stage export
```

Forcing the **flexible** stage preserves all rigid edits. Forcing the **rigid** stage recomputes automatic offsets and replaces the editable CSV, saving the previous CSV as `rigid_offsets.csv.bak`. A normal rigid rerun preserves an existing CSV. Use a new export directory after changing registration, because old TIFF output has a different signature.

Missing/incomplete rigid checkpoints block the flexible stage; missing/incomplete/stale flexible checkpoints block export when flexible stitching is enabled. Neither stage silently reruns an earlier algorithm. With `rigid_stitch=False`, later stages explicitly use zero rigid offsets; with `flexible_stitch=False`, export explicitly uses zero flexible fields. Keep `rigid_stitch=True` to apply manual rigid edits.

### Registration cache and resume

Progress is printed immediately with timestamps and `flush=True`, including when stdout is redirected to a Slurm log. Rigid progress includes layer/ROI numbers, ROI volume loading, XYZ corrections, NCC/overlap acceptance, Z-refinement trials, checkpoint paths and elapsed time. Flexible progress includes overlap/sample planes, periodic patch counts, rejection totals, residual shifts, cumulative field ranges and checkpoints. Export prints each output path with image number/total, Z/channel/shape/dtype, tile-row progress, skip/rebuild decisions and final file size/time. Cached registration is explicitly reported as reused. Patch and tile-row updates are periodic to limit log volume.

Default `registration_dir`: `<dir_path>/_registration_xyz_flexible`. You can explicitly set it to any shared cluster directory.

```text
registration_dir/
    rigid_state.npz                 # exact signature, offsets, completed layer, QC rows; saved EACH completed layer
    rigid_registration_qc.csv       # XYZ offsets, accepted ROI count, NCC, status
    rigid_offsets.csv               # EDIT HERE: absolute per-layer XYZ corrections, nominal Z and notes
    overlap_previews/               # diagnostic TIFF overlap crops, masks, overlays and metadata
    flexible_state.npz              # exact signature, fields, completed layer, QC rows; saved EACH completed layer
    flexible_registration_qc.csv    # local-match statistics and status
    rigid_offsets_raw_xyz.npy      # convenient completed offsets array of shape [L,3]
    flexible_displacement_fields.npz # completed field, XYZ/index metadata
    registration_manifest.json      # settings and coordinate/array conventions
    tile_index.npz                  # folder names, corrected XYZ origins, layer IDs, physical origin
```

A `*.partial` file is never treated as a valid checkpoint. Completed layers are resumed from the checkpoint after job interruption. Set `force_registration=False` to reuse. **`force_registration=True` recomputes only the selected registration stage; with `workflow_stage="all"`, it recomputes both.** Export never recomputes registration.

The revised algorithm uses cache version v2. Old v1 checkpoints deliberately fail signature validation. Use a new registration directory for this version, or deliberately recompute. Resume also checks output TIFF headers, shapes, dtypes and tile payload extents; truncated files are rebuilt. This is structural validation, not a pixel checksum.

The cache stores a strict signature based on paths, acquisition-layer directory names, geometry, rigid settings, flexible settings, corrected offsets, and field bounds. A mismatch raises an error: choose another `registration_dir` or set `force_registration=True` for a deliberate recomputation.

**Caution:** Fingerprints check directory names and registration parameters, *not a full checksum of petabytes of TIFF data*. If the TIFF contents are replaced in-place, explicitly recompute registration. If an export folder already contains TIFFs, a changed export configuration is also detected and `resume=True` fails instead of silently mixing results; use a new export folder or set `resume=False`.

### Main parameters

| Config | Default | Meaning / unit |
|---|---:|---|
| `registration_channel` | 0 | Channel with well-distributed features in both layers |
| `rigid_stitch` | True | Enable automatic XYZ translation |
| `rigid_xy_downsample` | 16 | Lateral thumbnail factor, raw XY pixels |
| `rigid_z_downsample` | 8 | Axial sampling factor, raw Z voxels |
| `rigid_roi_px` | 4096 | ROI width and height in **raw XY pixels** |
| `rigid_roi_count` | 3 | Maximum number of signal-ranked full 3D ROIs for each layer pair |
| `rigid_signal_candidates` | 64 | Maximum candidate regions previewed for signal in both layers |
| `rigid_signal_z_samples` | 5 | Preview planes per candidate per layer, near possible Z overlap |
| `rigid_skip_on_failure` | True | Keep nominal stage positions if no trustworthy rigid shift is found |
| `rigid_max_xy_shift_px` | 512 | Maximum absolute correction in X and Y (each axis), raw voxels |
| `rigid_max_z_shift_px` | 256 | Maximum absolute correction in Z, raw voxels |
| `rigid_min_ncc` | 0.10 | Minimum 3D alignment normalized correlation |
| `rigid_min_overlap` | 0.12 | Minimum common 3D valid coverage |
| `rigid_refine_z` | True | Check ±`rigid_z_downsample` Z voxels around coarse estimate at 1-voxel resolution |
| `rigid_refine_z_planes` | 5 | Z slices to sample when refining |
| `rigid_max_volume_megavoxels` | 8 | Maximum voxels per 3D ROI after downsampling, in millions |
| `registration_downsample` | 16 | XY factor used for fine deformation estimation |
| `registration_mesh_px` | 4096 | Field control-grid spacing in **raw XY pixels** |
| `registration_patch_px` | 8192 | Local XY phase-correlation window in **raw XY pixels** |
| `registration_z_samples` | 1 | Matching planes per adjacent-layer overlap |
| `max_local_shift_px` | 256 | Maximum residual local XY shift magnitude, raw pixels |
| `field_smoothing_sigma` | 0.8 | Gaussian field smoothing in **control-grid nodes** |
| `max_registration_megapixels` | 50 | Pixel cap on each padded flexible registration patch, MP |
| `cache_open_files` | 64 | Maximum simultaneously open source TIFF handles |
| `strict_input` | True | Stop on missing, malformed or unreadable requested input pages |
| `opencv_threads` | 1 | OpenCV workers; bounds nested thread pools |
| `workflow_stage` | all (launcher: rigid) | Run rigid, flexible, export, preview, or the original automatic pipeline |
| `save_registration_previews` | True | Save diagnostic registration overlap regions |
| `registration_preview_flexible_patches` | 3 | Maximum accepted flexible patch comparisons per sampled Z plane; 0 disables |
| `registration_only` | False | With stage all, calculate/save both registrations and skip output |
| `force_registration` | False | Recompute the selected registration stage; stage all recomputes both |
| `resume` | True | Skip existing validated TIFF output files |
| `stop_on_registration_failure` | True | Stop on flexible failures; also stop on rigid failures when `rigid_skip_on_failure=False` |

With `rigid_z_downsample=8`, the 3D coarse estimate has approximately 8-voxel Z quantization. Enabling `rigid_refine_z` can improve it to approximately a 1-voxel step, but *does not guarantee* accurate axial alignment, particularly if structures look very similar in neighboring Z planes.

### Memory and performance

Registration uses **3D ROI volumes and streamed XY patches**, not whole-brain arrays. The source page cache is bounded by `cache_mb`; TIFF handles/page directories are separately bounded by `cache_open_files`. Flexible patch padding covers the preceding cumulative field. The MP limit applies to the padded patch, and several images, masks, coordinate maps and FFT buffers coexist. Neither pixel-count limit is a total process RAM limit. The directory index, per-layer KD trees and all control fields remain in RAM.

TIFF page reads reuse file handles, with one decoder worker per page. This avoids repeatedly opening TIFFs and reparsing page directories. Cache entries contain XY-downsampled pages; whole source pages are still decoded, even for small output ROIs. Source page memory, TIFF directory metadata and compression buffers are additional to `cache_mb`. Try larger `tiff_tile` values such as 512 or 1024 when repeated source-page decoding dominates, and measure throughput on the actual filesystem before budgeting a PB export. Current execution is a single process; Slurm CPU requests do not distribute registration or export across nodes.

3D phase correlation also requires intermediate arrays substantially larger than its input volumes; size the Slurm memory appropriately. 3D rigid registration can be I/O-intensive because many Z planes must be read; adjust `rigid_roi_count`, `rigid_z_downsample`, and `rigid_roi_px` if necessary. A GPU is not required in this implementation.

### Quality control and limitations

- The relative Z positions encoded in the folder names must provide at least some **actual tissue overlap**; if the true Z overlap is missing, the registration algorithm cannot invent corresponding features. For a 640-plane stack and nominal layer gap of 350 planes, an error greater than the nominal overlap may lead to failure.
- Good XYZ registration requires genuine *3D* texture; constant-intensity features, empty channels or nearly uniform Z projections may be ambiguous. Review QC, particularly `accepted_rois` and `median_ncc`, and visualize crossing neurites/vessels at seams.
- Local phase correlation can give false matches for repeated structures. Coarse alignment is limited by downsampling, stage-coordinate accuracy and usable overlap. Unsuccessful rigid matches now retain nominal stage positions and are explicitly marked in QC; this fallback does not establish successful alignment.
- Rigid registration is **not** an independent rotation/scale search. If your sample experiences nonlinear changes in Z, use a 3D deformation field as a future extension.
- A subregion only affects output, not the registration domain or cache, so one registration can support whole-brain and cropped export.
- `downsample_factor=(5,5,100)` uses XY area downsampling and **samples** every hundredth Z plane; it does not average Z planes.
- Source and output axes are TIFF `Z,Y,X` and page `Y,X`; configuration, stage and offsets use `X,Y,Z`.
- Source TIFFs must contain exactly `stack_shape[2]` separate YX pages of the configured XY shape. Multi-file OME series, time axes and pyramidal level selection are not supported by this reader.
- XY intensity downsampling averages fixed-size raw blocks, including smaller final blocks for non-divisible dimensions. Z downsampling remains sampling. Shifted tile grids use interpolated XY averages, not an exact integration of the final deformed image.
- `if_xy_blend=False` selects the tile with the greatest edge clearance per pixel. `if_z_blend=False` selects the deepest available layer per pixel. Sole-coverage areas fall back to the available tile/layer. XY intensities are normalized within each layer before Z blending.
- Full-image export bounds include a conservative flexible-displacement margin; ROI export still clips to the requested bounds. Fluorescence output is clipped to uint16. TIFFs are individual tiled 2D BigTIFF planes, with XY physical resolution tags; the complete XYZ geometry is in `export_metadata.json`, not an OME volume.
- Label images are sampled using nearest-neighbor and can retain `uint32` instance IDs by setting `segmentation_mode='preserve_labels'`.

### Inspecting the overlap regions

With `save_registration_previews=True`, new rigid calculations save the selected ROI overlap regions under:

```text
registration_dir/overlap_previews/
    rigid/layer_0000_to_0001/attempt_.../
        selection.json
        roi_001/before/
        roi_001/candidate/
        final/
    flexible/layer_0000_to_0001/attempt_.../z_.../
        patch_.../before_residual/
        patch_.../candidate_aligned/
    checkpoint_review/attempt_.../layer_0000_to_0001/
        roi_001/nominal_moving/
        roi_001/corrected/
```

Each image folder contains:

- `reference.tif` and `moving.tif`: float32 registration thumbnails without display contrast scaling. Rigid previews are ZYX stacks over the actual nominal/proposed common Z interval, sampled on matching global Z planes; flexible previews are 2D YX patches.
- `reference_valid.tif` and `moving_valid.tif`: uint8 source-coverage masks. Zero intensity alone does not distinguish absent coverage from real dark tissue.
- `overlay_xy.png`: red=reference, green=moving, yellow=coincident projected brightness. Rigid folders also include `overlay_xz.png` and `overlay_yz.png`.
- `metadata.json`: layer IDs, ROI position, channel, downsampling, overlap bounds, exact sampled global Z values/local source Z indices for rigid crops, proposed correction, match status and applicable QC values.

Open both TIFF stacks in Fiji, merge them as channels and step through corresponding slices. PNGs use a shared percentile contrast range for each view and maximum-intensity projections for 3D stacks. Yellow in a projection does not prove 3D correspondence; inspect the TIFF planes and masks. XZ/YZ PNGs use thumbnail-pixel aspect ratios; use the recorded sampling intervals to interpret physical distances.

Rigid previews include rejected candidates, with acceptance status in metadata. `before` shows reference at its current correction and moving at nominal coordinates; `candidate` shows the proposed ROI correction; `final` shows the combined/refined automatic result at the best-scoring ROI. A failed signal scout saves geometric fallback crops, marked `scout_failed`; no Z overlap produces metadata instead of an empty TIFF. Attempts have separate directories so earlier images are not mistaken for the latest result.

Flexible previews show both layers sampled through the preceding cumulative field, before and after the candidate residual translation. They are local match diagnostics, not the final interpolated/smoothed field. Up to `registration_preview_flexible_patches` accepted patches are saved per sampled plane; if there are no accepted patches, one failed patch is saved for inspection. No full-layer image is assembled for diagnostics.

For **already completed registration or manually edited rigid offsets**, run:

```bash
python -u run_auto_xyz_export.py --stage preview
```

This reads `rigid_offsets.csv` and saves new `checkpoint_review` crops comparing nominal and corrected moving positions against the corrected reference. It reuses the latest saved ROI centers when available, otherwise scouts regions. It reads source images but does not rerun rigid/flexible registration, overwrite edits or export the mosaic. Keep `save_registration_previews=True`. Preview settings do not invalidate registration caches. Cached registration does not backfill automatic-stage diagnostics; use this preview stage for rigid review. Flexible diagnostics are produced when the flexible algorithm actually runs.

Diagnostics add TIFF reads/writes and disk usage. Rigid crops use the configured rigid XY/Z downsampling and volume cap, and flexible crops use the registration thumbnail resolution. Source-tile placement in these thumbnails follows the same rounded grid placement used by registration; full-resolution export uses the export sampler. Set `save_registration_previews=False` to disable diagnostics.

### Sparse signal and skipped rigid pairs

Rigid ROI selection first scouts downsampled previews around actual tile centers and geometric anchors. It ranks candidates using high-pass morphology present in **both** layers, with a preference for spatially separated regions. Only the selected regions are read as complete 3D volumes. Previews are no wider than one source tile, and their Z samples cover the possible overlap band expanded by the rigid Z correction limit.

```python
rigid_signal_candidates=64,
rigid_signal_z_samples=5,
rigid_roi_count=3,
rigid_skip_on_failure=True,
```

Scouting is bounded: when there are more candidates than the budget, tile centers are sampled reproducibly. Very rare signal or signal confined to a few Z planes may be missed; increase the candidate/plane budgets in that case. Scouting adds source-page reads but avoids spending full-volume correlation on clearly empty ROIs. Signal ranking does not relax NCC, overlap or shift acceptance limits.

If no signal-bearing candidate or trustworthy rigid match exists, `rigid_skip_on_failure=True` keeps the moving layer at its **nominal absolute stage position**, with zero XYZ correction. It does not inherit the previous layer's rigid correction. The warning, reason and `skipped_nominal` status are saved in rigid QC/checkpoints, and later pairs are still processed. Input I/O errors, memory/configuration failures and interruptions are not treated as missing alignments.

Flexible registration still runs afterward and follows `stop_on_registration_failure`. With that option `False`, missing flexible matches carry the preceding XY field, as before. To stop on an unsuccessful rigid match, set `rigid_skip_on_failure=False` and `stop_on_registration_failure=True`.

The scout/fallback parameters are part of the registration signature. Checkpoints from before the scouting change require a new registration directory or deliberate recomputation of both registration stages. Adding workflow-stage selection itself does not change the registration signature; existing compatible completed rigid states can generate the editable CSV without recomputing alignment.

### Illumination standardization during export

Set these options in `run_auto_xyz_export.py`:

```python
if_standardization=True,
stand_img_path="/path/to/average_illumination.tif",
```

The TIFF must contain **one grayscale YX slice**, with the same shape and detector orientation as a raw tile page. For `stack_shape=(2048,2048,640)`, the profile must be 2048 rows by 2048 columns. Do not transpose a TIFF profile. The slice represents the average device illumination/signal profile, rather than a precomputed inverse gain map.

Each signal page is corrected using `corrected = raw * mean(profile) / profile` at native source pixels, before XY downsampling, flexible warping and blending. The same device profile is used for every raw signal channel and every Z page. Correction stays float32 until the existing uint16 export conversion. Segmentation labels are never intensity-corrected. The profile is loaded once per export run; corrected/downsampled pages use the existing bounded page cache. Original source TIFFs are not modified.

The profile must have finite, nonnegative values and a positive mean. Zero or extremely dark profile pixels are limited to a maximum correction gain of 100, with a warning. No dark-field subtraction, profile estimation or smoothing is performed. Legacy `.npy` profiles remain supported in their original XY convention; new TIFF profiles use normal YX.

This is an **export-only** option, so existing rigid and flexible registrations can be reused. The rigid/flexible stages skip loading/applying the profile. Use `workflow_stage="export"` for corrected output; in legacy stage all, `registration_only=True` skips export. Use a new export directory when enabling or changing correction. Resume fingerprints the actual correction factors, so replacing the illumination TIFF at the same path is detected; geometric registration signatures are unaffected.

### Synthetic tests

```bash
cd /path/to/package
OPENBLAS_NUM_THREADS=1 python test_auto_xyz_flexible_export.py
```

The thirteen integration tests cover real synthetic TIFF inputs: three-layer XYZ recovery with Z refinement; interrupted rigid/flexible checkpoint resume; cumulative flexible drift greater than the residual cap; bounded patch processing; coordinate signs; missing-file failures; blend selection and sole-layer coverage; cropped/downsampled streaming; LZW/multiple channels; non-divisible XY block averages; uint32 IDs above 4 billion; incompatible-cache rejection; truncated-output rebuilding; TIFF profile validation; native-resolution and downsampled multichannel shading correction; unchanged segmentation labels; rejection of output reuse after a profile changes in place; off-diagonal sparse-signal ROI discovery; blank-pair skip/export/resume; continuation after a skipped layer with nominal absolute positions; independent stages with manual Fiji corrections; stale-field rejection; incomplete-checkpoint rejection; preservation of edits; rigid recomputation backups; matched global-Z overlap TIFFs and XYZ corrections; saved flexible candidate improvement; and checkpoint preview without registration reruns. Tests validate algorithm behavior on small data, not PB throughput or acquisition-specific alignment quality.

See `CODE_REVIEW.md` for the audit findings and remaining limits. TIFF page decoding follows the [tifffile API](https://www.cgohlke.com/docs/tifffile/); rigid correlation uses the shift-to-apply convention of [scikit-image registration](https://scikit-image.org/docs/0.22.x/api/skimage.registration.html).
