# Registration and export audit

The current rigid format uses adjacent-layer offsets (`adjacent_layer_v1`).
Saved `dx_raw`, `dy_raw`, `dz_raw` rows are accumulated only when constructing
working coordinates. Editing one pair propagates through all later layers;
automatic XY/Z limits apply per pair. Only the current adjacent-layer checkpoint
format is supported.
The historical audit below describes the earlier absolute-offset format where noted.

The revised pipeline passes thirteen synthetic integration tests, including saved overlap previews, independent workflow stages, manual rigid corrections, TIFF-based illumination standardization, sparse-signal ROI discovery, and skipped rigid pairs. The supplied workspace contains code only; the acquisition path is a Linux cluster path unavailable here. Successful registration of every real layer and PB-scale runtime therefore remain unverified.

## Coordinate interpretation

With `flip_stage_xy=True`, folder `X150700.000_Y39097.500_Z03676.904` represents physical XYZ `(-150700.000, -39097.500, 3676.904)` in micrometers. Input pages are YX, stack/configuration shapes are XYZ. Division by the configured voxel size converts micrometers to raw voxels. The nominal dataset minimum is subtracted and positions are rounded. Rigid corrections are absolute XYZ corrections to these nominal positions, with layer zero fixed. A final normalization shifts the corrected minimum to zero.

The manifest and tile index now record the physical stage origin of normalized raw coordinates. For output raw coordinate q, physical coordinate is `normalized_origin_stage_xyz_um + q * voxel_size`. For XY intensity output, full-sized downsampling blocks are centered at `export_origin + output_index * factor + (factor - 1)/2`; the final partial block has its own center. Z output samples `z0 + index * dz`. Flexible fields map normalized output XY to corrected source XY; physical interpretation describes the stitched reference frame.

## Corrections made

| Finding | Change |
| --- | --- |
| Flexible registration allocated entire-layer mosaics; broad layers could exceed the mosaic cap | Stream bounded patches with spatial KD-tree queries; enforce a padded-patch pixel cap |
| Flexible shift acceptance capped cumulative drift instead of the residual between adjacent layers | Measure both layers through the preceding field and compose the residual with that field |
| Invalid-area boundaries dominated fluorescence features | Use mask-normalized Gaussian filtering before correlation |
| Half-thumbnail ROI origins introduced an extra thumbnail-pixel translation bias | Snap rigid ROI origins to the thumbnail grid |
| Reopening TIFF files per page repeatedly parsed page directories | Bound and reuse open TIFF handles; close them on success and exceptions |
| Missing/corrupt pages silently became black regions | Default to strict reads and check page count/XY geometry on accessed files; permissive mode remains explicit |
| Z blend endpoints discarded the only covering layer at some XY pixels | Keep positive endpoint weights and normalize over available pixels |
| Disabling Z blending globally selected a layer even where it had no XY coverage | Choose the preferred available layer per pixel |
| Disabling XY blending still averaged overlapping intensities | Select one tile per pixel by edge clearance |
| Flattening all XY/Z weights let XY tile multiplicity change Z blend weights | Normalize tile contributions inside each layer before Z blending |
| Downsampling non-divisible dimensions changed the effective source pitch | Average fixed-size raw blocks with correct partial-block centers |
| Downsampled image remapping darkened source edges | Replicate valid edge samples instead of interpolating against zero padding |
| A label lookup outside its downsampled array could still win selection | Restrict label selection to valid lookup indices |
| Existing output files were skipped without inspecting their structure | Validate dimensions/dtype and tile payload extents; rebuild incomplete files |
| Layer indexing repeatedly scanned N tiles for every layer | Group layer IDs with one stable sort and split |
| Export rebuilt the directory index by rescanning every folder | Apply corrections to the existing index |
| Nonrigid displacements could move content beyond rigid export bounds | Expand full export bounds conservatively; keep ROI clipping explicit |
| Physical stage origin was discarded | Save it in registration/export metadata and save folder-to-position mappings in `tile_index.npz` |
| Invalid configuration values could reach divisions or array allocation | Validate channel selection, spatial sizes, ROI, voxel sizes, worker counts and thresholds early |
| Slurm scripts can execute from a spool location | Resolve the launch directory from `SLURM_SUBMIT_DIR` |

Cache signature version is now v2. Registration cache reuse is intentionally rejected across this algorithm change. Permissive reads/failure behavior are part of the signature, so a strict run cannot silently reuse permissive registration.

## Remaining scientific and operational limits

1. **There is no independent XY registration of individual tiles within a layer.** Their relative XYZ origins come from stage names. One rigid translation and one depth-invariant XY field are fitted per Z layer. If within-layer tile offsets are wrong, a tile-neighbor registration graph and global position optimization are still needed.
2. Rigid registration estimates translation only. Flexible registration does not estimate local Z deformation, rotation/scale explicitly, or depth-varying XY deformation. It cannot align data without shared tissue overlap.
3. The nominal stage Z value, rounded to raw voxels, defines each layer. Z jitter among folders can create unintended layers. Check layer counts against the acquisition design. No layer clustering tolerance is implemented.
4. Downsampling limits localization precision. There is no full-resolution XY rigid refinement; with the supplied factor 16, small features or shifts may need a lower factor. The Z refinement searches only around the coarse solution. QC NCC describes accepted coarse ROIs; it is not a certificate of seam accuracy after refinement.
5. Phase correlation can accept incorrect matches in repetitive or sparse tissue. There is no global loop closure, field Jacobian/folding constraint, or spatial uncertainty map. Sequential alignment can accumulate drift. Review QC and actual seams, especially where accepted-match counts are small.
6. Source data are assumed immutable. Registration signatures identify directory names/settings, not source TIFF contents. Export signatures include a checksum of the illumination correction factors, so changed calibration is detected even at the same path. Recompute registration or use a new export directory after replacing source data in place. Concurrent writers sharing registration or export directories are unsupported.
7. Strict validation occurs when a file/page is requested. It is not an exhaustive dataset integrity scan, and registration-only runs do not open all channels or every tile. Resume validates output structure without decompressing/checksumming every tile.
8. A requested page is decoded in full. Registration patches bound assembled-array memory, but source TIFF page/metadata allocations, caches, FFT intermediates, all fields and the directory index add overhead. Checkpoints rewrite the fields array each layer. Hundreds of thousands of folders still impose filesystem metadata costs.
9. Output is serial across planes/channels in one process, with limited compression threading. No GPU, multi-node partitioning, region-level resume, or throughput estimate is implemented. Interruption rewrites the current TIFF plane. Benchmark representative data before allocating a full PB export.
10. Output signal intensities are uint16. Labels use nearest-neighbor and optionally uint32; labels independently assigned in different source tiles are not reconciled into globally unique instances. Z subsampling does not perform anti-alias averaging.

## Validation and first real-data run

Run `python test_auto_xyz_flexible_export.py` in an environment containing all documented dependencies. The audit used `D:\Anaconda\envs\sknext314\python.exe`; other installed environments had missing packages or incompatible NumPy/scikit-image binaries.

The launcher now starts with `workflow_stage="rigid"`. Review the rigid QC and edit `rigid_offsets.csv`, then run `--stage flexible`, inspect flexible QC and run `--stage export` for a cropped seam check. Compare output to source landmarks in several XY regions and Z overlaps before exporting the whole dataset. An ROI affects output only: registration still covers all layers. Keep the same registration directory for subsequent resolutions/channels and use distinct export directories for different settings. Manual offsets are absolute XYZ corrections to nominal stage positions in raw voxels, with layer 0 fixed. Changed offsets invalidate earlier flexible results; forcing the flexible stage recomputes only those fields and preserves the rigid CSV.

The supplied tests exercise real TIFF read/write paths, XYZ recovery, cumulative flexible drift, both checkpoint resumes, cache mismatch rejection, crop, exact non-divisible XY block averages, multiple channels, LZW, uint32 labels, blend fallbacks, and structural output repair. No real acquisition or PB throughput benchmark was possible in this workspace.
