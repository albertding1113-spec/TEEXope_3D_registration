# Program correctness review — 2026-10-28

Reviewed the current working-tree registration/export module, launcher, Slurm script, README and tests. Production files were not modified. All 15 existing tests passed using `D:\Anaconda\envs\sknext314\python.exe test_auto_xyz_flexible_export.py`. Additional synthetic checks in `audit_reproductions.py` reproduced the seven findings below. Acquisition data and cluster throughput were unavailable, so this review cannot establish real-data alignment accuracy or PB-scale performance.

## 1. P1 — Outlier filtering restores rejected measurements

**Location:** `auto_xyz_flexible_export.py:456–467`.

The filtered arrays replace the original arrays only if at least three measurements survive. When zero, one, or two survive, interpolation uses every original measurement, including explicitly rejected outliers. The caller subsequently records the layer as `ok`.

**Reproduction:** Five nearby shifts `[100,0], [-100,0], [0,100], [0,-100], [100,100]`, with `max_local_shift_px=256`, all fail the 64-pixel neighborhood-deviation criterion. Nevertheless, the returned 2×2 field contains the rejected ±100-pixel shifts.

**Fix:** Always apply the rejection mask. Treat insufficient surviving support as registration failure, or use an explicitly configured conservative fallback. Do not restore rejected matches.

## 2. P1 — One noisy match can deform an entire layer

**Location:** `auto_xyz_flexible_export.py:473–484` and `746–760`.

The caller requires only one match. With one or two points, interpolation fills the entire field with their componentwise median. With more points, nearest-neighbor extrapolation extends measurements beyond their spatial support. There is no minimum spatial coverage requirement, and the supplied confidence scores are unused in fitting. The resulting field is composed into subsequent layers.

**Reproduction:** One match with residual `[200,0]` and score `0.2` produces X displacement 200 at every node of a 10×10 field.

**Fix:** Require adequate spatial support, robust agreement across sampled Z planes, and confidence-aware fitting. Preserve the previous field where there is no reliable new support, or stop for review. Validate local stretch/shear and folding before accepting the composition.

## 3. P2 — Z refinement can replace an accepted result with a poor match

**Location:** `auto_xyz_flexible_export.py:1058–1067`.

Refinement checks overlap and finiteness but never requires the refined NCC to meet `rigid_min_ncc`. It initializes its best score to a very negative number, so even a negative NCC can win. The final refined result is not revalidated against the acceptance thresholds; returned QC still summarizes coarse ROI scores.

**Reproduction:** With all refinement trials reporting NCC −0.5, required NCC 0.2, coarse dz=0 and search radius 2, refinement returns dz=−2 because the first poor trial wins.

**Fix:** Retain the coarse solution unless a sufficiently supported, acceptable refinement improves the evaluated baseline. Revalidate and report the final transform's score and overlap.

## 4. P2 — Half-voxel Z offsets duplicate and omit source pages

**Location:** `auto_xyz_flexible_export.py:851` and `1340`.

Manual CSV offsets accept arbitrary finite floats. Local Z lookup uses Python's ties-to-even `round(z-pz)`. At a half-voxel layer start, consecutive output planes can map to repeated source indices, and the last active output plane can map outside the stack.

**Reproduction:** A layer starting at Z=10.5 with depth 4 is active at integer Z=11,12,13,14. Those planes map to local indices `[0,2,2,4]`: index 1 is omitted, index 2 duplicated, and index 4 is out of range. The final plane has no contribution from that layer.

**Fix:** Define an explicit fractional-Z sampling policy. Either reject unsupported fractional corrections, consistently choose nearest source voxel with matching support bounds, or interpolate signal along Z while retaining a defined nearest-neighbor policy for labels. Apply the same policy in registration, previews and export.

## 5. P2 — Downsampled label export loses valid edge pixels

**Location:** `auto_xyz_flexible_export.py:1364–1372`.

Labels are first decimated and then indexed with `rint(rx/dx)`. At a shifted tile's valid far edge, that index can equal the decimated array length, so the pixel is discarded despite valid raw source coverage. This creates gaps that are absent from the matching signal export.

**Reproduction:** Two overlapping 12×12 tiles at X=0 and X=1, filled with signal 100 and label 7, exported at XY factor 4. The last signal column is `[100,100,100]`; the last label column is `[0,0,0]`.

**Fix:** Define label sampling consistently with output coordinates and source coverage. Read/select the intended native source voxel or correctly clamp the decimated lookup within valid coverage. Add checks for shifted tiles and partial output blocks.

## 6. P2 — Disabling rigid registration overwrites its saved checkpoint

**Location:** `auto_xyz_flexible_export.py:1462–1468`.

In `all` or `rigid` stage, `rigid_stitch=False` causes the existing `rigid_state.npz` to be replaced with a zero-offset state even without force. An existing CSV is preserved, leaving state and CSV inconsistent. Reenabling rigid registration then rejects the saved signature. The automatic checkpoint has already been lost.

**Reproduction:** Save automatic layer-1 offset `[3,0,0]`, run the rigid stage with registration disabled, and inspect the checkpoint: the offset is now zero. Reenable registration with the original settings: cache mismatch error.

**Fix:** Bypass saved rigid state when disabled. Do not replace an existing enabled-registration checkpoint with disabled-mode state; keep separate state if disabled-mode persistence is needed.

## 7. P2 — Re-export leaves obsolete TIFFs beside new output

**Location:** `auto_xyz_flexible_export.py:1837–1853`.

With `resume=False`, existing metadata can be replaced and current target files overwritten, but old TIFFs outside the new Z/channel target set remain. Directory-based stack readers can then combine stale files with the new export, even though metadata describes only the new range.

**Reproduction:** Export two Z planes, then re-export a one-plane ROI to the same directory with `resume=False`. Metadata reports one plane, but both `ch00_z000000.tif` and stale `ch00_z000001.tif` remain.

**Fix:** Require a new/empty destination when the target set changes, or explicitly reconcile obsolete pipeline-owned files with a documented overwrite policy. A manifest of current output files would also make consumption safer.

## Other limits to retain in QC

- Sequential registration can accumulate drift; there is no independent global anchor or loop closure.
- Fields are constant through each layer's depth, and individual XY tile positions are not independently registered.
- Source-content changes at the same paths are not fingerprinted. Replacing TIFF data in place can reuse stale registration/output.
- The launcher currently permits 512 raw pixels of flexible residual shift, so accepted noisy matches can have substantial influence. Tightening this limit can reduce exposure but does not repair findings 1–2.
- Existing tests exercise useful TIFF integration paths but do not cover the seven failure cases above.

Run the additional checks with `D:\Anaconda\envs\sknext314\python.exe audit_reproductions.py`. They print observed behavior and clean up their temporary synthetic datasets; they do not modify production code or real acquisition data.
