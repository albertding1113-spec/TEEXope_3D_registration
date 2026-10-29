#!/usr/bin/env python3
"""
Out-of-core PB-scale TIFF exporter: automatic coarse XYZ translation registration,
then flexible XY displacement fields, then streamed TIFF output.

Dependencies: numpy, scipy, opencv-python-headless, scikit-image, tifffile,
imagecodecs (for LZW output). Input TIFF axes Z,Y,X; stage/config XYZ.
The 3D rigid step translates XYZ but DOES NOT deform Z nonrigidly.
The flexible step estimates XY sample offsets varying by XY, not by depth.
Both stages checkpoint each Z layer; exports reuse their registration directory.
Rigid checkpoints and editable CSV rows store adjacent-layer XYZ corrections.
All geometry stages accumulate those corrections internally; automatic limits are per pair.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import time
from collections import OrderedDict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import tifffile
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.spatial import cKDTree

NAME_RE = re.compile(r"^X([-+]?\d+(?:\.\d+)?)_Y([-+]?\d+(?:\.\d+)?)_Z([-+]?\d+(?:\.\d+)?)$")
RIGID_OFFSET_CONVENTION = 'adjacent_layer_v1'


def accumulate_rigid_offsets(relative: np.ndarray) -> np.ndarray:
    """Convert saved adjacent-layer corrections into internal absolute corrections."""
    relative = np.asarray(relative, dtype=np.float64)
    if relative.ndim != 2 or relative.shape[1] != 3 or not len(relative) or not np.all(np.isfinite(relative)):
        raise ValueError('Rigid offsets must be a finite nonempty [layers,3] XYZ array')
    if np.any(relative[0] != 0):
        raise ValueError('Rigid reference layer 0 must retain XYZ correction [0,0,0]')
    with np.errstate(over='ignore', invalid='ignore'):
        absolute = np.cumsum(relative, axis=0)
    if not np.all(np.isfinite(absolute)):
        raise ValueError('Accumulated rigid offsets must be finite')
    return absolute


def relative_rigid_offsets(absolute: np.ndarray) -> np.ndarray:
    """Difference absolute corrections; row i corrects layer i relative to i-1."""
    absolute = np.asarray(absolute, dtype=np.float64)
    if absolute.ndim != 2 or absolute.shape[1] != 3 or not len(absolute) or not np.all(np.isfinite(absolute)):
        raise ValueError('Rigid offsets must be a finite nonempty [layers,3] XYZ array')
    if np.any(absolute[0] != 0):
        raise ValueError('Rigid reference layer 0 must retain XYZ correction [0,0,0]')
    with np.errstate(over='ignore', invalid='ignore'):
        relative = np.diff(absolute, axis=0, prepend=np.zeros((1, 3)))
    if not np.all(np.isfinite(relative)):
        raise ValueError('Pairwise rigid offsets must be finite')
    return relative


def progress(message: str, *args) -> None:
    """Always emit progress immediately, including when stdout is redirected."""
    if args:
        message = message % args
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


class RigidRegistrationError(RuntimeError):
    """No trustworthy alignment; distinct from I/O/configuration/job failures."""


@dataclass
class Config:
    dir_path: str = "/HWFS/zhaohu_lab/dingjiayi/data/261009_mito-mBaoJin_Cdh5-mScarlet3H"
    export_path: str = "/HWFS/zhaohu_lab/dingjiayi/data/261009_mito-mBaoJin_Cdh5-mScarlet3H/flexible_export"
    ch_num: int = 2
    stack_shape: tuple[int, int, int] = (2048, 2048, 640)  # XYZ, raw pixels
    voxel_size: tuple[float, float, float] = (0.1489, 0.1489, 0.5)  # XYZ, um
    flip_stage_xy: bool = True  # preserve original position conversion
    layer_z_tolerance_um: float = 1.0  # group Z spans strictly below this; 0 keeps legacy voxel grouping
    registration_dir: Optional[str] = None  # independent of export_path; default: dir_path/_registration_xyz_flexible
    registration_only: bool = False  # compute/cache alignment without exporting TIFF images
    workflow_stage: str = "all"  # all, rigid, flexible, export, preview
    save_registration_previews: bool = True
    registration_preview_flexible_patches: int = 3  # per sampled plane, bounded diagnostic output
    rigid_stitch: bool = True  # automatically estimate XYZ translations BEFORE flexible XY
    rigid_xy_downsample: int = 16
    rigid_z_downsample: int = 8
    rigid_roi_px: int = 4096  # XY region side in raw pixels (3D coarse registration)
    rigid_roi_count: int = 3  # maximum signal-ranked 3D matching regions
    rigid_signal_candidates: int = 64  # bounded scouting budget, not full 3D volumes
    rigid_signal_z_samples: int = 5  # planes per candidate per layer
    rigid_skip_on_failure: bool = True  # save zero pair shift and inherit preceding correction on failure
    rigid_max_xy_shift_px: float = 512.0  # maximum additional X/Y correction per adjacent layer
    rigid_max_z_shift_px: float = 256.0  # maximum additional Z correction per adjacent layer
    rigid_min_ncc: float = 0.10  # normalized cross correlation of valid 3D voxels
    rigid_min_overlap: float = 0.12  # required valid common fraction after shifting
    rigid_max_volume_megavoxels: float = 8.0  # per 3D ROI thumbnail
    rigid_refine_z: bool = True  # refine coarse axial shift to single raw Z voxel
    rigid_refine_z_planes: int = 5
    stop_on_registration_failure: bool = True  # flexible failures; rigid too when skip is disabled

    if_downsample: bool = True
    downsample_factor: tuple[int, int, int] = (5, 5, 100)  # XYZ
    z_downsample_mode: str = "sample"  # 'sample': read 1 Z plane every dz (not averaging)
    if_xy_blend: bool = True
    xy_blend_overlap_ratio: float = 0.05
    if_z_blend: bool = True
    if_standardization: bool = False
    stand_img_path: Optional[str] = None  # native YX illumination profile for registration and signal export
    segmented_ch: tuple[str, ...] = ()
    segmentation_mode: str = "binary_255"  # or 'preserve_labels'; NN, never averaging

    if_subregion: bool = False
    subregion_range: tuple[tuple[int, int], ...] = ((41500, 44500), (23000, 26000), (3104, 3300))
    export_margin_xy: int = 0  # optional margin, in raw pixels

    compression: Optional[str] = "lzw"  # None or 'lzw' (requires imagecodecs)
    tiff_tile: int = 256  # TIFF block side; a multiple of 16
    tiff_maxworkers: int = 2  # concurrent TIFF tile compression workers
    opencv_threads: int = 1  # avoid nested thread pools on Slurm nodes
    cache_mb: int = 512  # decompressed (and XY downsampled) source page LRU cache
    cache_open_files: int = 64  # bounded LRU of TIFF handles and page directories
    strict_input: bool = True  # missing/corrupt requested pages stop the job
    resume: bool = True

    flexible_stitch: bool = True
    registration_channel: int = 0  # a shared fluorophore/signal between layers
    registration_downsample: int = 16  # XY thumbnail; independent of export downsample
    registration_mesh_px: int = 4096  # control-point spacing in RAW pixels
    registration_patch_px: int = 8192  # local phase correlation patch width in RAW pixels
    registration_z_samples: int = 1  # 1 or 3 usually adequate
    min_phase_response: float = 0.10
    min_patch_coverage: float = 0.50
    max_local_shift_px: float = 256.0  # residual XY registration cap after automatic XYZ rigid shift
    field_smoothing_sigma: float = 0.8  # units: control-grid nodes
    force_registration: bool = False
    max_registration_megapixels: float = 50.0  # memory protection, per thumbnail


@dataclass
class Index:
    dirs: list[Path]
    positions: np.ndarray  # Nx3, float raw voxels, normalized after rigid shifts
    layer_ids: np.ndarray  # N
    layer_starts: np.ndarray  # L, raw z positions
    layer_tiles: list[np.ndarray]
    xy_min: np.ndarray  # XY original export origin (often 0,0)
    xy_max: np.ndarray  # XY exclusive
    z_min: int
    z_max: int  # exclusive
    layer_trees: list[cKDTree]
    stage_origin_um: Optional[np.ndarray] = None  # physical XYZ at normalized raw origin


class PageCache:
    """Bounded LRU of *post-XY-downsampling* single 2D TIFF pages."""
    def __init__(self, capacity_mb: int, max_open_files: int = 64,
                 strict: bool = True, expected_shape: Optional[tuple] = None,
                 standard_factor: Optional[np.ndarray] = None):
        self.limit = int(capacity_mb * 1024 * 1024)
        self.cache = OrderedDict()
        self.nbytes = 0
        self.handles = OrderedDict()
        self.max_open_files = max(1, max_open_files)
        self.strict = strict
        self.expected_shape = expected_shape
        self.standard_factor = standard_factor

    def close(self):
        for handle in self.handles.values():
            handle.close()
        self.handles.clear()

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def read(self, path: Path, z_page: int, xy_factor: tuple[int, int], nearest: bool,
             standard_factor: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        # Labels are never intensity-corrected. Distinguish corrected cache entries.
        factor = None if nearest else (standard_factor if standard_factor is not None
                                       else self.standard_factor)
        key = (str(path), int(z_page), *xy_factor, bool(nearest), id(factor))
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        try:
            name = str(path)
            if name not in self.handles:
                if len(self.handles) >= self.max_open_files:
                    _, old = self.handles.popitem(last=False)
                    old.close()
                self.handles[name] = tifffile.TiffFile(path)
            self.handles.move_to_end(name)
            handle = self.handles[name]
            if self.expected_shape is not None:
                sx, sy, sz = self.expected_shape
                if len(handle.pages) != sz or handle.pages[0].shape != (sy, sx):
                    raise ValueError(f"Expected {sz} YX pages of shape {(sy, sx)}, "
                                     f"got {len(handle.pages)} pages, {handle.pages[0].shape}")
            im = handle.pages[int(z_page)].asarray(maxworkers=1)  # YX
            if im.ndim != 2:
                raise ValueError(f"Expected one 2D Z page, got {im.shape}: {path}")
        except (OSError, IndexError, KeyError, ValueError) as exc:
            if self.strict:
                raise OSError(f"Cannot read {path} at z={z_page}: {exc}") from exc
            progress("WARNING: Cannot read %s at z=%s: %s", path, z_page, exc)
            return None
        if factor is not None:
            if factor.shape != im.shape:
                raise ValueError(f"Illumination correction shape {factor.shape} does not match "
                                 f"source page {im.shape}: {path}")
            # Correct native detector pixels before averaging or spatial remapping.
            im = im.astype(np.float32) * factor
        dx, dy = xy_factor
        if (dx, dy) != (1, 1):
            hh, ww = im.shape
            if nearest:
                # Exact 32-bit instance IDs: OpenCV does not safely resize uint32.
                im = im[::dy, ::dx]
            else:
                # Fixed-size raw blocks, including a smaller final block.
                # Resizing to ceil(W/dx) would instead change the sampling pitch.
                rows = np.arange(0, hh, dy)
                cols = np.arange(0, ww, dx)
                sums = np.add.reduceat(im.astype(np.float32), rows, axis=0)
                sums = np.add.reduceat(sums, cols, axis=1)
                counts = np.minimum(dy, hh-rows)[:, None] * np.minimum(dx, ww-cols)[None, :]
                im = (sums / counts).astype(np.float32)
        im = np.ascontiguousarray(im)
        if im.nbytes <= self.limit:
            while self.cache and self.nbytes + im.nbytes > self.limit:
                _, old = self.cache.popitem(last=False)
                self.nbytes -= old.nbytes
            self.cache[key] = im
            self.nbytes += im.nbytes
        return im


def load_index(c: Config, rigid_offsets: Optional[np.ndarray] = None,
               normalize: bool = False) -> Index:
    """Index source tiles; optional rigid_offsets are saved adjacent-layer corrections."""
    if not np.isfinite(c.layer_z_tolerance_um) or c.layer_z_tolerance_um < 0:
        raise ValueError("layer_z_tolerance_um must be finite and nonnegative")
    root = Path(c.dir_path)
    entries = []
    for e in os.scandir(root):
        if e.is_dir():
            m = NAME_RE.fullmatch(e.name)
            if m:
                entries.append((e.name, Path(e.path), [float(v) for v in m.groups()]))
    entries.sort(key=lambda e: e[0])
    if not entries:
        raise RuntimeError(f"No X..._Y..._Z... tile directories found under {root}")

    dirs = [v[1] for v in entries]
    physical = np.array([v[2] for v in entries], dtype=np.float64)
    if c.flip_stage_xy:
        physical[:, :2] *= -1
    if c.layer_z_tolerance_um > 0:
        # Cluster physical coordinates before voxel rounding. Anchor each group
        # at its minimum so a chain of close neighbors cannot span many layers.
        z_values, z_inverse = np.unique(physical[:, 2], return_inverse=True)
        grouped_z = np.empty_like(z_values)
        anchor = z_values[0]
        for i, z in enumerate(z_values):
            if z - anchor >= c.layer_z_tolerance_um:
                anchor = z
            grouped_z[i] = anchor
        physical[:, 2] = grouped_z[z_inverse]
    pos = physical / np.asarray(c.voxel_size, dtype=np.float64)
    pos = np.rint(pos - pos.min(axis=0)).astype(np.float64)

    starts, layer_ids = np.unique(pos[:, 2], return_inverse=True)
    n_layers = len(starts)
    stage_origin = physical.min(axis=0)
    if rigid_offsets is not None:
        offsets = accumulate_rigid_offsets(rigid_offsets)
        if offsets.shape != (n_layers, 3):
            raise ValueError(f"rigid offsets must have shape ({n_layers},3): {offsets.shape}")
        pos += offsets[layer_ids]
        starts = starts + offsets[:, 2]
        if normalize:
            shift = pos.min(axis=0)
            pos -= shift  # export pixel origins follow original convention
            starts -= shift[2]
            stage_origin += shift * np.asarray(c.voxel_size)
    order = np.argsort(layer_ids, kind="stable")
    layer_tiles = list(np.split(order, np.cumsum(np.bincount(layer_ids))[:-1]))
    # Retain acquisition layer identity even if shifted Z positions are modified.
    xyz_shape = np.asarray(c.stack_shape)
    xy_min = pos[:, :2].min(axis=0) - c.export_margin_xy
    xy_max = np.max(pos[:, :2] + xyz_shape[:2], axis=0) + c.export_margin_xy
    zmin = int(np.floor(pos[:, 2].min()))
    zmax = int(np.ceil(np.max(pos[:, 2] + xyz_shape[2])))
    trees = [cKDTree(pos[ids, :2] + xyz_shape[:2] / 2) for ids in layer_tiles]
    progress("Indexed %s tiles, %s Z layers. Raw XY = %.0f x %.0f pixels, Z=[%s,%s)",
             len(dirs), n_layers, *(xy_max - xy_min), zmin, zmax)
    return Index(dirs, pos, layer_ids, starts, layer_tiles, xy_min, xy_max, zmin, zmax, trees,
                 stage_origin)


def corrected_index(c: Config, idx: Index, offsets: np.ndarray) -> Index:
    """Accumulate saved pairwise corrections without a second directory traversal."""
    offsets = accumulate_rigid_offsets(offsets)
    if offsets.shape != (len(idx.layer_starts),3):
        raise ValueError('Rigid offsets must have one XYZ row per layer')
    pos = idx.positions + offsets[idx.layer_ids]
    shift = pos.min(axis=0)
    pos -= shift
    shape = np.asarray(c.stack_shape)
    trees = [cKDTree(pos[ids, :2] + shape[:2]/2) for ids in idx.layer_tiles]
    return replace(idx, positions=pos,
                   layer_starts=idx.layer_starts + offsets[:, 2] - shift[2],
                   xy_min=pos[:, :2].min(axis=0)-c.export_margin_xy,
                   xy_max=(pos[:, :2]+shape[:2]).max(axis=0)+c.export_margin_xy,
                   z_min=math.floor(pos[:, 2].min()),
                   z_max=math.ceil((pos[:, 2]+shape[2]).max()), layer_trees=trees,
                   stage_origin_um=idx.stage_origin_um + shift*np.asarray(c.voxel_size))


def channel_path(c: Config, idx: Index, tile_id: int, channel: int) -> Path:
    if channel < c.ch_num:
        return idx.dirs[tile_id] / "0" / f"ch{channel:02d}.ome.tif"
    segment = c.segmented_ch[channel - c.ch_num]
    return idx.dirs[tile_id] / "segmentation" / segment / f"{segment}.tif"


def feather_weight(local_x: np.ndarray, local_y: np.ndarray, wx: int, wy: int,
                   blend: bool, ratio: float) -> np.ndarray:
    if not blend:
        return np.ones_like(local_x, dtype=np.float32)
    border_x = max(1., wx * ratio)
    border_y = max(1., wy * ratio)
    ux = np.minimum(local_x, wx - 1 - local_x)
    uy = np.minimum(local_y, wy - 1 - local_y)
    return np.maximum(0.01, np.minimum(1.0, ux / border_x)) * \
           np.maximum(0.01, np.minimum(1.0, uy / border_y))


def layer_mosaic(c: Config, idx: Index, layer: int, z: int,
                 ds: int, cache: PageCache) -> tuple[np.ndarray, np.ndarray]:
    """One small YX thumbnail of a single original layer (no registration warp)."""
    width = int(math.ceil((idx.xy_max[0] - idx.xy_min[0]) / ds))
    height = int(math.ceil((idx.xy_max[1] - idx.xy_min[1]) / ds))
    total = width * height
    if total > c.max_registration_megapixels * 1e6:
        raise MemoryError(f"Registration mosaic {width}x{height} = {total/1e6:.1f} MP; "
                          "increase registration_downsample or restrict data")
    sums = np.zeros((height, width), np.float32)
    weights = np.zeros_like(sums)
    sx, sy, sz = c.stack_shape
    for tile_id in idx.layer_tiles[layer]:
        origin = idx.positions[tile_id]
        local_z = int(round(z - origin[2]))
        if not 0 <= local_z < sz:
            continue
        arr = cache.read(channel_path(c, idx, int(tile_id), c.registration_channel),
                         local_z, (ds, ds), False)
        if arr is None:
            continue
        h, w = arr.shape
        xx = int(round((origin[0] - idx.xy_min[0]) / ds))
        yy = int(round((origin[1] - idx.xy_min[1]) / ds))
        x0, x1 = max(0, xx), min(width, xx + w)
        y0, y1 = max(0, yy), min(height, yy + h)
        if x0 >= x1 or y0 >= y1:
            continue
        xa, xb = x0 - xx, x1 - xx
        ya, yb = y0 - yy, y1 - yy
        pix = arr[ya:yb, xa:xb].astype(np.float32)
        xxg = np.arange(xa, xb, dtype=np.float32)[None, :]
        yyg = np.arange(ya, yb, dtype=np.float32)[:, None]
        edge = feather_weight(np.broadcast_to(xxg, pix.shape),
                              np.broadcast_to(yyg, pix.shape), w, h,
                              True, c.xy_blend_overlap_ratio)
        sums[y0:y1, x0:x1] += pix * edge
        weights[y0:y1, x0:x1] += edge
    valid = weights > 0
    np.divide(sums, weights, out=sums, where=valid)
    return sums, valid


def sample_field(field: np.ndarray, xs_raw: np.ndarray, ys_raw: np.ndarray,
                 min_xy: np.ndarray, mesh_px: int) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate 2-channel displacement field (in RAW XY pixels) at global XY coordinates."""
    grid_x = (xs_raw - min_xy[0]) / mesh_px
    grid_y = (ys_raw - min_xy[1]) / mesh_px
    coords = [grid_y, grid_x]
    dx = map_coordinates(field[:, :, 0], coords, order=1, mode="nearest").astype(np.float32)
    dy = map_coordinates(field[:, :, 1], coords, order=1, mode="nearest").astype(np.float32)
    return dx, dy


def field_grid(c: Config, idx: Index) -> tuple[np.ndarray, np.ndarray]:
    nx = max(2, math.ceil((idx.xy_max[0] - idx.xy_min[0]) / c.registration_mesh_px) + 1)
    ny = max(2, math.ceil((idx.xy_max[1] - idx.xy_min[1]) / c.registration_mesh_px) + 1)
    x_coords = idx.xy_min[0] + c.registration_mesh_px * np.arange(nx)
    y_coords = idx.xy_min[1] + c.registration_mesh_px * np.arange(ny)
    return x_coords, y_coords


def warped_mosaic(im: np.ndarray, valid: np.ndarray, field: np.ndarray,
                  idx: Index, c: Config, ds: int) -> tuple[np.ndarray, np.ndarray]:
    h, w = im.shape
    # Build one low-res lookup map for this registration mosaic.
    xs = idx.xy_min[0] + np.arange(w, dtype=np.float32)[None, :] * ds
    ys = idx.xy_min[1] + np.arange(h, dtype=np.float32)[:, None] * ds
    coords_x = np.broadcast_to(xs, (h, w))
    coords_y = np.broadcast_to(ys, (h, w))
    dx, dy = sample_field(field, coords_x, coords_y, idx.xy_min, c.registration_mesh_px)
    map_x = np.broadcast_to(np.arange(w, dtype=np.float32)[None, :], (h, w)) + dx / ds
    map_y = np.broadcast_to(np.arange(h, dtype=np.float32)[:, None], (h, w)) + dy / ds
    return (cv2.remap(im, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT),
            cv2.remap(valid.astype(np.uint8), map_x, map_y, cv2.INTER_NEAREST,
                      borderMode=cv2.BORDER_CONSTANT) > 0)


def feature_image(src: np.ndarray, valid: Optional[np.ndarray] = None) -> np.ndarray:
    # Log transform preserves fine morphology across a large fluorescence dynamic range.
    a = np.log1p(np.maximum(src, 0)).astype(np.float32)
    if valid is not None:
        # Normalized convolution prevents tile/missing-data boundaries from
        # dominating morphology in fluorescence images with a large baseline.
        mask = valid.astype(np.float32)
        def blur(sigma):
            num = cv2.GaussianBlur(a * mask, (0, 0), sigma)
            den = cv2.GaussianBlur(mask, (0, 0), sigma)
            return np.divide(num, den, out=np.zeros_like(num), where=den > 1e-6)
        result = blur(1.) - blur(10.)
        result[~valid] = 0
        return result
    a = cv2.GaussianBlur(a, (0, 0), 1.0)
    return a - cv2.GaussianBlur(a, (0, 0), 10.0)


def local_phases(c: Config, ref: np.ndarray, mov: np.ndarray,
                 ref_valid: np.ndarray, mov_valid: np.ndarray,
                 sample_z: int) -> tuple[list, list, list]:
    """Return (center_x_reg,center_y_reg), [dx_reg,dy_reg], and responses."""
    h, w = ref.shape
    step = max(16, int(round(c.registration_mesh_px / c.registration_downsample)))
    patch = max(64, int(round(c.registration_patch_px / c.registration_downsample)))
    # Evaluate patches at REGULAR grid points to ensure spatial coverage.
    xs = np.arange(step // 2, w, step, dtype=int)
    ys = np.arange(step // 2, h, step, dtype=int)
    points, shifts, scores = [], [], []
    ar = feature_image(ref)
    am = feature_image(mov)
    maxshift = c.max_local_shift_px / c.registration_downsample
    for cy in ys:
        y0 = max(0, int(cy - patch // 2))
        y1 = min(h, y0 + patch)
        y0 = max(0, y1 - patch)
        if y1 - y0 < 32:
            continue
        for cx in xs:
            x0 = max(0, int(cx - patch // 2))
            x1 = min(w, x0 + patch)
            x0 = max(0, x1 - patch)
            if x1 - x0 < 32:
                continue
            common = ref_valid[y0:y1, x0:x1] & mov_valid[y0:y1, x0:x1]
            if common.mean() < c.min_patch_coverage:
                continue
            a = ar[y0:y1, x0:x1].copy()
            b = am[y0:y1, x0:x1].copy()
            a[~common] = 0
            b[~common] = 0
            if min(float(a.std()), float(b.std())) < 1e-4:
                continue
            win = cv2.createHanningWindow((a.shape[1], a.shape[0]), cv2.CV_32F)
            delta, response = cv2.phaseCorrelate(a, b, win)
            if (not np.all(np.isfinite(delta)) or not np.isfinite(response) or
                    response < c.min_phase_response or np.linalg.norm(delta) > maxshift):
                continue
            points.append((float(cx), float(cy)))
            shifts.append(delta)
            scores.append(float(response))
    progress("Z=%d: accepted %d local matches", sample_z, len(points))
    return points, shifts, scores


def interpolate_matches(points: list, shifts: list, scores: list,
                        c: Config, idx: Index, field_shape: tuple) -> np.ndarray:
    """Robust 2D interpolation + Gaussian smoothing of scattered XY offsets."""
    ny, nx = field_shape
    out = np.zeros((ny, nx, 2), np.float32)
    if not points:
        return out
    p = np.asarray(points, np.float64)
    v = np.asarray(shifts, np.float64) * c.registration_downsample  # to RAW voxels
    # Locally reject inconsistent/low-confidence matches before interpolation.
    if len(v) >= 5:
        tree = cKDTree(p)
        keep = np.ones(len(p), bool)
        radius = 2.5 * c.registration_mesh_px / c.registration_downsample
        for i, nbrs in enumerate(tree.query_ball_point(p, r=radius)):
            if len(nbrs) >= 4:
                med = np.median(v[nbrs], axis=0)
                if np.linalg.norm(v[i] - med) > max(12.0, 0.25 * c.max_local_shift_px):
                    keep[i] = False
        if np.count_nonzero(keep) >= 3:
            p, v = p[keep], v[keep]
    xv = np.arange(nx) * c.registration_mesh_px / c.registration_downsample
    yv = np.arange(ny) * c.registration_mesh_px / c.registration_downsample
    # registration mosaic pixel origin is idx.xy_min, so coords are just relative.
    gx, gy = np.meshgrid(xv, yv)
    for ch in range(2):
        if len(p) >= 3:
            # QHull may fail when all points lie on a line.
            try:
                interp = griddata(p, v[:, ch], (gx, gy), method="linear")
                nearest = griddata(p, v[:, ch], (gx, gy), method="nearest")
                interp = np.where(np.isfinite(interp), interp, nearest)
            except Exception:
                interp = np.full((ny, nx), np.median(v[:, ch]), np.float64)
        else:
            interp = np.full((ny, nx), np.median(v[:, ch]), np.float64)
        out[:, :, ch] = gaussian_filter(interp.astype(np.float32),
                                        c.field_smoothing_sigma, mode="nearest")
    return out


def registration_preview_root(c: Config) -> Path:
    regdir = Path(c.registration_dir) if c.registration_dir else Path(c.dir_path)/"_registration_xyz_flexible"
    return regdir/"overlap_previews"


def write_preview_json(path: Path, metadata: dict) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp = path.with_name(path.name+".partial")
    tmp.write_text(json.dumps(metadata,indent=2,allow_nan=False),encoding='utf-8')
    os.replace(tmp,path)


def save_region_preview(folder: Path, ref: np.ndarray, mov: np.ndarray,
                        rv: np.ndarray, mv: np.ndarray, metadata: dict) -> None:
    """Numeric TIFFs for Fiji and red/green maximum projections for quick review."""
    folder.mkdir(parents=True,exist_ok=True)
    for name,data,mask in (("reference",ref,rv),("moving",mov,mv),
                           ("reference_valid",rv,None),("moving_valid",mv,None)):
        arr = (np.where(mask,data,0).astype(np.float32) if mask is not None
               else data.astype(np.uint8))
        path = folder/(name+'.tif')
        tmp = path.with_name(path.name+'.partial')
        tifffile.imwrite(tmp,arr,photometric='minisblack',
                         metadata={'axes':'ZYX' if arr.ndim == 3 else 'YX'})
        os.replace(tmp,path)
    views = [('xy',0),('xz',1),('yz',2)] if ref.ndim == 3 else [('xy',None)]
    ranges = {}
    for name,axis in views:
        if axis is None:
            a = np.where(rv,ref,0); b = np.where(mv,mov,0)
        else:
            a = ref.max(axis=axis,where=rv,initial=0)
            b = mov.max(axis=axis,where=mv,initial=0)
        values = np.concatenate([a[a > 0],b[b > 0]])
        lo,hi = np.percentile(values,[1,99.5]) if values.size else (0.,1.)
        if hi <= lo:
            lo,hi = 0.,max(1.,float(hi))
        red = np.clip((a-lo)/(hi-lo)*255,0,255).astype(np.uint8)
        green = np.clip((b-lo)/(hi-lo)*255,0,255).astype(np.uint8)
        rgb = np.stack([red,green,np.zeros_like(red)],axis=-1)
        ok,encoded = cv2.imencode('.png',rgb[:,:,::-1])
        if not ok:
            raise OSError(f"Cannot encode overlap preview: {folder}")
        path = folder/('overlay_'+name+'.png')
        tmp = path.with_name(path.name+'.partial')
        tmp.write_bytes(encoded.tobytes()); os.replace(tmp,path)
        ranges[name] = [float(lo),float(hi)]
    metadata = dict(metadata,shape=list(ref.shape),
                    axes='ZYX' if ref.ndim == 3 else 'YX',
                    overlay='red=reference, green=moving, yellow=coincident projected intensity',
                    overlay_display_ranges=ranges,
                    note='TIFF values are unscaled registration thumbnails; masks mark source coverage. '
                         'Projection overlays alone cannot establish 3D correspondence.')
    write_preview_json(folder/'metadata.json',metadata)
    progress("Saved overlap preview: %s",folder)


def save_rigid_overlap_preview(c: Config, idx: Index, prev: int, cur: int,
                               cx: float, cy: float, cache: PageCache,
                               folder: Path, correction_xyz, **details) -> None:
    """Read both ROIs on matching global Z planes after a proposed moving shift."""
    if not c.save_registration_previews:
        return
    dx,dy,dz = np.asarray(correction_xyz,dtype=float)
    lo = math.ceil(max(idx.layer_starts[prev],idx.layer_starts[cur]+dz))
    hi = math.ceil(min(idx.layer_starts[prev],idx.layer_starts[cur]+dz)+c.stack_shape[2])
    metadata = dict(reference_layer=int(prev),moving_layer=int(cur),
                    source=str(Path(c.dir_path).resolve()),
                    coordinate_frame='Registration working XYZ, before final dataset-origin normalization; '
                                     'automatic rigid matching initializes the moving layer with the preceding '
                                     'absolute correction, and candidate corrections are additional residuals',
                    reference_layer_start_raw=float(idx.layer_starts[prev]),
                    moving_layer_start_raw=float(idx.layer_starts[cur]),
                    registration_channel=int(c.registration_channel),
                    roi_center_xy_raw=[float(cx),float(cy)],
                    xy_origin_raw=[float(cx-c.rigid_roi_px/2),float(cy-c.rigid_roi_px/2)],
                    roi_width_raw=int(c.rigid_roi_px),
                    moving_placement_correction_xyz_raw=[float(dx),float(dy),float(dz)],
                    z_overlap_raw=[lo,hi],xy_downsample=int(c.rigid_xy_downsample),
                    z_downsample=int(c.rigid_z_downsample),**details)
    if hi <= lo:
        write_preview_json(folder/'metadata.json',dict(metadata,status='no_z_overlap'))
        progress("No Z overlap to preview: %s",folder)
        return
    zs = np.arange(lo,hi,c.rigid_z_downsample)
    wh = math.ceil(c.rigid_roi_px/c.rigid_xy_downsample)
    if len(zs)*wh*wh > c.rigid_max_volume_megavoxels*1e6:
        raise MemoryError("Overlap preview exceeds rigid_max_volume_megavoxels")
    ref = np.zeros((len(zs),wh,wh),np.float32); mov = np.zeros_like(ref)
    rv = np.zeros(ref.shape,bool); mv = np.zeros_like(rv)
    x0,y0 = metadata['xy_origin_raw']
    for k,z in enumerate(zs):
        ref[k],rv[k] = roi_mosaic(c,idx,prev,int(z),c.rigid_xy_downsample,x0,y0,c.rigid_roi_px,cache)
        mov[k],mv[k] = roi_mosaic(c,idx,cur,float(z-dz),c.rigid_xy_downsample,
                                 x0-dx,y0-dy,c.rigid_roi_px,cache)
    metadata['sampled_global_z_raw'] = zs.tolist()
    metadata['reference_local_z_indices'] = np.rint(zs-idx.layer_starts[prev]).astype(int).tolist()
    metadata['moving_local_z_indices'] = np.rint(zs-dz-idx.layer_starts[cur]).astype(int).tolist()
    metadata['common_valid_fraction'] = float((rv & mv).mean())
    save_region_preview(folder,ref,mov,rv,mv,metadata)


def streaming_local_matches(c: Config, idx: Index, prev: int, cur: int, z: int,
                            previous_field: np.ndarray, cache: PageCache):
    """Measure residual shifts through the previous field, using bounded XY patches."""
    ds = c.registration_downsample
    width = math.ceil((idx.xy_max[0] - idx.xy_min[0]) / ds)
    height = math.ceil((idx.xy_max[1] - idx.xy_min[1]) / ds)
    step = max(16, round(c.registration_mesh_px / ds))
    patch = max(64, round(c.registration_patch_px / ds))
    pad = math.ceil(float(np.max(np.abs(previous_field))) / ds) + 2
    side = patch + 2 * pad
    if side * side > c.max_registration_megapixels * 1e6:
        raise MemoryError("Flexible patch exceeds max_registration_megapixels; increase "
                          "registration_downsample or decrease registration_patch_px")
    points, shifts, scores = [], [], []
    xs = np.arange(step // 2, width, step) if width > step // 2 else [width // 2]
    ys = np.arange(step // 2, height, step) if height > step // 2 else [height // 2]
    total = len(xs) * len(ys)
    every = max(1, math.ceil(total / 20))
    attempted = low_coverage = low_texture = rejected = 0
    started = time.perf_counter()
    preview_limit = c.registration_preview_flexible_patches if c.save_registration_previews else 0
    preview_folder = (registration_preview_root(c)/'flexible'/f'layer_{prev:04d}_to_{cur:04d}'/
                      f'attempt_{time.time_ns()}'/f'z_{z:06d}')
    saved_previews = 0
    fallback_preview = None
    def record_patch(status, delta=None, response=None):
        nonlocal saved_previews,fallback_preview
        if not preview_limit or saved_previews >= preview_limit:
            return
        details = dict(status=status,reference_layer=prev,moving_layer=cur,sampled_global_z_raw=int(z),
                       registration_channel=c.registration_channel,xy_downsample=ds,
                       xy_origin_raw=[float(x0+pad*ds),float(y0+pad*ds)],
                       common_valid_fraction=float((rv & mv).mean()),
                       coordinates='Both layers sampled through the preceding cumulative XY field',
                       residual_sampling_shift_xy_raw=(np.asarray(delta)*ds).tolist()
                           if delta is not None and np.all(np.isfinite(delta)) else None,
                       phase_response=float(response) if response is not None and np.isfinite(response) else None)
        if status == 'accepted' and saved_previews < preview_limit:
            folder = preview_folder/f'patch_{attempted:06d}'
            save_region_preview(folder/'before_residual',ref,mov,rv,mv,details)
            xx,yy = np.meshgrid(np.arange(patch,dtype=np.float32),np.arange(patch,dtype=np.float32))
            mx = np.ascontiguousarray(xx+delta[0],dtype=np.float32)
            my = np.ascontiguousarray(yy+delta[1],dtype=np.float32)
            aligned = cv2.remap(mov,mx,my,cv2.INTER_LINEAR)
            mask = cv2.remap(mv.astype(np.uint8),mx,my,cv2.INTER_NEAREST) > 0
            save_region_preview(folder/'candidate_aligned',ref,aligned,rv,mask,details)
            saved_previews += 1
            fallback_preview = None
        elif status != 'accepted' and saved_previews == 0 and fallback_preview is None:
            fallback_preview = (attempted,ref.copy(),mov.copy(),rv.copy(),mv.copy(),details)
    progress("Flexible layers %d->%d, Z=%d: %d patches, patch=%dx%d, padded=%dx%d, XY ds=%d",
             prev, cur, z, total, patch, patch, side, side, ds)
    for cy in ys:
        for cx in xs:
            attempted += 1
            if attempted == 1 or attempted % every == 0 or attempted == total:
                progress("Flexible Z=%d: patch %d/%d, center raw XY=(%.1f,%.1f), "
                         "accepted=%d, skipped coverage/texture=%d/%d, rejected=%d, elapsed=%.1fs",
                         z, attempted, total, idx.xy_min[0]+cx*ds, idx.xy_min[1]+cy*ds,
                         len(points), low_coverage, low_texture, rejected, time.perf_counter()-started)
            x0 = idx.xy_min[0] + (cx - patch // 2 - pad) * ds
            y0 = idx.xy_min[1] + (cy - patch // 2 - pad) * ds
            ref, rv = roi_mosaic(c, idx, prev, z, ds, x0, y0, side * ds, cache)
            mov, mv = roi_mosaic(c, idx, cur, z, ds, x0, y0, side * ds, cache)
            xx, yy = np.meshgrid(x0 + np.arange(side, dtype=np.float32) * ds,
                                 y0 + np.arange(side, dtype=np.float32) * ds)
            ux, uy = sample_field(previous_field, xx, yy, idx.xy_min, c.registration_mesh_px)
            mx = np.ascontiguousarray((xx - x0 + ux) / ds, dtype=np.float32)
            my = np.ascontiguousarray((yy - y0 + uy) / ds, dtype=np.float32)
            crop = (slice(pad, pad + patch), slice(pad, pad + patch))
            ref = cv2.remap(ref, mx, my, cv2.INTER_LINEAR)[crop]
            mov = cv2.remap(mov, mx, my, cv2.INTER_LINEAR)[crop]
            rv = cv2.remap(rv.astype(np.uint8), mx, my, cv2.INTER_NEAREST)[crop] > 0
            mv = cv2.remap(mv.astype(np.uint8), mx, my, cv2.INTER_NEAREST)[crop] > 0
            common = rv & mv
            if common.mean() < c.min_patch_coverage:
                low_coverage += 1
                record_patch('low_coverage')
                continue
            a, b = feature_image(ref, rv), feature_image(mov, mv)
            a[~common] = 0; b[~common] = 0
            if min(float(a.std()), float(b.std())) < 1e-4:
                low_texture += 1
                record_patch('low_texture')
                continue
            win = cv2.createHanningWindow((patch, patch), cv2.CV_32F)
            delta, score = cv2.phaseCorrelate(a, b, win)
            if (np.all(np.isfinite(delta)) and np.isfinite(score) and
                    score >= c.min_phase_response and
                    np.linalg.norm(delta) * ds <= c.max_local_shift_px):
                points.append((float(cx), float(cy)))
                shifts.append(delta); scores.append(float(score))
                record_patch('accepted',delta,score)
            else:
                rejected += 1
                record_patch('rejected_response_or_shift',delta,score)
    if not saved_previews and fallback_preview is not None:
        number,a,b,va,vb,details = fallback_preview
        save_region_preview(preview_folder/f'patch_{number:06d}'/'failed_match',a,b,va,vb,details)
    progress("Flexible Z=%d complete: tested=%d, accepted=%d, low coverage=%d, "
             "low texture=%d, rejected response/shift=%d, elapsed=%.1fs",
             z, attempted, len(points), low_coverage, low_texture, rejected, time.perf_counter()-started)
    return points, shifts, scores


def estimate_fields(c: Config, idx: Index, regdir: Path, signature: str,
                    standard_factor: Optional[np.ndarray] = None) -> np.ndarray:
    """Estimate cumulative XY sampling fields, checkpointed after EACH layer.

    Cache uses strict parameter/data signatures; no incomplete calculation is treated as final.
    """
    cache_path = regdir / "flexible_state.npz"
    gx, gy = field_grid(c, idx)
    shape = (len(idx.layer_starts), len(gy), len(gx), 2)
    layer_num = shape[0]
    fields = np.zeros(shape, np.float32)
    completed, rows = 0, []
    if cache_path.is_file() and not c.force_registration:
        with np.load(cache_path, allow_pickle=False) as state:
            if str(state["signature"].item()) != signature or state["fields"].shape != shape:
                raise RuntimeError("Existing flexible registration cache is incompatible. "
                                   "Use a new registration_dir or force_registration=True.")
            completed = int(state["completed"].item())
            fields[:] = state["fields"]
            rows = json.loads(str(state["qc_json"].item()))
        progress("Resuming flexible XY registration from layer %d / %d", completed, layer_num-1)
    elif c.force_registration and cache_path.exists():
        cache_path.unlink()
    if completed >= layer_num-1 or layer_num == 1:
        progress("Flexible registration complete/reused: %d layers, field shape=%s", layer_num, shape)
        return fields
    if standard_factor is None:
        standard_factor = load_standardization(c)
    with PageCache(min(c.cache_mb, 1024), c.cache_open_files, c.strict_input,
                   c.stack_shape, standard_factor) as cache:
        for cur in range(max(1, completed+1), layer_num):
            layer_started = time.perf_counter()
            prev = cur-1
            z0 = int(math.ceil(max(idx.layer_starts[prev], idx.layer_starts[cur])))
            z1 = int(math.floor(min(idx.layer_starts[prev]+c.stack_shape[2]-1,
                                    idx.layer_starts[cur]+c.stack_shape[2]-1)))
            progress("Flexible layer %d/%d: reference=%d, moving=%d, Z overlap=[%d,%d] (%d planes)",
                     cur, layer_num-1, prev, cur, z0, z1, max(0,z1-z0+1))
            if z1 < z0:
                if c.stop_on_registration_failure:
                    raise RuntimeError(f"Flexible layer {cur}: no Z overlap after rigid stitching. "
                                       "Inspect rigid_registration_qc.csv")
                progress("WARNING: Layer %d: no Z overlap; carrying previous field", cur)
                fields[cur] = fields[prev]
                rows.append([prev,cur,z0,z1,0,None,None,"no_overlap"])
            else:
                zs = np.unique(np.rint(np.linspace(z0,z1,c.registration_z_samples+2)[1:-1]).astype(int))
                progress("Flexible layer %d: sampling Z planes %s", cur, zs.tolist())
                points, shifts, scores = [], [], []
                for z in zs:
                    a,b,q = streaming_local_matches(c,idx,prev,cur,int(z),fields[prev],cache)
                    points.extend(a); shifts.extend(b); scores.extend(q)
                if not points:
                    if c.stop_on_registration_failure:
                        raise RuntimeError(f"Flexible layer {cur}: no reliable local XY matches. "
                                           "Reduce min_phase_response, check rigid XYZ QC, or inspect input.")
                    progress("WARNING: Flexible layer %d: no matches, carrying previous field",cur)
                    fields[cur]=fields[prev]
                    rows.append([prev,cur,z0,z1,0,None,None,"no_matches"])
                else:
                    residual = interpolate_matches(points,shifts,scores,c,idx,shape[1:3])
                    xx, yy = np.meshgrid(gx, gy)
                    ux, uy = sample_field(fields[prev], xx + residual[..., 0],
                                          yy + residual[..., 1], idx.xy_min, c.registration_mesh_px)
                    fields[cur, ..., 0] = residual[..., 0] + ux
                    fields[cur, ..., 1] = residual[..., 1] + uy
                    med = np.median(np.asarray(shifts),axis=0)*c.registration_downsample
                    rows.append([prev,cur,z0,z1,len(points),float(med[0]),float(med[1]),"ok"])
                    progress("Flexible layer %d/%d: %d matches, median xy=(%.1f,%.1f)",
                             cur,layer_num-1,len(points),*med)
            completed=cur
            atomic_savez(cache_path,signature=np.asarray(signature),completed=np.asarray(completed),
                         fields=fields,qc_json=np.asarray(json.dumps(rows)))
            write_qc(regdir / "flexible_registration_qc.csv",
                     ["prev_layer","moving_layer","z_start","z_end","matches",
                      "median_dx_raw","median_dy_raw","status"],rows)
            progress("Flexible checkpoint saved: %s, cumulative XY range=(%.2f..%.2f, %.2f..%.2f), "
                     "layer elapsed=%.1fs", cache_path, fields[cur,...,0].min(), fields[cur,...,0].max(),
                     fields[cur,...,1].min(), fields[cur,...,1].max(), time.perf_counter()-layer_started)
        return fields


def atomic_savez(path: Path, **values) -> None:
    tmp = path.with_name(path.name + ".partial")
    with tmp.open("wb") as f:
        np.savez_compressed(f, **values)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp,path)


def write_qc(path: Path, headings: list[str], rows: list) -> None:
    tmp = path.with_name(path.name + ".partial")
    with tmp.open("w",newline="") as f:
        w=csv.writer(f)
        w.writerow(headings)
        w.writerows(rows)
    os.replace(tmp,path)


def config_signature(c: Config, idx: Index, kind: str,
                     rigid_offsets: Optional[np.ndarray] = None,
                     standard_factor: Optional[np.ndarray] = None) -> str:
    """Strict deterministic source/config fingerprint to guard reuse."""
    h=hashlib.sha256()
    for dr in idx.dirs:
        h.update(dr.name.encode("utf-8")); h.update(b"\0")
    if kind == 'geometry':
        params = dict(source=str(Path(c.dir_path).resolve()), stack_shape=c.stack_shape,
                      voxel_size=c.voxel_size, flip_stage_xy=c.flip_stage_xy,
                      layer_z_tolerance_um=c.layer_z_tolerance_um)
        h.update(json.dumps(params,sort_keys=True).encode())
        return h.hexdigest()
    params = {"version": "auto-xyz-flex-v2", "kind":kind,
              "source":str(Path(c.dir_path).resolve()),"stack_shape":c.stack_shape,
              "voxel_size":c.voxel_size,"flip_stage_xy":c.flip_stage_xy,
              "layer_z_tolerance_um":c.layer_z_tolerance_um,
              "channel":c.registration_channel,"strict_input":c.strict_input,
              "stop_on_registration_failure":c.stop_on_registration_failure,
              "xy_blend_overlap_ratio":c.xy_blend_overlap_ratio,
              "rigid": {k:getattr(c,k) for k in (
                 "rigid_stitch","rigid_xy_downsample","rigid_z_downsample",
                 "rigid_roi_px","rigid_roi_count","rigid_max_xy_shift_px",
                 "rigid_signal_candidates","rigid_signal_z_samples","rigid_skip_on_failure",
                 "rigid_max_z_shift_px","rigid_min_ncc","rigid_min_overlap",
                 "rigid_max_volume_megavoxels","rigid_refine_z","rigid_refine_z_planes")}}
    params['rigid']['algorithm'] = 'adjacent-residual-pairwise-storage-v1'
    if c.if_standardization:
        if standard_factor is None:
            standard_factor = load_standardization(c)
        params["standardization"] = {
            "algorithm":"native-yx-mean-over-profile-v1",
            "gain_sha256":hashlib.sha256(standard_factor.tobytes()).hexdigest(),
        }
    if kind=="flexible":
        # Completed rigid offsets are inputs; their estimation settings are not.
        params.pop("rigid")
        params["flexible"]={k:getattr(c,k) for k in (
            "flexible_stitch","registration_downsample","registration_mesh_px",
            "registration_patch_px","registration_z_samples","min_phase_response",
            "min_patch_coverage","max_local_shift_px","field_smoothing_sigma",
            "xy_blend_overlap_ratio")}
        if rigid_offsets is not None:
            params["offsets_sha256"]=hashlib.sha256(
                np.asarray(rigid_offsets,dtype=np.float64).tobytes()).hexdigest()
        params["registration_xy_bounds"]=[idx.xy_min.tolist(),idx.xy_max.tolist()]
        params["registration_layer_starts"]=idx.layer_starts.tolist()
    h.update(json.dumps(params,sort_keys=True).encode())
    return h.hexdigest()


def roi_mosaic(c: Config, idx: Index, layer: int, z: float, ds: int,
               x0: float, y0: float, width_raw: int, cache: PageCache):
    """Spatially bounded XY montage; source filename axes XYZ, data page axes YX."""
    w=int(math.ceil(width_raw/ds)); h=w
    acc=np.zeros((h,w),np.float32)
    weight=np.zeros((h,w),np.float32)
    sx,sy,sz=c.stack_shape
    ids=candidate_ids(c,idx,layer,x0,x0+width_raw,y0,y0+width_raw,0)
    for tid in ids:
        px,py,pz=idx.positions[tid]
        iz=int(round(z-pz))
        if iz<0 or iz>=sz: continue
        im=cache.read(channel_path(c,idx,int(tid),c.registration_channel),iz,(ds,ds),False)
        if im is None: continue
        ih,iw=im.shape
        xx=int(round((px-x0)/ds)); yy=int(round((py-y0)/ds))
        a,b=max(0,xx),min(w,xx+iw)
        d,e=max(0,yy),min(h,yy+ih)
        if a>=b or d>=e: continue
        xa,xb=a-xx,b-xx; ya,yb=d-yy,e-yy
        pixels=im[ya:yb,xa:xb].astype(np.float32)
        # Weighted stitching avoids discontinuities at tile boundaries.
        xxg=np.broadcast_to(np.arange(xa,xb,dtype=np.float32)[None,:],pixels.shape)
        yyg=np.broadcast_to(np.arange(ya,yb,dtype=np.float32)[:,None],pixels.shape)
        wei=feather_weight(xxg,yyg,iw,ih,True,c.xy_blend_overlap_ratio)
        acc[d:e,a:b]+=pixels*wei
        weight[d:e,a:b]+=wei
    valid=weight>0
    np.divide(acc,weight,out=acc,where=valid)
    return acc,valid


def rigid_volume(c: Config, idx: Index, layer: int, cx: float, cy: float,
                 cache: PageCache) -> tuple[np.ndarray,np.ndarray]:
    xy=c.rigid_roi_px; dz=c.rigid_z_downsample; ds=c.rigid_xy_downsample
    n=math.ceil(c.stack_shape[2]/dz)
    wh=math.ceil(xy/ds)
    volume_mpx=n*wh*wh/1e6
    if volume_mpx>c.rigid_max_volume_megavoxels:
        raise MemoryError(f"Rigid ROI volume {volume_mpx:.1f} million voxels > "
                          f"limit {c.rigid_max_volume_megavoxels}. Increase rigid XY/Z downsampling "
                          "or decrease rigid_roi_px.")
    vol=np.zeros((n,wh,wh),np.float32)
    valid=np.zeros((n,wh,wh),bool)
    x0=cx-xy/2; y0=cy-xy/2
    for zi in range(n):
        z=float(idx.layer_starts[layer]+zi*dz)
        vol[zi],valid[zi]=roi_mosaic(c,idx,layer,z,ds,x0,y0,xy,cache)
        if zi == 0 or (zi+1) % max(1, math.ceil(n/10)) == 0 or zi == n-1:
            progress("Rigid layer %d: loaded ROI plane %d/%d, raw Z=%.1f, volume ZYX=%s",
                     layer, zi+1, n, z, vol.shape)
    return vol,valid


def shifted_common(ref, mov, ref_mask, mov_mask, shift_zyx):
    """Integer shift bringing moving sample to fixed sample; no circular wrapping."""
    slices_a=[]; slices_b=[]
    for n,s in zip(ref.shape,shift_zyx):
        s=int(round(s))
        if abs(s)>=n:
            return None,None,None
        if s>=0:
            slices_a.append(slice(s,n)); slices_b.append(slice(0,n-s))
        else:
            slices_a.append(slice(0,n+s)); slices_b.append(slice(-s,n))
    ra=ref[tuple(slices_a)]; rb=mov[tuple(slices_b)]
    valid=ref_mask[tuple(slices_a)]&mov_mask[tuple(slices_b)]
    return ra,rb,valid


def masked_ncc(ref,mov,rv,mv,shift_zyx) -> tuple[float,float]:
    a,b,v=shifted_common(ref,mov,rv,mv,shift_zyx)
    if a is None:return -1.,0.
    coverage=float(v.sum())/max(1,min(np.count_nonzero(rv),np.count_nonzero(mv)))
    if v.sum()<100:return -1.,coverage
    a=a[v].astype(np.float64); b=b[v].astype(np.float64)
    a-=a.mean(); b-=b.mean()
    # Elementwise reductions avoid spawning a BLAS thread pool per ROI/plane.
    denom=math.sqrt(float(np.sum(a*a))*float(np.sum(b*b)))
    return (float(np.sum(a*b)/denom) if denom>1e-10 else -1.),coverage


def rigid_roi_centers(c:Config,idx:Index,prev:int,cur:int) -> list[tuple[float,float]]:
    sx,sy,_=c.stack_shape
    def limits(layer):
        pts=idx.positions[idx.layer_tiles[layer],:2]
        return np.min(pts,axis=0),np.max(pts+np.array([sx,sy]),axis=0)
    a0,a1=limits(prev); b0,b1=limits(cur)
    lo=np.maximum(a0,b0); hi=np.minimum(a1,b1)
    if np.any(hi-lo<min(sx,sy)*0.5):
        raise RigidRegistrationError(f"Adjacent layers {prev},{cur} lack a common nominal XY region")
    # Favor broad spatial coverage but ensure each ROI intersects both layer bounds.
    radius=np.asarray([c.rigid_roi_px/2]*2,dtype=float)
    midpoint=(lo+hi)/2
    low=np.where(hi-lo>=2*radius,lo+radius,midpoint)
    high=np.where(hi-lo>=2*radius,hi-radius,midpoint)
    ts=np.linspace(0.25,0.75,c.rigid_roi_count) if c.rigid_roi_count>1 else [0.5]
    centers = [low + (high-low)*t for t in ts]
    # A half-thumbnail origin can round opposite tile edges in opposite
    # directions and introduce an entire downsampled-pixel translation bias.
    ds = c.rigid_xy_downsample
    return [tuple(np.rint((p-radius)/ds)*ds + radius) for p in centers]


def signal_rigid_roi_centers(c: Config, idx: Index, prev: int, cur: int,
                             cache: PageCache) -> list[tuple[float, float]]:
    """Rank bounded candidate previews by morphology present in BOTH layers.

    Sample actual tile centers as well as geometric anchors. Only shortlisted
    ROIs are read as full 3D volumes. No whole-layer mosaic is allocated.
    """
    anchors = rigid_roi_centers(c, idx, prev, cur)
    size = np.asarray(c.stack_shape[:2])
    points = np.concatenate([idx.positions[idx.layer_tiles[k], :2] + size/2
                             for k in (prev, cur)])
    bounds = [(idx.positions[idx.layer_tiles[k], :2].min(axis=0),
               (idx.positions[idx.layer_tiles[k], :2]+size).max(axis=0)) for k in (prev,cur)]
    lo = np.maximum(bounds[0][0], bounds[1][0])
    hi = np.minimum(bounds[0][1], bounds[1][1])
    points = points[np.all((points >= lo) & (points <= hi), axis=1)]
    # Keep scouting bounded even for hundreds of thousands of tiles.
    budget = c.rigid_signal_candidates
    if len(points) > budget:
        sample = np.random.default_rng(0).choice(len(points), budget, replace=False)
        points = points[np.sort(sample)]
    half = c.rigid_roi_px/2
    preview_width = min(c.rigid_roi_px, min(c.stack_shape[:2]))
    ds = c.rigid_xy_downsample
    # Tile centers are useful even when the ROI is wider than the tissue island.
    candidates = list(dict.fromkeys(tuple(np.rint((np.asarray(p)-half)/ds)*ds+half)
                                    for p in list(points)+list(anchors)))
    if len(candidates) > budget:
        # Preserve a few geometric anchors as a fallback to sampled tile centers.
        keep_anchors = min(len(anchors), max(1,budget//4))
        candidates = list(dict.fromkeys(candidates[:budget-keep_anchors]+anchors[:keep_anchors]))
    zlo = max(idx.layer_starts[prev],idx.layer_starts[cur])-c.rigid_max_z_shift_px
    zhi = min(idx.layer_starts[prev],idx.layer_starts[cur])+c.stack_shape[2]-1+c.rigid_max_z_shift_px
    planes = {}
    for layer in (prev,cur):
        a = max(idx.layer_starts[layer], zlo)
        b = min(idx.layer_starts[layer]+c.stack_shape[2]-1,zhi)
        if b < a:
            raise RigidRegistrationError("No possible Z overlap within the rigid Z search limit")
        planes[layer] = np.unique(np.rint(np.linspace(a,b,c.rigid_signal_z_samples)).astype(int))
    progress("Rigid signal scout %d->%d: %d candidate ROIs, Z samples=%s, channel=%d",
             prev,cur,len(candidates),{k:v.tolist() for k,v in planes.items()},c.registration_channel)
    ranked = []
    for number,(cx,cy) in enumerate(candidates,1):
        energies = []
        for layer in (prev,cur):
            best = 0.
            for z in planes[layer]:
                image,valid = roi_mosaic(c,idx,layer,int(z),ds,cx-preview_width/2,
                                        cy-preview_width/2,preview_width,cache)
                if np.count_nonzero(valid) < 16:
                    continue
                feature = feature_image(image,valid)
                energy = float(np.sqrt(np.mean(feature[valid]**2))) * math.sqrt(float(valid.mean()))
                best = max(best,energy)
            energies.append(best)
        score = min(energies)
        progress("Rigid signal scout %d/%d: XY=(%.1f,%.1f), texture ref/moving=%.5g/%.5g, score=%.5g",
                 number,len(candidates),cx,cy,*energies,score)
        if np.isfinite(score) and score > 1e-6:
            ranked.append((score,(cx,cy)))
    ranked.sort(key=lambda item:item[0],reverse=True)
    selected = []
    for _,center in ranked:
        if all(np.linalg.norm(np.asarray(center)-other) >= c.rigid_roi_px/2 for other in selected):
            selected.append(center)
        if len(selected) >= c.rigid_roi_count:
            break
    for _,center in ranked:
        if len(selected) >= c.rigid_roi_count:
            break
        if center not in selected:
            selected.append(center)
    if not selected:
        raise RigidRegistrationError(f"No signal-bearing rigid ROI found for layers {prev}->{cur}")
    progress("Rigid signal scout selected %d ROI centers: %s",len(selected),selected)
    return selected


def refine_rigid_z(c,idx,prev,cur,cx,cy,raw_shift_xyz,cache):
    """Optimize axial displacement at single RAW Z-voxel resolution via 2D NCC."""
    if not c.rigid_refine_z or c.rigid_z_downsample==1:
        return raw_shift_xyz
    dx,dy,dz=raw_shift_xyz
    # ref z indices vs mov z indices: qmov = qref - (nominal Z delta + correction).
    relative_z=idx.layer_starts[cur]-idx.layer_starts[prev]+dz
    q0=max(0,relative_z); q1=min(c.stack_shape[2]-1,c.stack_shape[2]-1+relative_z)
    if q1-q0<max(3,c.rigid_refine_z_planes): return raw_shift_xyz
    progress("Rigid Z refinement %d->%d: coarse dz=%.1f, search +/- %d raw planes, samples=%d",
             prev, cur, dz, c.rigid_z_downsample, c.rigid_refine_z_planes)
    xs=cx-c.rigid_roi_px/2; ys=cy-c.rigid_roi_px/2
    qs=np.rint(np.linspace(q0+1,q1-1,c.rigid_refine_z_planes)).astype(int)
    ref_frames=[]
    for q in qs:
        r,rv=roi_mosaic(c,idx,prev,idx.layer_starts[prev]+q,c.rigid_xy_downsample,
                       xs,ys,c.rigid_roi_px,cache)
        ref_frames.append((feature_image(r),rv))
    best_score=-1e9; best_z=dz
    steps=int(c.rigid_z_downsample)
    for d in range(-steps,steps+1):
        trial=dz+d
        if abs(trial) > c.rigid_max_z_shift_px:
            continue
        zlag=idx.layer_starts[cur]-idx.layer_starts[prev]+trial
        corrs=[]
        for q,(ref,rv) in zip(qs,ref_frames):
            moving_index=int(round(q-zlag))
            if not 0<=moving_index<c.stack_shape[2]:continue
            m,mv=roi_mosaic(c,idx,cur,idx.layer_starts[cur]+moving_index,
                           c.rigid_xy_downsample,xs,ys,c.rigid_roi_px,cache)
            m=feature_image(m)
            r3=ref[None,:,:]; m3=m[None,:,:]
            mask3=rv[None,:,:]; mmask3=mv[None,:,:]
            ncc,coverage=masked_ncc(r3,m3,mask3,mmask3,
                                     (0,int(round(dy/c.rigid_xy_downsample)),
                                        int(round(dx/c.rigid_xy_downsample))))
            if coverage>=c.rigid_min_overlap and np.isfinite(ncc):corrs.append(ncc)
        if len(corrs)>=max(1,c.rigid_refine_z_planes//2):
            score=float(np.median(corrs))
            progress("Rigid Z refinement: trial dz=%.1f, valid samples=%d, median NCC=%.4f",
                     trial, len(corrs), score)
            if score>best_score:best_score=score;best_z=trial
        else:
            progress("Rigid Z refinement: trial dz=%.1f rejected, valid samples=%d", trial, len(corrs))
    progress("Rigid Z refinement complete: dz=%.1f -> %.1f, best NCC=%.4f", dz, best_z, best_score)
    return np.array([dx,dy,best_z],dtype=np.float64)


def rigid_pair(c:Config,idx:Index,prev:int,cur:int,cache:PageCache):
    """Automatic XYZ translation from masked 3D correlation of several XY ROIs."""
    from skimage.registration import phase_cross_correlation
    preview_folder = registration_preview_root(c)/'rigid'/f'layer_{prev:04d}_to_{cur:04d}'/f'attempt_{time.time_ns()}'
    try:
        centers=signal_rigid_roi_centers(c,idx,prev,cur,cache)
    except RigidRegistrationError as exc:
        if c.save_registration_previews:
            try:
                anchors = rigid_roi_centers(c,idx,prev,cur)
            except RigidRegistrationError:
                anchors = []
            write_preview_json(preview_folder/'selection.json',dict(status='scout_failed',reason=str(exc),
                               roi_centers_xy_raw=[list(p) for p in anchors]))
            for number,(cx,cy) in enumerate(anchors,1):
                save_rigid_overlap_preview(c,idx,prev,cur,cx,cy,cache,
                                           preview_folder/f'roi_{number:03d}'/'before',[0,0,0],
                                           status='scout_failed',reason=str(exc))
        raise
    if c.save_registration_previews:
        write_preview_json(preview_folder/'selection.json',dict(status='signal_ranked',
                           roi_centers_xy_raw=[list(p) for p in centers]))
    estimates=[]
    nominal_z_delta=idx.layer_starts[cur]-idx.layer_starts[prev]
    for roi_number,(cx,cy) in enumerate(centers,1):
        roi_started = time.perf_counter()
        progress("Rigid %d->%d: ROI %d/%d, raw XY center=(%.1f,%.1f), nominal Z delta=%.1f",
                 prev, cur, roi_number, len(centers), cx, cy, nominal_z_delta)
        save_rigid_overlap_preview(c,idx,prev,cur,cx,cy,cache,
                                   preview_folder/f'roi_{roi_number:03d}'/'before',[0,0,0],
                                   status='before_rigid_registration')
        ref,rv=rigid_volume(c,idx,prev,cx,cy,cache)
        mov,mv=rigid_volume(c,idx,cur,cx,cy,cache)
        if min(rv.mean(),mv.mean())<c.rigid_min_overlap or \
           min(ref[rv].std() if np.any(rv) else 0,mov[mv].std() if np.any(mv) else 0)<1e-4:
            progress("WARNING: Rigid ROI at %.0f,%.0f: insufficient valid data/texture",cx,cy)
            continue
        # Log/high-pass for fluorescence contrast; masked phase correlation tolerates
        # non-overlapping parts of the 3D stack and large Z separation.
        def features(volume, mask):
            a = np.log1p(np.maximum(volume, 0))
            weight = mask.astype(np.float32)
            def blur(sigma):
                num = gaussian_filter(a * weight, (0, sigma, sigma))
                den = gaussian_filter(weight, (0, sigma, sigma))
                return np.divide(num, den, out=np.zeros_like(num), where=den > 1e-6)
            out = blur(.6) - blur(5.)
            out[~mask] = 0
            return out
        aa=features(ref,rv); bb=features(mov,mv)
        progress("Rigid ROI %d: correlating ZYX=%s, valid reference/moving=%.3f/%.3f",
                 roi_number, aa.shape, rv.mean(), mv.mean())
        try:
            shift=phase_cross_correlation(aa,bb,reference_mask=rv,moving_mask=mv,
                                           overlap_ratio=c.rigid_min_overlap)
        except ValueError as exc:
            progress("WARNING: Rigid ROI %d correlation failed: %s",roi_number,exc)
            continue
        if isinstance(shift, tuple):
            shift = shift[0]
        shift=np.asarray(shift,dtype=np.float64)
        if shift.shape != (3,) or not np.all(np.isfinite(shift)):
            progress("WARNING: Rigid ROI %d returned invalid shift: %s",roi_number,shift)
            continue
        ncc,coverage=masked_ncc(aa,bb,rv,mv,shift)
        xyz=np.array([shift[2]*c.rigid_xy_downsample,
                      shift[1]*c.rigid_xy_downsample,
                      shift[0]*c.rigid_z_downsample-nominal_z_delta],np.float64)
        progress("Rigid ROI %.0f,%.0f: correction=(%.1f,%.1f,%.1f), NCC=%.3f, overlap=%.3f",
                 cx,cy,*xyz,ncc,coverage)
        accepted = bool(np.all(np.isfinite(xyz)) and np.isfinite(ncc) and
                        abs(xyz[0])<=c.rigid_max_xy_shift_px and abs(xyz[1])<=c.rigid_max_xy_shift_px and
                        abs(xyz[2])<=c.rigid_max_z_shift_px and ncc>=c.rigid_min_ncc and
                        coverage>=c.rigid_min_overlap)
        save_rigid_overlap_preview(c,idx,prev,cur,cx,cy,cache,
                                   preview_folder/f'roi_{roi_number:03d}'/'candidate',xyz,
                                   status='accepted_candidate' if accepted else 'rejected_candidate',
                                   coarse_ncc=float(ncc) if np.isfinite(ncc) else None,
                                   coarse_overlap=float(coverage))
        if not accepted:
            progress("Rigid ROI %d rejected: limits |XY|<=%.1f, |Z|<=%.1f, NCC>=%.3f, overlap>=%.3f",
                     roi_number, c.rigid_max_xy_shift_px, c.rigid_max_z_shift_px,
                     c.rigid_min_ncc, c.rigid_min_overlap)
            progress("WARNING: Reject rigid ROI %.0f,%.0f",cx,cy)
            continue
        estimates.append((xyz,ncc,coverage,cx,cy))
        progress("Rigid ROI %d accepted, elapsed=%.1fs", roi_number, time.perf_counter()-roi_started)
    if not estimates:
        raise RigidRegistrationError(f"No acceptable automatic XYZ alignment for layers {prev}->{cur}. "
                           "Check overlap / channel / stage direction / rigid search limits.")
    xyzs=np.asarray([item[0] for item in estimates]); result=np.median(xyzs,axis=0)
    # Reject outlier ROIs by proximity to robust median.
    deviations=np.linalg.norm((xyzs-result)/np.array([c.rigid_xy_downsample,
                                                      c.rigid_xy_downsample,c.rigid_z_downsample]),axis=1)
    good=deviations<=max(3.,np.median(deviations)*3)
    if np.any(good): result=np.median(xyzs[good],axis=0)
    best=max(estimates,key=lambda q:q[1])
    if c.rigid_refine_z:
        # Reuse best-scoring XY area for full-slice axial fine search.
        result=refine_rigid_z(c,idx,prev,cur,best[3],best[4],result,cache)
    save_rigid_overlap_preview(c,idx,prev,cur,best[3],best[4],cache,
                               preview_folder/'final',result,status='final_automatic_rigid',
                               accepted_rois=len(estimates))
    return result, len(estimates),float(np.median([item[1] for item in estimates]))


def estimate_rigid(c:Config,idx:Index,regdir:Path,signature:str,
                   standard_factor: Optional[np.ndarray] = None):
    """Register adjacent layers; save/return pairwise corrections, accumulate only in memory."""
    n=len(idx.layer_starts)
    offsets=np.zeros((n,3),np.float64)
    qc=[]; completed=0
    path=regdir/'rigid_state.npz'
    if path.is_file() and not c.force_registration:
        with np.load(path,allow_pickle=False) as old:
            if ('offset_convention' not in old.files or
                    str(old['offset_convention'].item()) != RIGID_OFFSET_CONVENTION):
                raise RuntimeError("Rigid checkpoint must use adjacent-layer offsets. "
                                   "Run the rigid stage in a new registration_dir.")
            if str(old['signature'].item())!=signature or old['offsets'].shape!=offsets.shape:
                raise RuntimeError("Rigid registration cache does not match input/settings. "
                                   "Use a new registration_dir or force_registration=True.")
            offsets[:]=accumulate_rigid_offsets(old['offsets'])
            completed=int(old['completed'].item())
            qc=json.loads(str(old['qc_json'].item()))
        progress("Resume XYZ rigid registration: layer %d / %d",completed,n-1)
    if completed>=n-1 or n==1:
        progress("Rigid registration complete/reused: %d layers", n)
        return relative_rigid_offsets(offsets)
    if standard_factor is None:
        standard_factor = load_standardization(c)
    with PageCache(min(c.cache_mb,1024),c.cache_open_files,c.strict_input,
                   c.stack_shape,standard_factor) as cache:
        # Do NOT rescan hundreds of thousands of folders at each registration step.
        # Initialize each moving layer with the preceding absolute correction.
        # Correlation then measures a bounded residual, not accumulated drift.
        work=Index(idx.dirs,idx.positions.copy(),idx.layer_ids,idx.layer_starts.copy(),
                   idx.layer_tiles,idx.xy_min.copy(),idx.xy_max.copy(),
                   idx.z_min,idx.z_max,list(idx.layer_trees))
        def update_work_layer(layer, xyz):
            work.positions[work.layer_tiles[layer]] += xyz
            work.layer_starts[layer] += xyz[2]
            raw_xy=np.asarray(c.stack_shape[:2],dtype=np.float64)
            work.layer_trees[layer]=cKDTree(work.positions[work.layer_tiles[layer],:2]+raw_xy/2)
        for k in range(1,completed+1):
            update_work_layer(k,offsets[k])
        for cur in range(max(1,completed+1),n):
            layer_started = time.perf_counter()
            prev=cur-1
            baseline = offsets[prev].copy()
            update_work_layer(cur, baseline)
            progress("Rigid layer %d/%d: reference=%d Z=%.1f, moving=%d nominal Z=%.1f, "
                     "initial absolute XYZ=%s; limits apply to additional shift",
                     cur,n-1,prev,work.layer_starts[prev],cur,idx.layer_starts[cur],baseline)
            failure_reason = ""
            try:
                residual,matches,score=rigid_pair(c,work,prev,cur,cache)
                xyz = baseline + residual
                update_work_layer(cur, residual)
                progress("Rigid layer %d: adjacent residual XYZ=%s, absolute XYZ=%s",cur,residual,xyz)
            except RigidRegistrationError as exc:
                if not c.rigid_skip_on_failure and c.stop_on_registration_failure:
                    raise
                progress("WARNING: Skipping rigid alignment %d->%d: %s; using zero adjacent "
                         "correction and inheriting the preceding layer's correction",prev,cur,exc)
                failure_reason = str(exc)
                residual=np.zeros(3,np.float64);xyz=baseline.copy();matches=0;score=float('nan')
            offsets[cur]=xyz
            completed=cur
            qc.append([prev,cur,*residual.tolist(),matches,score,
                       work.layer_starts[prev],idx.layer_starts[cur],
                       "ok" if matches else "skipped_inherited",failure_reason])
            saved_relative = relative_rigid_offsets(offsets)
            saved_relative[completed+1:] = 0
            atomic_savez(path,signature=np.asarray(signature),completed=np.asarray(completed),
                         geometry_signature=np.asarray(config_signature(c,idx,'geometry')),
                         offset_convention=np.asarray(RIGID_OFFSET_CONVENTION),
                         offsets=saved_relative,qc_json=np.asarray(json.dumps(qc)))
            write_qc(regdir/'rigid_registration_qc.csv',
                     ['prev_layer','moving_layer','correction_x_raw','correction_y_raw',
                      'correction_z_raw','accepted_rois','median_ncc',
                      'ref_layer_z_start','moving_nominal_z_start','status','reason'],qc)
            progress("Rigid checkpoint: layer %d/%d adjacent offset XYZ=%s",cur,n-1,residual)
            progress("Rigid layer %d complete: accepted ROIs=%d, median NCC=%.4f, checkpoint=%s, elapsed=%.1fs",
                     cur,matches,score,path,time.perf_counter()-layer_started)
        return relative_rigid_offsets(offsets)


def candidate_ids(c: Config, idx: Index, layer: int,
                  x0: float, x1: float, y0: float, y1: float, max_shift: float) -> np.ndarray:
    """KD-tree spatial shortlist; avoids scanning all tiles for each output block."""
    center = np.array([(x0+x1)/2, (y0+y1)/2])
    sx, sy, _ = c.stack_shape
    r = max((x1-x0)/2 + sx/2, (y1-y0)/2 + sy/2) + max_shift + 2
    ids = idx.layer_trees[layer].query_ball_point(center, r, p=np.inf)
    if not ids:
        return np.array([], dtype=int)
    global_ids = idx.layer_tiles[layer][np.asarray(ids, dtype=int)]
    ori = idx.positions[global_ids]
    ok = ((ori[:, 0] < x1 + max_shift) & (ori[:, 0] + sx > x0 - max_shift) &
          (ori[:, 1] < y1 + max_shift) & (ori[:, 1] + sy > y0 - max_shift))
    return global_ids[ok]


def active_layers(c: Config, idx: Index, z: int) -> tuple[list[int], np.ndarray]:
    candidates = np.flatnonzero((idx.layer_starts <= z) &
                                (idx.layer_starts + c.stack_shape[2] > z)).tolist()
    if not candidates:
        return [], np.array([], np.float32)
    if len(candidates) == 1:
        return candidates, np.ones(1, np.float32)
    if not c.if_z_blend:
        # Choose the deepest available layer separately at every XY pixel.
        weights = [max(0.01, min(z - idx.layer_starts[i] + 1,
                   idx.layer_starts[i] + c.stack_shape[2] - z)) for i in candidates]
        return candidates, np.asarray(weights, np.float32)
    if len(candidates) == 2:
        candidates.sort(key=lambda k: idx.layer_starts[k])
        a, b = candidates
        overlap_start = max(idx.layer_starts[a], idx.layer_starts[b])
        overlap_end = min(idx.layer_starts[a], idx.layer_starts[b]) + c.stack_shape[2] - 1
        t = float(np.clip((z - overlap_start) / max(1., overlap_end - overlap_start), 0., 1.))
        # Positive endpoints retain sole-coverage pixels at irregular XY edges.
        return candidates, np.array([max(1e-6, 1-t), max(1e-6, t)], np.float32)
    # Three or more layers: relative depth weights, normalized per pixel at export.
    weights = np.array([max(0.01, min(z - idx.layer_starts[k] + 1,
                        idx.layer_starts[k] + c.stack_shape[2] - z)) for k in candidates],
                       dtype=np.float32)
    return candidates, weights / weights.sum()


def render_output_tile(c: Config, idx: Index, fields: np.ndarray, cache: PageCache,
                       z: int, channel: int, oy: int, ox: int,
                       out_h: int, out_w: int, standard_factor: Optional[np.ndarray],
                       export_xy_min: np.ndarray, field_margin_raw: float = 0.0,
                       export_xy_max: Optional[np.ndarray] = None) -> np.ndarray:
    """Assemble one output TIFF tile with bounded RAM, blending valid pixels only."""
    block = c.tiff_tile
    dx, dy, _ = c.downsample_factor if c.if_downsample else (1, 1, 1)
    xs_out = np.arange(ox, ox + block, dtype=np.float64)
    ys_out = np.arange(oy, oy + block, dtype=np.float64)
    # Pixel CENTER in RAW global coordinates; behaves correctly for non-unit ds.
    end = idx.xy_max if export_xy_max is None else export_xy_max
    startx = export_xy_min[0] + xs_out[None, :] * dx
    starty = export_xy_min[1] + ys_out[:, None] * dy
    xw = startx + (np.minimum(dx, np.maximum(1, end[0]-startx)) - 1) / 2
    yw = starty + (np.minimum(dy, np.maximum(1, end[1]-starty)) - 1) / 2
    worldx = np.broadcast_to(xw, (block, block))
    worldy = np.broadcast_to(yw, (block, block))
    in_output = (np.arange(oy, oy + block)[:, None] < out_h) & \
                (np.arange(ox, ox + block)[None, :] < out_w)
    layers, layer_weights = active_layers(c, idx, z)
    is_label = channel >= c.ch_num
    dtype = np.uint32 if is_label and c.segmentation_mode == "preserve_labels" else np.uint16
    if not layers:
        return np.zeros((block, block), dtype=dtype)
    if is_label:
        best = np.zeros((block, block), np.float32)
        result = np.zeros((block, block), dtype=dtype)
    else:
        accum = np.zeros((block, block), np.float32)
        weights = np.zeros((block, block), np.float32)
        best_z = np.zeros((block, block), np.float32)
    box = (float(worldx[0, 0]), float(worldx[0, -1]),
           float(worldy[0, 0]), float(worldy[-1, 0]))
    sx, sy, sz = c.stack_shape
    # Include a conservative search margin for corrections at the edges.
    margin = max(c.max_local_shift_px, field_margin_raw) if c.flexible_stitch else 0.
    for layer, zweight in zip(layers, layer_weights):
        if zweight <= 0:
            continue
        field = fields[layer]
        ux, uy = sample_field(field, worldx, worldy, idx.xy_min, c.registration_mesh_px)
        sampled_x, sampled_y = worldx + ux, worldy + uy
        layer_values = np.zeros((block, block), dtype=dtype if is_label else np.float32)
        layer_coverage = np.zeros((block, block), np.float32)
        for tid in candidate_ids(c, idx, layer, *box, max_shift=margin):
            px, py, pz = idx.positions[tid]
            local_z = int(round(z - pz))
            if not 0 <= local_z < sz:
                continue
            rx = sampled_x - px
            ry = sampled_y - py
            valid = ((rx >= 0) & (rx <= sx - 1) & (ry >= 0) & (ry <= sy - 1) & in_output)
            if not np.any(valid):
                continue
            im = cache.read(channel_path(c, idx, int(tid), channel), local_z,
                            (dx, dy), nearest=is_label, standard_factor=standard_factor)
            if im is None:
                continue
            # Map raw coordinates to fixed-size XY averaging blocks.
            mapx = ((rx + 0.5) / dx - 0.5).astype(np.float32)
            mapy = ((ry + 0.5) / dy - 0.5).astype(np.float32)
            if sx % dx:
                starts = np.arange(im.shape[1]) * dx
                centers = starts + (np.minimum(dx, sx-starts)-1)/2
                mapx = np.interp(rx, centers, np.arange(im.shape[1])).astype(np.float32)
            if sy % dy:
                starts = np.arange(im.shape[0]) * dy
                centers = starts + (np.minimum(dy, sy-starts)-1)/2
                mapy = np.interp(ry, centers, np.arange(im.shape[0])).astype(np.float32)
            mapx = np.ascontiguousarray(mapx); mapy = np.ascontiguousarray(mapy)
            if is_label:
                # Exact integer indexing: preserves full uint32 neuron/instance IDs.
                ix = np.rint(rx / dx).astype(np.int32)
                iy = np.rint(ry / dy).astype(np.int32)
                inside = valid & (ix >= 0) & (ix < im.shape[1]) & (iy >= 0) & (iy < im.shape[0])
                sampled = np.zeros((block, block), dtype=im.dtype)
                sampled[inside] = im[iy[inside], ix[inside]]
                valid = inside
            else:
                sampled = cv2.remap(im, mapx, mapy, cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_REPLICATE)
            w = feather_weight(rx, ry, sx, sy, True, c.xy_blend_overlap_ratio)
            w *= valid
            if is_label:
                choose = w > layer_coverage
                if c.segmentation_mode == "binary_255":
                    layer_values[choose] = np.where(sampled[choose] > 0, 255, 0).astype(dtype)
                else:
                    layer_values[choose] = sampled[choose].astype(dtype)
                layer_coverage[choose] = w[choose]
            else:
                v = sampled.astype(np.float32)
                if c.if_xy_blend:
                    layer_values += v * w
                    layer_coverage += w
                else:
                    choose = w > layer_coverage
                    layer_values[choose] = v[choose]
                    layer_coverage[choose] = w[choose]
        covered = layer_coverage > 0
        if is_label:
            choose = covered & (zweight > best)
            result[choose] = layer_values[choose]
            best[choose] = zweight
        else:
            if c.if_xy_blend:
                np.divide(layer_values, layer_coverage, out=layer_values, where=covered)
            if c.if_z_blend:
                accum[covered] += layer_values[covered] * zweight
                weights[covered] += zweight
            else:
                choose = covered & (zweight > best_z)
                accum[choose] = layer_values[choose]
                weights[choose] = 1
                best_z[choose] = zweight
    if is_label:
        return result
    good = weights > 0
    np.divide(accum, weights, out=accum, where=good)
    accum[~good] = 0
    return np.rint(np.clip(accum, 0, 65535)).astype(np.uint16)


def load_standardization(c: Config) -> Optional[np.ndarray]:
    """Read a device illumination profile and return mean(profile)/profile in YX.

    TIFF input is one grayscale slice in native detector orientation. Legacy
    .npy inputs retain the former XY convention. No dark-field subtraction is
    inferred from a single illumination image.
    """
    if not c.if_standardization:
        return None
    if not c.stand_img_path:
        raise ValueError("if_standardization=True requires stand_img_path")
    path = Path(c.stand_img_path)
    if path.suffix.lower() == '.npy':
        st = np.load(path, allow_pickle=False).T  # legacy XY array
    else:
        with tifffile.TiffFile(path) as handle:
            if len(handle.pages) != 1:
                raise ValueError("stand_img_path must contain exactly one grayscale TIFF slice")
            st = handle.pages[0].asarray()
    expected = (c.stack_shape[1], c.stack_shape[0])
    if st.ndim != 2 or st.shape != expected:
        raise ValueError(f"Illumination profile must be a YX grayscale slice of shape "
                         f"{expected}, got {st.shape}")
    st = st.astype(np.float64)
    if not np.all(np.isfinite(st)) or np.any(st < 0):
        raise ValueError("Illumination profile must contain finite nonnegative intensities")
    mean = float(st.mean())
    if not np.isfinite(mean) or mean <= 0:
        raise ValueError("Illumination profile must have positive mean intensity")
    # Retain the former 100x gain limit; zero/near-zero corners cannot define
    # reliable inverse gains and must not produce infinities.
    floor = max(mean / 100., np.finfo(np.float64).tiny)
    limited = int(np.count_nonzero(st < floor))
    if limited:
        progress("WARNING: Illumination profile: limiting correction gain to 100x at %d pixels", limited)
    factor = np.ascontiguousarray(mean / np.maximum(st, floor), dtype=np.float32)
    progress("Loaded illumination profile %s: YX=%s, correction gain=[%.4g, %.4g]",
             path, st.shape, factor.min(), factor.max())
    return factor


def write_rigid_edit_checkpoint(c: Config, idx: Index, regdir: Path,
                                signature: str, offsets: np.ndarray) -> None:
    """Publish adjacent-layer XYZ corrections without overwriting manual edits."""
    accumulate_rigid_offsets(offsets)  # validate the public pairwise representation
    state_path = regdir/"rigid_state.npz"
    if not state_path.exists() or not c.rigid_stitch or (c.force_registration and len(offsets) == 1):
        atomic_savez(state_path, signature=np.asarray(signature), offsets=offsets,
                     offset_convention=np.asarray(RIGID_OFFSET_CONVENTION),
                     geometry_signature=np.asarray(config_signature(c,idx,'geometry')),
                     completed=np.int64(len(offsets)-1), qc_json=np.asarray("[]"))
    path = regdir/"rigid_offsets.csv"
    if path.exists() and not c.force_registration:
        progress("Preserving editable rigid checkpoint: %s",path)
        return
    statuses = {}
    with np.load(state_path,allow_pickle=False) as state:
        for row in json.loads(str(state['qc_json'].item())):
            statuses[int(row[1])] = str(row[9])
    if path.exists():
        backup = path.with_name(path.name+".bak")
        backup.write_bytes(path.read_bytes())
        progress("Backed up previous editable rigid offsets: %s",backup)
    write_pairwise_rigid_csv(path, idx, offsets, statuses)
    progress("Editable rigid checkpoint saved: %s (adjacent-layer XYZ corrections in raw voxels)",path)


def write_pairwise_rigid_csv(path: Path, idx: Index, offsets: np.ndarray,
                            statuses: Optional[dict] = None, notes: Optional[dict] = None) -> None:
    statuses, notes = statuses or {}, notes or {}
    rows = [[layer,layer-1,float(idx.layer_starts[layer]),*offsets[layer].tolist(),
             RIGID_OFFSET_CONVENTION, statuses.get(layer,'reference' if layer == 0 else 'ok'),
             notes.get(layer,'')] for layer in range(len(offsets))]
    write_qc(path,['layer','prev_layer','nominal_z_raw','dx_raw','dy_raw','dz_raw',
                   'offset_convention','status','note'],rows)


def load_rigid_checkpoint(c: Config, idx: Index, regdir: Path, signature: str,
                          reuse_completed: bool = False) -> np.ndarray:
    """Return pairwise corrections; CSV overrides automatic estimates before accumulation."""
    path = regdir/"rigid_state.npz"
    if not path.is_file():
        raise RuntimeError("Missing rigid checkpoint. Run workflow_stage='rigid' first.")
    count = len(idx.layer_starts)
    with np.load(path,allow_pickle=False) as state:
        mismatch = str(state['signature'].item()) != signature
        if mismatch and not reuse_completed:
            raise RuntimeError("Rigid checkpoint does not match source/settings. Use the same rigid "
                               "configuration, or run the rigid stage in a new registration_dir.")
        if int(state['completed'].item()) != count-1:
            raise RuntimeError("Rigid checkpoint is incomplete. Resume workflow_stage='rigid' first.")
        if ('geometry_signature' not in state.files or
                str(state['geometry_signature'].item()) != config_signature(c,idx,'geometry')):
            raise RuntimeError("Rigid checkpoint source geometry does not match current input. "
                               "Check dir_path, tile folders, stack_shape, voxel_size, flip_stage_xy "
                               "and layer_z_tolerance_um against the rigid-stage configuration.")
        offsets = np.asarray(state['offsets'],dtype=np.float64).copy()
        if ('offset_convention' not in state.files or
                str(state['offset_convention'].item()) != RIGID_OFFSET_CONVENTION):
            raise ValueError('Rigid checkpoint must use adjacent-layer offsets')
    if offsets.shape != (count,3) or not np.all(np.isfinite(offsets)):
        raise ValueError("Rigid checkpoint offsets must be a finite [layers,3] XYZ array")
    csv_path = regdir/"rigid_offsets.csv"
    if csv_path.is_file():
        automatic = offsets.copy()
        seen = set()
        with csv_path.open(newline='',encoding='utf-8-sig') as stream:
            reader = csv.DictReader(stream)
            required = {'layer','prev_layer','nominal_z_raw','dx_raw','dy_raw','dz_raw','offset_convention'}
            if not required.issubset(reader.fieldnames or []):
                raise ValueError(f"{csv_path}: required columns are {sorted(required)}")
            for row in reader:
                layer = int(row['layer'])
                if layer in seen or not 0 <= layer < count:
                    raise ValueError(f"{csv_path}: duplicate or invalid layer {layer}")
                if row['offset_convention'] != RIGID_OFFSET_CONVENTION:
                    raise ValueError(f'{csv_path}: expected adjacent-layer offset_convention')
                if int(row['prev_layer']) != layer-1:
                    raise ValueError(f'{csv_path}: prev_layer must be layer-1')
                if not np.isclose(float(row['nominal_z_raw']),idx.layer_starts[layer],rtol=0,atol=1e-6):
                    raise ValueError(f"{csv_path}: nominal Z does not match layer {layer}")
                offsets[layer] = [float(row[key]) for key in ('dx_raw','dy_raw','dz_raw')]
                seen.add(layer)
        if len(seen) != count or not np.all(np.isfinite(offsets)):
            raise ValueError(f"{csv_path}: require exactly one finite XYZ offset row per layer")
        changed = np.flatnonzero(np.any(offsets != automatic,axis=1))
        for layer in changed:
            progress("Manual rigid override: layer %d, automatic XYZ=%s -> edited XYZ=%s (raw voxels)",
                     layer,automatic[layer].tolist(),offsets[layer].tolist())
        progress("Loaded editable rigid checkpoint: %s, %d manually changed layers",csv_path,len(changed))
    if np.any(offsets[0] != 0):
        raise ValueError("Rigid reference layer 0 must retain XYZ correction [0,0,0]")
    if mismatch and reuse_completed:
        progress("Reusing completed rigid offsets; rigid estimation settings/profile changes do not rerun rigid registration")
    accumulate_rigid_offsets(offsets)
    return offsets


def load_flexible_checkpoint(c: Config, idx: Index, regdir: Path, signature: str) -> np.ndarray:
    """Export must never trigger registration or reuse fields from different rigid offsets."""
    path = regdir/"flexible_state.npz"
    if not path.is_file():
        path = regdir/"flexible_displacement_fields.npz"
    if not path.is_file():
        raise RuntimeError("Missing flexible checkpoint. Run workflow_stage='flexible' first.")
    gx,gy = field_grid(c,idx)
    shape = (len(idx.layer_starts),len(gy),len(gx),2)
    with np.load(path,allow_pickle=False) as state:
        if str(state['signature'].item()) != signature:
            raise RuntimeError("Flexible checkpoint is stale or incompatible with the current rigid "
                               "offsets/settings. Run workflow_stage='flexible' with force_registration=True.")
        if 'completed' in state and int(state['completed'].item()) != shape[0]-1:
            raise RuntimeError("Flexible checkpoint is incomplete. Resume workflow_stage='flexible' first.")
        fields = state['fields'].copy()
    if fields.shape != shape or not np.all(np.isfinite(fields)):
        raise ValueError(f"Flexible checkpoint must contain finite fields of shape {shape}")
    progress("Loaded flexible checkpoint for export: %s",path)
    return fields


def save_checkpoint_overlap_previews(c: Config, idx: Index, regdir: Path,
                                     offsets: np.ndarray,
                                     standard_factor: Optional[np.ndarray] = None) -> None:
    """Accumulate and review edited pairs without rerunning registration."""
    offsets = accumulate_rigid_offsets(offsets)
    if not c.save_registration_previews:
        progress("Overlap previews disabled by save_registration_previews=False")
        return
    if len(idx.layer_starts) < 2:
        progress("No adjacent layers to preview")
        return
    pos = idx.positions + offsets[idx.layer_ids]
    view = replace(idx,positions=pos,layer_starts=idx.layer_starts+offsets[:,2],
                   layer_trees=[cKDTree(pos[ids,:2]+np.asarray(c.stack_shape[:2])/2)
                                for ids in idx.layer_tiles])
    root = registration_preview_root(c)/'checkpoint_review'/f'attempt_{time.time_ns()}'
    if standard_factor is None:
        standard_factor = load_standardization(c)
    with PageCache(c.cache_mb,c.cache_open_files,c.strict_input,
                   c.stack_shape,standard_factor) as cache:
        for cur in range(1,len(idx.layer_starts)):
            prev = cur-1
            name = f'layer_{prev:04d}_to_{cur:04d}'
            selections = sorted((registration_preview_root(c)/'rigid'/name).glob('attempt_*/selection.json'))
            centers = []
            if selections:
                saved = json.loads(selections[-1].read_text(encoding='utf-8'))
                centers = saved['roi_centers_xy_raw']
            if not centers:
                try:
                    centers = signal_rigid_roi_centers(c,view,prev,cur,cache)
                except RigidRegistrationError:
                    try:
                        centers = rigid_roi_centers(c,view,prev,cur)
                    except RigidRegistrationError:
                        centers = []
            if not centers:
                write_preview_json(root/name/'metadata.json',dict(status='no_common_xy_region',
                                   reference_layer=prev,moving_layer=cur,
                                   effective_offsets_xyz_raw=offsets[[prev,cur]].tolist()))
            for number,(cx,cy) in enumerate(centers,1):
                folder = root/name/f'roi_{number:03d}'
                details = dict(effective_reference_correction_xyz_raw=offsets[prev].tolist(),
                               effective_moving_correction_xyz_raw=offsets[cur].tolist(),
                               adjacent_correction_xyz_raw=(offsets[cur]-offsets[prev]).tolist(),
                               offset_convention=RIGID_OFFSET_CONVENTION)
                save_rigid_overlap_preview(c,view,prev,cur,cx,cy,cache,folder/'before_pair_correction',
                                           offsets[prev]-offsets[cur],
                                           status='moving_initialized_with_preceding_correction',**details)
                save_rigid_overlap_preview(c,view,prev,cur,cx,cy,cache,folder/'corrected',
                                           [0,0,0],status='edited_rigid_checkpoint_applied',**details)
    progress("Rigid checkpoint overlap review saved: %s",root)


def run_rigid(c: Config) -> None:
    run(replace(c,workflow_stage="rigid"))


def run_flexible(c: Config) -> None:
    run(replace(c,workflow_stage="flexible"))


def run_export(c: Config) -> None:
    run(replace(c,workflow_stage="export"))


def run(c: Config) -> None:
    stage = c.workflow_stage
    if stage not in ('all','rigid','flexible','export','preview'):
        raise ValueError("workflow_stage must be 'all', 'rigid', 'flexible', 'export' or 'preview'")
    do_export = stage == 'export' or (stage == 'all' and not c.registration_only)
    progress("Starting workflow stage: %s",stage)
    if any(v <= 0 for v in c.stack_shape + c.downsample_factor):
        raise ValueError("stack_shape and downsample_factor must be positive")
    if c.tiff_tile <= 0 or c.tiff_tile % 16:
        raise ValueError("tiff_tile must be a multiple of 16")
    if c.z_downsample_mode != "sample":
        raise NotImplementedError("Only Z subsampling is implemented; use 'sample'.")
    if c.segmentation_mode not in ("binary_255", "preserve_labels"):
        raise ValueError("segmentation_mode must be 'binary_255' or 'preserve_labels'")
    if c.tiff_maxworkers < 1:
        raise ValueError("tiff_maxworkers must be positive")
    if c.opencv_threads < 1:
        raise ValueError("opencv_threads must be positive")
    cv2.setNumThreads(c.opencv_threads)
    if c.registration_z_samples <= 0 or c.rigid_z_downsample <= 0 or c.rigid_xy_downsample <= 0:
        raise ValueError("registration and rigid downsampling factors must be positive")
    if min(c.rigid_signal_candidates,c.rigid_signal_z_samples) <= 0:
        raise ValueError("rigid_signal_candidates and rigid_signal_z_samples must be positive")
    if c.registration_preview_flexible_patches < 0:
        raise ValueError("registration_preview_flexible_patches must be nonnegative")
    if c.rigid_roi_px <= 0 or c.rigid_roi_count <= 0 or c.rigid_refine_z_planes <= 0:
        raise ValueError("rigid_roi_px, rigid_roi_count and rigid_refine_z_planes must be positive")
    if (len(c.voxel_size) != 3 or not np.all(np.isfinite(c.voxel_size)) or
            min(c.voxel_size) <= 0):
        raise ValueError("voxel_size must contain three finite positive XYZ values")
    if min(c.registration_downsample, c.registration_mesh_px, c.registration_patch_px,
           c.cache_open_files, c.ch_num) <= 0 or c.cache_mb < 0:
        raise ValueError("Registration sizes, channel count and open-file limit must be positive")
    if not 0 <= c.registration_channel < c.ch_num:
        raise ValueError("registration_channel must reference a raw signal channel")
    for name in ("rigid_min_overlap", "min_patch_coverage"):
        if not 0 < getattr(c, name) <= 1:
            raise ValueError(f"{name} must be in (0, 1]")
    for name in ("rigid_max_volume_megavoxels", "max_registration_megapixels"):
        if not np.isfinite(getattr(c, name)) or getattr(c, name) <= 0:
            raise ValueError(f"{name} must be finite and positive")
    for name in ("rigid_max_xy_shift_px", "rigid_max_z_shift_px", "max_local_shift_px",
                 "field_smoothing_sigma", "export_margin_xy", "xy_blend_overlap_ratio"):
        if not np.isfinite(getattr(c, name)) or getattr(c, name) < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if c.if_subregion:
        limits = np.asarray(c.subregion_range, dtype=float)
        if limits.shape != (3, 2) or not np.all(np.isfinite(limits)) or np.any(limits[:, 1] <= limits[:, 0]):
            raise ValueError("subregion_range requires finite increasing XYZ bounds")
    # Correct native detector pixels consistently in registration, review and export.
    standard_factor = load_standardization(c)
    outdir = Path(c.export_path)
    if do_export:
        outdir.mkdir(parents=True, exist_ok=True)
    regdir = Path(c.registration_dir) if c.registration_dir else Path(c.dir_path)/"_registration_xyz_flexible"
    regdir.mkdir(parents=True, exist_ok=True)
    raw_index = load_index(c)
    if c.compression and str(c.compression).lower() == "lzw" and do_export:
        try:
            import imagecodecs  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("LZW requires imagecodecs: pip install imagecodecs") from exc
    if c.registration_channel >= c.ch_num:
        raise ValueError("registration_channel must reference a raw signal channel")
    rigid_sig=config_signature(c,raw_index,'rigid',standard_factor=standard_factor)
    if stage in ('all','rigid'):
        pair_offsets=(estimate_rigid(c,raw_index,regdir,rigid_sig,standard_factor) if c.rigid_stitch
                       else np.zeros((len(raw_index.layer_starts),3),np.float64))
        write_rigid_edit_checkpoint(c,raw_index,regdir,rigid_sig,pair_offsets)
    pair_offsets=(load_rigid_checkpoint(c,raw_index,regdir,rigid_sig,
                                        reuse_completed=stage in ('flexible','export','preview')) if c.rigid_stitch
                   else np.zeros((len(raw_index.layer_starts),3),np.float64))
    # Only internal geometry uses cumulative offsets. All saved/editable offsets are pairwise.
    rigid_offsets = accumulate_rigid_offsets(pair_offsets)
    if c.rigid_stitch:
        with np.load(regdir/'rigid_state.npz',allow_pickle=False) as state:
            rigid_sig = str(state['signature'].item())
    if stage == 'preview':
        save_checkpoint_overlap_previews(c,raw_index,regdir,pair_offsets,standard_factor)
        return
    if stage != 'export':
        tmp = regdir/"rigid_offsets_raw_xyz.npy.partial"
        with tmp.open('wb') as stream:
            np.save(stream,pair_offsets)
        os.replace(tmp,regdir/"rigid_offsets_raw_xyz.npy")
    if stage == 'rigid':
        progress("Rigid stage complete. Edit %s, then run workflow_stage='%s'.",
                 regdir/"rigid_offsets.csv", 'flexible' if c.flexible_stitch else 'export')
        return
    index=corrected_index(c,raw_index,pair_offsets)
    flex_sig=config_signature(c,index,'flexible',rigid_offsets,standard_factor)
    if c.flexible_stitch:
        fields=(load_flexible_checkpoint(c,index,regdir,flex_sig) if stage == 'export'
                else estimate_fields(c,index,regdir,flex_sig,standard_factor))
    else:
        if stage == 'export':
            progress("Exporting with flexible registration disabled; using %s positions only.",
                     'saved rigid' if c.rigid_stitch else 'nominal stage')
        gx,gy=field_grid(c,index)
        fields=np.zeros((len(index.layer_starts),len(gy),len(gx),2),np.float32)
    if stage != 'export':
        # Human-readable/standalone outputs; resume uses the signature-validated state
        # files above. Keep output copies separate from the TIFF export destination.
        flexible_finished=regdir/"flexible_displacement_fields.npz"
        atomic_savez(flexible_finished,fields=fields,mesh_px=np.int64(c.registration_mesh_px),
                     xy_min=index.xy_min,xy_max=index.xy_max,layer_starts=index.layer_starts,
                     reg_ds=np.int64(c.registration_downsample),signature=np.asarray(flex_sig))
        manifest={"source":str(Path(c.dir_path).resolve()),"layers":len(index.layer_starts),
                  "rigid_signature":rigid_sig,"flexible_signature":flex_sig,
                  "rigid_stitch":c.rigid_stitch,"flexible_stitch":c.flexible_stitch,
                  "rigid_offset_units":"raw XYZ voxels, additional correction relative to preceding layer",
                  "rigid_offset_convention":RIGID_OFFSET_CONVENTION,
                  "rigid_offset_axes":"X,Y,Z",
                  "rigid_edit_checkpoint":"rigid_offsets.csv (overrides automatic rigid_state.npz)",
                  "flexible_field_axes":"control_Y,control_X,channel_XY",
                  "flexible_field_units":"raw XY pixels, output-to-source sampling displacement",
                  "rigid_offsets":pair_offsets.tolist(),
                  "normalized_origin_stage_xyz_um":index.stage_origin_um.tolist(),
                  "parameters":{key:(value if isinstance(value,(str,int,float,bool,type(None)))
                                     else list(value) if isinstance(value,tuple) else value)
                                for key,value in vars(c).items() if key not in ("dir_path","export_path")}}
        manifest_path=regdir/"registration_manifest.json"
        manifest_tmp=Path(str(manifest_path)+".partial")
        manifest_tmp.write_text(json.dumps(manifest,indent=2),encoding="utf-8")
        os.replace(manifest_tmp,manifest_path)
        atomic_savez(regdir/"tile_index.npz",
                     directories=np.asarray([p.name for p in index.dirs]),
                     corrected_positions_xyz=index.positions, layer_ids=index.layer_ids,
                     normalized_origin_stage_xyz_um=index.stage_origin_um,
                     rigid_signature=np.asarray(rigid_sig))
        progress("Registration complete. Cache directory: %s",regdir)
    if not do_export:
        progress("%s registration complete; TIFF output skipped. Next: workflow_stage='export'.",stage)
        return
    dx, dy, dz = c.downsample_factor if c.if_downsample else (1, 1, 1)

    export_xy_min = index.xy_min.copy()
    export_xy_max = index.xy_max.copy()
    if c.flexible_stitch:
        # Output-to-source fields can move content beyond the rigid tile bounds.
        export_xy_min -= np.maximum(0, fields.max(axis=(0,1,2)))
        export_xy_max -= np.minimum(0, fields.min(axis=(0,1,2)))
        export_xy_min = np.floor(export_xy_min)
        export_xy_max = np.ceil(export_xy_max)
    if c.if_subregion:
        limits = np.asarray(c.subregion_range, dtype=np.float64)
        if limits.shape != (3, 2):
            raise ValueError("subregion_range must be [[xmin,xmax],[ymin,ymax],[zmin,zmax]] in raw voxels")
        # Restrict output only; retain original field-grid origin in index.
        export_xy_min = np.maximum(export_xy_min, limits[:2, 0])
        export_xy_max = np.minimum(export_xy_max, limits[:2, 1])
        z0 = max(index.z_min, int(limits[2, 0]))
        z1 = min(index.z_max, int(limits[2, 1]))
    else:
        z0, z1 = index.z_min, index.z_max
    if z1 <= z0 or np.any(export_xy_max <= export_xy_min):
        raise ValueError("Empty export range")
    width = math.ceil((export_xy_max[0] - export_xy_min[0]) / dx)
    height = math.ceil((export_xy_max[1] - export_xy_min[1]) / dy)
    field_margin_raw = float(np.max(np.abs(fields))) if c.flexible_stitch else 0.0
    meta = {
        "source": c.dir_path,
        "input_voxel_size_xyz_um": c.voxel_size,
        "output_voxel_size_xyz_um": tuple(float(a * b) for a,b in zip(c.voxel_size, (dx,dy,dz))),
        "export_origin_xyz_raw_voxels": (float(export_xy_min[0]), float(export_xy_min[1]), int(z0)),
        "normalized_origin_stage_xyz_um": index.stage_origin_um.tolist(),
        "shape_xyz": (width, height, math.ceil((z1-z0)/dz)),
        "layers": len(index.layer_starts),
        "tile_count": len(index.dirs),
        "flexible_stitch": c.flexible_stitch,
        "z_downsample_mode": c.z_downsample_mode,
        "compression": c.compression,
        "registration_channel": c.registration_channel,
        "registration_mesh_px": c.registration_mesh_px,
        "registration_downsample": c.registration_downsample,
        "registration_dir": str(regdir),
        "rigid_stitch": c.rigid_stitch,
        "rigid_offsets_xyz_raw": pair_offsets.tolist(),
        "rigid_offset_convention": RIGID_OFFSET_CONVENTION,
        "rigid_registration_signature": rigid_sig,
        "flexible_registration_signature": flex_sig,
        "output_settings": {
            "if_downsample":c.if_downsample,
            "downsample_factor":c.downsample_factor,
            "if_subregion":c.if_subregion,
            "subregion_range":c.subregion_range if c.if_subregion else None,
            "if_xy_blend":c.if_xy_blend,
            "if_z_blend":c.if_z_blend,
            "xy_blend_overlap_ratio":c.xy_blend_overlap_ratio,
            "segmented_ch":c.segmented_ch,
            "segmentation_mode":c.segmentation_mode,
            "if_standardization":c.if_standardization,
            "stand_img_path":c.stand_img_path,
            "tiff_tile":c.tiff_tile,
            "strict_input":c.strict_input,
        },
    }
    if standard_factor is not None:
        meta["output_settings"]["standardization"] = {
            "algorithm":"native-yx-mean-over-profile-v1",
            "gain_sha256":hashlib.sha256(standard_factor.tobytes()).hexdigest(),
            "max_gain":100.0,
        }
    serial=json.dumps(meta,sort_keys=True)
    meta["export_signature"]=hashlib.sha256(serial.encode()).hexdigest()
    metadata_path=outdir/"export_metadata.json"
    if c.resume and next(outdir.glob("ch*_z*.tif"),None) is not None:
        if not metadata_path.is_file():
            raise RuntimeError("Existing TIFFs have no export_metadata.json. Use a NEW output dir "
                               "or set resume=False to regenerate all TIFF files.")
        existing=json.loads(metadata_path.read_text(encoding="utf-8"))
        if existing.get("export_signature")!=meta["export_signature"]:
            raise RuntimeError("Existing TIFFs were generated using different export or "
                               "registration settings. Use a NEW output dir or resume=False.")
    meta_tmp=Path(str(metadata_path)+".partial")
    meta_tmp.write_text(json.dumps(meta,indent=2),encoding="utf-8")
    os.replace(meta_tmp,metadata_path)
    with PageCache(c.cache_mb,c.cache_open_files,c.strict_input,c.stack_shape) as cache:
        channels = c.ch_num + len(c.segmented_ch)
        target_count = math.ceil((z1-z0)/dz) * channels
        progress("Export %d TIFFs, each %dx%d pixels, %d channels", target_count, width, height, channels)
        for z in range(z0, z1, dz):
            for ch in range(channels):
                target = outdir / f"ch{ch:02d}_z{z:06d}.tif"
                image_number = ((z-z0)//dz)*channels + ch + 1
                dtype = (np.uint32 if ch >= c.ch_num and c.segmentation_mode == "preserve_labels"
                         else np.uint16)
                image_started = time.perf_counter()
                progress("Export image %d/%d: %s, raw Z=%d, channel=%d, YX=%dx%d, dtype=%s",
                         image_number,target_count,target,z,ch,height,width,np.dtype(dtype))
                if c.resume and target.is_file():
                    try:
                        with tifffile.TiffFile(target) as existing:
                            page = existing.pages[0]
                            size = target.stat().st_size
                            valid = (len(existing.pages) == 1 and page.shape == (height, width)
                                     and page.dtype == np.dtype(dtype) and page.is_tiled
                                     and all(n > 0 and offset > 0 and offset + n <= size
                                             for offset, n in zip(page.dataoffsets, page.databytecounts)))
                        if valid:
                            progress("Skip validated %s", target.name)
                            continue
                    except (OSError, ValueError, IndexError):
                        pass
                    progress("WARNING: Rebuilding incomplete/invalid output %s", target.name)
                tmp = Path(str(target) + ".partial")
                if tmp.exists():
                    tmp.unlink()
                dtype = (np.uint32 if ch >= c.ch_num and c.segmentation_mode == "preserve_labels"
                         else np.uint16)
                def image_tiles():
                    rows = math.ceil(height/c.tiff_tile)
                    for oy in range(0, height, c.tiff_tile):
                        row = oy//c.tiff_tile + 1
                        if row == 1 or row % max(1,math.ceil(rows/10)) == 0 or row == rows:
                            progress("Export image %d/%d %s: rendering tile row %d/%d, elapsed=%.1fs",
                                     image_number,target_count,target.name,row,rows,time.perf_counter()-image_started)
                        for ox in range(0, width, c.tiff_tile):
                            yield render_output_tile(c, index, fields, cache, z, ch, oy, ox,
                                                     height, width, standard_factor, export_xy_min,
                                                     field_margin_raw, export_xy_max)
                # Tiled BigTIFF generator: no full-size Z-plane is constructed in RAM.
                with tifffile.TiffWriter(tmp, bigtiff=True) as writer:
                    writer.write(image_tiles(), shape=(height, width), dtype=dtype,
                                 tile=(c.tiff_tile, c.tiff_tile), photometric="minisblack",
                                 compression=c.compression, metadata=None,
                                 resolution=(10000/(c.voxel_size[0]*dx), 10000/(c.voxel_size[1]*dy)),
                                 resolutionunit="CENTIMETER",
                                 maxworkers=c.tiff_maxworkers)
                os.replace(tmp, target)  # atomic on one filesystem; safe resume
                progress("Export image %d/%d saved: %s, size=%.2f MiB, elapsed=%.1fs",
                         image_number,target_count,target,target.stat().st_size/1024**2,
                         time.perf_counter()-image_started)
        progress("Export complete: %s", outdir)
        cache.close()


if __name__ == "__main__":
    run(Config())
