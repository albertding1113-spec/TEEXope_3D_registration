#!/usr/bin/env python3
"""Out-of-core ND2 mosaics: prepare placement once, render independent Z planes.

Translations are shared by all channels/Z. See ND2_STITCHING.md for assumptions,
Slurm examples, disk requirements, and registration quality control.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

import nd2
import numpy as np
from scipy import signal, sparse
from scipy.sparse.linalg import lsqr
from scipy.spatial import cKDTree
import tifffile


def frame_table(f):
    table = {}
    for index, loop in enumerate(f.loop_indices):
        if set(loop) - {'P', 'Z', 'T'}:
            raise ValueError(f'Unsupported acquisition loops: {loop}')
        key = (loop.get('T', 0), loop.get('P', 0), loop.get('Z', 0))
        if key in table:
            raise ValueError(f'Duplicate frame coordinates: {key}')
        table[key] = index
    s = f.sizes
    expected = s.get('T', 1) * s.get('P', 1) * s.get('Z', 1)
    if len(table) != expected:
        raise ValueError('Incomplete acquisition: missing frames; refusing silent zero fill.')
    return table


def read_frame(f, index):
    """nd2 read_frame returns C,Y,X (singleton C can be squeezed)."""
    a = f.read_frame(index)
    shape = (f.sizes.get('C', 1), f.sizes['Y'], f.sizes['X'])
    if a.size != math.prod(shape):
        raise ValueError(f'Unexpected frame shape {a.shape}; expected {shape}')
    return a.reshape(shape)


def stage_offsets(points, camera, voxel):
    # Nikon camera matrix maps camera axes into stage axes. Invert it, rather
    # than rotating/resampling each tile into the stage coordinate system.
    delta = np.asarray(points, dtype=float)[:, :2]
    delta = delta - delta[0]
    xy = np.linalg.solve(np.asarray(camera).reshape(2, 2), delta.T).T
    xy /= np.asarray(voxel[:2])
    return xy - xy.min(axis=0)


def overlap_shift(a, b, delta_xy, radius, min_score):
    """NCC match B's inner overlap template inside A; return B correction XY."""
    dx, dy = np.rint(delta_xy).astype(int)
    h, w = a.shape
    # Only A needs a search margin; B's template merely needs to stay in B.
    # Insetting both sides of the overlap unnecessarily discards narrow seams.
    x0, x1 = max(radius, dx), min(w - radius, dx + w)
    y0, y1 = max(radius, dy), min(h - radius, dy + h)
    if min(x1 - x0, y1 - y0) < 8:
        return None
    template = b[y0-dy:y1-dy, x0-dx:x1-dx].astype(np.float64)
    search = a[y0-radius:y1+radius, x0-radius:x1+radius].astype(np.float64)
    template -= template.mean()
    energy = np.sum(template**2)
    if energy < 1e-8:
        return None
    ones = np.ones(template.shape)
    sums = signal.fftconvolve(search, ones, mode='valid')
    sums2 = signal.fftconvolve(search**2, ones, mode='valid')
    variance = np.maximum(sums2 - sums**2/template.size, 0)
    numerator = signal.fftconvolve(search, template[::-1, ::-1], mode='valid')
    score = numerator / np.sqrt(np.maximum(variance * energy, 1e-20))
    y, x = np.unravel_index(np.argmax(score), score.shape)
    peak = float(score[y, x])
    if peak < min_score or x in (0, 2*radius) or y in (0, 2*radius):
        return None  # boundary peaks mean the true shift may exceed the search
    return np.array([x-radius, y-radius], dtype=float), peak


def refine(f, table, xy, args):
    """Use a small sampled-Z MIP per tile, then a stage-regularized global solve."""
    n = len(xy)
    ds = args.downsample
    zs = sorted(set(np.linspace(0, f.sizes.get('Z', 1)-1,
                                args.reference_slices, dtype=int).tolist()))
    radius = max(1, math.ceil(args.max_shift / ds))
    images = []
    for p in range(n):
        mip = None
        for z in zs:
            a = read_frame(f, table[(0, p, z)])[args.channel, ::ds, ::ds]
            mip = a.copy() if mip is None else np.maximum(mip, a)
        images.append(mip)
        if p % 100 == 0:
            print(f'Registration reference {p+1}/{n}', flush=True)
    h, w = images[0].shape
    edges = []
    for i, j in sorted(cKDTree(xy / ds).query_pairs(math.hypot(w, h))):
        delta = (xy[j] - xy[i]) / ds
        if abs(delta[0]) >= w or abs(delta[1]) >= h:
            continue
        result = overlap_shift(images[i], images[j], delta, radius, args.min_score)
        if result is not None:
            correction, score = result
            correction = (correction + np.rint(delta) - delta) * ds
            if np.max(np.abs(correction)) <= args.max_shift:
                edges.append((i, j, correction, score))
    if not edges:
        raise ValueError('No reliable overlap matches. Try another channel/Z sampling '
                         'or omit --refine to use stage positions.')
    # Per-tile stage priors anchor disconnected components and limit drift.
    rows, cols, values, target = [], [], [], []
    for row, (i, j, correction, score) in enumerate(edges):
        weight = score
        rows.extend([row, row]); cols.extend([i, j]); values.extend([-weight, weight])
        target.append(correction * weight)
    for i in range(n):
        rows.append(len(edges)+i); cols.append(i); values.append(0.2)
        target.append(np.zeros(2))
    matrix = sparse.coo_matrix((values, (rows, cols)), shape=(len(target), n)).tocsr()
    rhs = np.asarray(target)
    shifts = np.column_stack([lsqr(matrix, rhs[:, axis], atol=1e-8, btol=1e-8)[0]
                              for axis in range(2)])
    if not np.all(np.isfinite(shifts)):
        raise ValueError('Global registration produced non-finite corrections.')
    degree = np.zeros(n, dtype=int)
    for i, j, _, _ in edges:
        degree[i] += 1; degree[j] += 1
    report = {'sampled_z': zs, 'channel': args.channel, 'downsample': ds,
              'accepted_edges': len(edges), 'unmatched_tiles': np.flatnonzero(degree == 0).tolist(),
              'max_correction_px': float(np.max(np.abs(shifts))),
              'pairwise_max_shift_px': args.max_shift,
              'global_max_shift_px': getattr(args, 'max_global_shift', None),
              'tile_corrections_xy': shifts.tolist(),
              'edges': [{'i': i, 'j': j, 'correction_xy': c.tolist(), 'ncc': s,
                         'residual_px': float(np.linalg.norm(shifts[j]-shifts[i]-c))}
                        for i, j, c, s in edges]}
    residuals = [e['residual_px'] for e in report['edges']]
    report['residual_px'] = {'median': float(np.median(residuals)),
                             'p95': float(np.percentile(residuals, 95)),
                             'max': float(np.max(residuals))}
    print(f'Registration: {len(edges)} accepted overlaps, '
          f'{len(report["unmatched_tiles"])} unmatched tiles; '
          f'max global correction {report["max_correction_px"]:.2f} px; '
          f'overlap residual p95 {report["residual_px"]["p95"]:.2f} px', flush=True)
    # Save diagnostics before applying the optional global guard so failed
    # preparation still leaves the information needed to assess the matches.
    if getattr(args, 'plan', None):
        diagnostics = Path(args.plan).with_suffix('.registration.json')
        diagnostics.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf8', dir=diagnostics.parent,
                                         delete=False) as stream:
            json.dump(report, stream, indent=2)
            pending = Path(stream.name)
        try:
            os.replace(pending, diagnostics)
        finally:
            pending.unlink(missing_ok=True)
        print(f'Registration diagnostics: {diagnostics}', flush=True)
    global_limit = getattr(args, 'max_global_shift', None)
    if global_limit is not None and report['max_correction_px'] > global_limit:
        raise ValueError(f'Global correction {report["max_correction_px"]:.2f} px exceeds '
                         f'--max-global-shift {global_limit} px. Review registration diagnostics; '
                         'the pairwise search radius --max-shift is independent.')
    if report['max_correction_px'] > args.max_shift:
        print('NOTE: Global tile corrections exceed the pairwise search radius. '
              'Corrections can accumulate across tiles; inspect residuals and pilot seams.', flush=True)
    return xy + shifts, report


def resolve_shift_defaults(args, sizes):
    tile_size = min(sizes['Y'], sizes['X'])
    if args.max_shift is None:
        args.max_shift = max(1, tile_size // 20)
    if args.max_global_shift is None:
        args.max_global_shift = max(1, tile_size // 10)


def prepare(args):
    if Path(args.plan).exists():
        raise FileExistsError(f'Plan already exists: {args.plan}; choose a new filename.')
    source = Path(args.input).resolve()
    with nd2.ND2File(source) as f:
        if f.is_rgb or f.is_legacy:
            raise ValueError('This pipeline requires a modern, non-RGB ND2 acquisition.')
        if set(f.sizes) - {'P', 'Z', 'T', 'C', 'Y', 'X'}:
            raise ValueError(f'Unsupported dimensions: {f.sizes}')
        if np.dtype(f.dtype).kind != 'u':
            raise ValueError('This pipeline currently supports unsigned integer images.')
        resolve_shift_defaults(args, f.sizes)
        if args.refine:
            print(f'Registration limits: pairwise {args.max_shift} px, '
                  f'global {args.max_global_shift} px', flush=True)
        table = frame_table(f)
        voxel = list(f.voxel_size())
        if not np.all(np.isfinite(voxel)) or min(voxel) <= 0:
            raise ValueError('Missing/invalid spatial calibration.')
        points = [p.stagePositionUm for e in f.experiment if e.type == 'XYPosLoop'
                  for p in e.parameters.points]
        if len(points) != f.sizes.get('P', 1):
            raise ValueError('Missing stage positions: cannot infer a grid safely.')
        camera = f.metadata.channels[0].volume.cameraTransformationMatrix
        if args.stage_matrix:
            camera = args.stage_matrix
        if camera is None:
            raise ValueError('Camera calibration absent; supply --stage-matrix A B C D.')
        xy = stage_offsets(points, camera, voxel)
        if not np.all(np.isfinite(xy)):
            raise ValueError('Invalid stage placement.')
        if len(xy) > 1 and cKDTree(xy).query(xy, k=2)[0][:, 1].min() < 1:
            raise ValueError('Duplicate/near-duplicate tile positions.')
        # Fail on tile-dependent Z grids; silently joining unequal depths is unsafe.
        z_start = None
        signed_step = None
        for t in range(f.sizes.get('T', 1)):
            for p in range(len(points)):
                endpoints = [f.frame_metadata(table[(t, p, z)]).channels[0].position.stagePositionUm.z
                             for z in sorted({0, f.sizes.get('Z', 1)-1})]
                if z_start is None:
                    z_start = endpoints[0]
                    signed_step = ((endpoints[-1]-endpoints[0]) /
                                   max(1, f.sizes.get('Z', 1)-1))
                expected_end = z_start + signed_step * (f.sizes.get('Z', 1)-1)
                if max(abs(endpoints[0]-z_start), abs(endpoints[-1]-expected_end)) > voxel[2]*0.1:
                    raise ValueError('Tiles/timepoints have different physical Z grids; '
                                     '3D resampling is needed before stitching.')
        report = None
        if args.refine:
            if args.channel >= f.sizes.get('C', 1):
                raise ValueError('Registration channel is out of range.')
            xy, report = refine(f, table, xy, args)
        xy -= xy.min(axis=0)
        shape = [int(math.ceil(xy[:, 1].max())) + f.sizes['Y'],
                 int(math.ceil(xy[:, 0].max())) + f.sizes['X']]
        stat = source.stat()
        plan = {'version': 1, 'input': str(source), 'input_bytes': stat.st_size,
                'input_mtime_ns': stat.st_mtime_ns, 'sizes': dict(f.sizes),
                'dtype': str(f.dtype), 'voxel_um': voxel, 'stage_matrix': list(camera),
                'stage_origin_um': list(points[0]), 'z_start_um': z_start,
                'signed_z_step_um': signed_step, 'shape_yx': shape,
                'positions_xy': xy.tolist(), 'channels': [c.channel.name for c in f.metadata.channels],
                'feather': args.feather, 'registration': report}
    path = Path(args.plan)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf8') as stream:
        json.dump(plan, stream, indent=2)
    pixels = math.prod(shape)
    c = plan['sizes'].get('C', 1)
    print(json.dumps({'plan': str(path), 'shape_yx': shape,
                      'output_GiB': pixels*c*plan['sizes'].get('Z', 1)*
                                    plan['sizes'].get('T', 1)*np.dtype(plan['dtype']).itemsize/2**30,
                      'scratch_GiB_per_worker': pixels*(c+1)*4/2**30}, indent=2))


def feather_weights(h, w, width):
    y = np.minimum(np.arange(h)+1, np.arange(h, 0, -1))
    x = np.minimum(np.arange(w)+1, np.arange(w, 0, -1))
    return (np.minimum(y[:, None], width) * np.minimum(x[None, :], width) /
            float(width**2)).astype(np.float32)


def splat(sums, weights, frame, xy, feather):
    """Bilinear deposition preserves subpixel translations and shared coverage."""
    x, y = xy
    ix, iy = math.floor(x), math.floor(y)
    fx, fy = x-ix, y-iy
    h, w = feather.shape
    for ox, wx in ((0, 1-fx), (1, fx)):
        for oy, wy in ((0, 1-fy), (1, fy)):
            if wx*wy == 0:
                continue
            region = np.s_[iy+oy:iy+oy+h, ix+ox:ix+ox+w]
            weight = feather * (wx*wy)
            weights[region] += weight
            for c in range(frame.shape[0]):
                sums[c][region] += frame[c] * weight


def render_plane(job):
    plan, output, scratch, t, z, resume, digest = job
    out = Path(output)
    if plan['sizes'].get('T', 1) > 1:
        out /= f't{t:04d}'
    out.mkdir(parents=True, exist_ok=True)
    paths = [out / f'ch{c:02d}_z{z:05d}.tif' for c in range(plan['sizes'].get('C', 1))]
    marker = out / f'z{z:05d}.done.json'
    if marker.exists() and resume:
        saved = json.loads(marker.read_text())
        if saved.get('plan_sha256') == digest and len(saved.get('bytes', [])) == len(paths) and all(p.exists() and p.stat().st_size == size
                                                      for p, size in zip(paths, saved['bytes'])):
            return f'Skip T={t} Z={z}'
    if not resume and (marker.exists() or any(p.exists() for p in paths)):
        raise FileExistsError(f'Output exists at T={t} Z={z}; use --resume for the same plan.')
    lock = out / f'z{z:05d}.lock'
    # Exclusive lock prevents two array jobs from writing the same plane.
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.close(fd)
    try:
        marker.unlink(missing_ok=True)
        h, w = plan['shape_yx']
        with tempfile.TemporaryDirectory(prefix=f'nd2-t{t}-z{z}-', dir=scratch) as tmp, ExitStack() as mappings:
            sums = np.memmap(Path(tmp)/'sums.bin', mode='w+', dtype='float32', shape=(len(paths), h, w))
            mappings.callback(sums._mmap.close)
            weights = np.memmap(Path(tmp)/'weights.bin', mode='w+', dtype='float32', shape=(h, w))
            mappings.callback(weights._mmap.close)
            # New mapped files are zero initialized by the OS.
            feather = feather_weights(plan['sizes']['Y'], plan['sizes']['X'], plan['feather'])
            with nd2.ND2File(plan['input']) as f:
                table = frame_table(f)
                for p, xy in enumerate(plan['positions_xy']):
                    splat(sums, weights, read_frame(f, table[(t, p, z)]), xy, feather)
            for c, path in enumerate(paths):
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary = path.with_suffix('.partial.tif')
                temporary.unlink(missing_ok=True)
                vx, vy, vz = plan['voxel_um']
                result = tifffile.memmap(temporary, shape=(h, w), dtype=plan['dtype'], bigtiff=True,
                    photometric='minisblack', resolution=(10000/vx, 10000/vy), resolutionunit='CENTIMETER',
                    metadata={'axes': 'YX', 'PhysicalSizeX': vx, 'PhysicalSizeY': vy,
                              'PhysicalSizeZ': vz, 'unit': 'um', 'channel': plan['channels'][c],
                              'z_index': z, 't_index': t,
                              'z_um': plan['z_start_um'] + z*plan['signed_z_step_um']})
                mappings.callback(result._mmap.close)
                for row in range(0, h, 128):
                    sl = np.s_[row:min(row+128, h), :]
                    block = np.zeros(sums[c][sl].shape, dtype=np.float32)
                    np.divide(sums[c][sl], weights[sl], out=block, where=weights[sl] > 0)
                    result[sl] = np.clip(np.rint(block), 0, np.iinfo(plan['dtype']).max).astype(plan['dtype'])
                result.flush()
                result._mmap.close()
                del result
                os.replace(temporary, path)
            # ExitStack closes mappings even on errors, before scratch cleanup.
        temp_marker = marker.with_suffix('.partial.json')
        temp_marker.write_text(json.dumps({'plan_sha256': digest, 'bytes': [p.stat().st_size for p in paths]}))
        os.replace(temp_marker, marker)
        return f'Finished T={t} Z={z}'
    finally:
        lock.unlink(missing_ok=True)


def stitch(args):
    raw = Path(args.plan).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    plan = json.loads(raw)
    if plan.get('version') != 1:
        raise ValueError('Unsupported plan version.')
    stat = Path(plan['input']).stat()
    if stat.st_size != plan['input_bytes'] or stat.st_mtime_ns != plan['input_mtime_ns']:
        raise ValueError('ND2 file changed since preparation; generate a new plan.')
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    identity = output / 'placement.json'
    # Publish a fully written manifest atomically, including concurrent Slurm jobs.
    with tempfile.NamedTemporaryFile(dir=output, prefix='.plan-', delete=False) as stream:
        stream.write(raw)
        pending = Path(stream.name)
    try:
        try:
            os.link(pending, identity)
        except FileExistsError:
            if identity.read_bytes() != raw:
                raise ValueError('Output directory belongs to a different plan.')
    finally:
        pending.unlink()
    if args.scratch:
        Path(args.scratch).mkdir(parents=True, exist_ok=True)
    nz, nt = plan['sizes'].get('Z', 1), plan['sizes'].get('T', 1)
    start, stop = args.z_range or (0, nz)
    if not 0 <= start < stop <= nz or not 0 <= args.shard_index < args.shards:
        raise ValueError('Invalid Z range or shard index.')
    jobs = [(plan, str(output), args.scratch, t, z, args.resume, digest)
            for t in range(nt) for z in range(start, stop)
            if (t*nz+z) % args.shards == args.shard_index]
    if args.workers == 1:
        for job in jobs:
            print(render_plane(job), flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for message in pool.map(render_plane, jobs, chunksize=1):
                print(message, flush=True)


def positive(value):
    n = int(value)
    if n < 1:
        raise argparse.ArgumentTypeError('must be positive')
    return n


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('prepare', help='Inspect metadata and save a reusable placement plan')
    p.add_argument('input'); p.add_argument('--plan', required=True)
    p.add_argument('--stage-matrix', type=float, nargs=4, metavar=('A', 'B', 'C', 'D'))
    p.add_argument('--feather', type=positive, default=64, help='Edge taper width in pixels')
    p.add_argument('--refine', action='store_true', help='Refine XY translations using image overlaps')
    p.add_argument('--channel', type=int, default=0, help='Zero-based registration channel')
    p.add_argument('--downsample', type=positive, default=1)
    p.add_argument('--reference-slices', type=positive, default=5)
    p.add_argument('--max-shift', type=positive,
                   help='Pairwise limit in full-resolution pixels; default: min(Y, X) // 20 (at least 1)')
    p.add_argument('--max-global-shift', type=positive,
                   help='Global correction component limit in pixels; default: min(Y, X) // 10 (at least 1)')
    p.add_argument('--min-score', type=float, default=0.6, help='Minimum overlap normalized correlation')
    p.set_defaults(func=prepare)
    p = commands.add_parser('stitch', help='Render a TIFF per channel/time/Z using a prepared plan')
    p.add_argument('--plan', required=True); p.add_argument('--output', required=True)
    p.add_argument('--scratch', help='Local SSD directory for disk-backed float buffers')
    p.add_argument('--workers', type=positive, default=1)
    p.add_argument('--z-range', type=int, nargs=2, metavar=('START', 'STOP'), help='Zero-based; STOP exclusive')
    p.add_argument('--shards', type=positive, default=1)
    p.add_argument('--shard-index', type=int, default=0)
    p.add_argument('--resume', action='store_true')
    p.set_defaults(func=stitch)
    args = parser.parse_args()
    if getattr(args, 'channel', 0) < 0 or not 0 <= getattr(args, 'min_score', 0.6) <= 1:
        parser.error('channel must be nonnegative and min-score must be between 0 and 1')
    args.func(args)


if __name__ == '__main__':
    main()
