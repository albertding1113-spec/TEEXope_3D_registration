"""Bounded-memory TIFF-series to IMS conversion. See IMARIS_CONVERSION.md."""
from __future__ import annotations

import argparse
import ctypes
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
import json
import logging
import math
import os
from pathlib import Path
import re
import sys
import time
import uuid

import numpy as np
import tifffile

LOG = logging.getLogger("imaris")
CHANNEL = re.compile(r"(?:^|[_\-.])(?:channel|ch|c)(\d+)(?=$|[_\-.])", re.I)
SUPPORTED = {"uint8", "uint16", "uint32", "float32"}


def natural_key(value):
    return tuple((1, int(s)) if s.isdigit() else (0, s.lower())
                 for s in re.split(r"(\d+)", str(value)))


def tiffs(directory):
    return sorted((p for p in directory.iterdir()
                   if p.is_file() and p.suffix.lower() in {".tif", ".tiff"}),
                  key=lambda p: natural_key(p.name))


def discover(root, pattern=None, channels=None, order="channel-major"):
    """Group files by explicit layout, channel folders, or filename labels."""
    files = tiffs(root)
    folders = [(p.name, tiffs(p)) for p in sorted(root.iterdir(), key=natural_key)
               if p.is_dir()]
    folders = [(name, paths) for name, paths in folders if paths]
    if not files:
        if not folders:
            raise ValueError(f"No TIFFs in {root} or its immediate subdirectories")
        if pattern or channels:
            raise ValueError("Channel options require TIFFs directly in the input directory")
        return folders
    if folders:
        raise ValueError("TIFFs exist at the root and in subdirectories; select one layout")
    if channels:
        if len(files) % channels:
            raise ValueError("File count is not divisible by --channels")
        n = len(files) // channels
        return [(f"Channel {c}", files[c*n:(c+1)*n] if order == "channel-major"
                 else files[c::channels]) for c in range(channels)]
    rx = re.compile(pattern) if pattern else CHANNEL
    if pattern and "c" not in rx.groupindex:
        raise ValueError("--channel-regex must contain a named group (?P<c>...)")
    matches = [rx.search(p.stem) for p in files]
    if any(matches):
        if not all(matches):
            raise ValueError("Only some filenames have channel labels; specify the layout explicitly")
        groups, keys = {}, {}
        for path, match in zip(files, matches):
            label = match.group("c") if pattern else match.group(1)
            key = path.stem[:match.start()] + "{channel}" + path.stem[match.end():]
            groups.setdefault(label, []).append(path)
            keys.setdefault(label, []).append(key)
        if len({tuple(v) for v in keys.values()}) != 1:
            raise ValueError("Channel filenames do not describe matching Z sequences")
        return [(f"Channel {k}", groups[k]) for k in sorted(groups, key=natural_key)]
    if pattern:
        raise ValueError("No filenames match --channel-regex")
    LOG.info("No channel labels found: interpreting files as one Z series")
    return [("Channel 0", files)]


@dataclass(frozen=True)
class Plane:
    path: Path
    page: int
    sample: int
    height: int
    width: int
    dtype: str
    segment_rows: int
    scratch_bytes: int
    memmap: bool


@dataclass
class Dataset:
    channels: list[list[Plane]]
    names: list[str]
    z_origin: int = 0

    @property
    def first(self):
        return self.channels[0][0]

    @property
    def shape(self):
        return (len(self.channels), len(self.channels[0]), self.first.height, self.first.width)


def inspect_dataset(groups):
    result, names = [], []
    expected = None
    for name, paths in groups:
        samples, stacks = None, []
        for path in paths:
            with tifffile.TiffFile(path) as tif:
                # Never silently flatten known time or channel axes into Z.
                if len(tif.series) != 1 or any(a in tif.series[0].axes for a in "TC"):
                    raise ValueError(f"{path}: expected one spatial series, no T/C axes; export channels separately")
                for index, page in enumerate(tif.pages):
                    if page.axes not in {"YX", "YXS", "SYX"} or page.imagedepth != 1:
                        raise ValueError(f"{path}, page {index}: unsupported axes {page.axes}")
                    if int(page.photometric) not in {0, 1, 2}:
                        raise ValueError(f"{path}: palette/YCbCr and other photometric modes are unsupported")
                    orientation = page.tags.get("Orientation")
                    if orientation and orientation.value != 1:
                        raise ValueError(f"{path}: normalize TIFF orientation before conversion")
                    dtype = np.dtype(page.dtype).name
                    if dtype not in SUPPORTED:
                        raise ValueError(f"{path}: unsupported dtype {dtype}; no implicit lossy conversion")
                    signature = (page.imagelength, page.imagewidth, dtype)
                    if expected is None:
                        expected = signature
                    if signature != expected:
                        raise ValueError(f"{path}, page {index}: {signature} differs from {expected}")
                    count = page.samplesperpixel
                    if samples is None:
                        samples, stacks = count, [[] for _ in range(count)]
                    if count != samples:
                        raise ValueError(f"{path}: inconsistent sample/channel count")
                    rows = page.tilelength if page.is_tiled else min(page.rowsperstrip, page.imagelength)
                    cols = page.tilewidth if page.is_tiled else page.imagewidth
                    contiguous = count if page.planarconfig == 1 else 1
                    decoded = rows * cols * contiguous * np.dtype(dtype).itemsize
                    scratch = max(page.databytecounts, default=0) + 3 * decoded
                    for sample in range(count):
                        stacks[sample].append(Plane(path, index, sample, *signature,
                                                   rows, scratch, page.is_memmappable))
        result.extend(stacks)
        names.extend([name] if samples == 1 else [f"{name} sample {s}" for s in range(samples)])
    if not result or not result[0]:
        raise ValueError("Empty input")
    if len({len(stack) for stack in result}) != 1:
        raise ValueError("Channels have unequal Z depths")
    return Dataset(result, names)


def read_region(plane, y0, y1):
    """Read a full-width band with at most one TIFF segment being decoded.

    Each task owns its file handle. Uncompressed contiguous pages are memory
    mapped; other pages decode only intersecting strips/tiles. No page.asarray
    fallback, full-plane allocation, or Zarr dependency.
    """
    dtype = np.dtype(plane.dtype)
    if plane.memmap:
        mapped = tifffile.memmap(plane.path, page=plane.page, mode="r")
        try:
            with tifffile.TiffFile(plane.path) as tif:
                axes = tif.pages[plane.page].axes
            region = (mapped[y0:y1, :] if axes == "YX" else
                      mapped[y0:y1, :, plane.sample] if axes == "YXS" else
                      mapped[plane.sample, y0:y1, :])
            return np.array(region, dtype=dtype, order="C", copy=True)
        finally:
            mapped._mmap.close()
    out = np.zeros((y1-y0, plane.width), dtype=dtype)
    with tifffile.TiffFile(plane.path) as tif:
        page = tif.pages[plane.page]
        rows = page.tilelength if page.is_tiled else page.rowsperstrip
        cols = page.tilewidth if page.is_tiled else page.imagewidth
        nx, ny = math.ceil(plane.width / cols), math.ceil(plane.height / rows)
        separate = page.planarconfig == 2
        offset = plane.sample * nx * ny if separate else 0
        for ty in range(y0 // rows, math.ceil(y1 / rows)):
            for tx in range(nx):
                index = offset + ty * nx + tx
                size = page.databytecounts[index]
                if not size:  # Sparse TIFF segments represent zeros.
                    continue
                tif.filehandle.seek(page.dataoffsets[index])
                raw = tif.filehandle.read(size)
                if len(raw) != size:
                    raise OSError(f"Truncated TIFF segment in {plane.path}, page {plane.page}")
                decoded, position, _ = page.decode(raw, index, jpegtables=page.jpegtables)
                if decoded is None:
                    continue
                sy, sx = position[2:4]
                bottom = min(y1, sy + decoded.shape[1])
                right = min(plane.width, sx + decoded.shape[2])
                top = max(y0, sy)
                out[top-y0:bottom-y0, sx:right] = decoded[
                    0, top-sy:bottom-sy, :right-sx, 0 if separate else plane.sample]
                del raw, decoded
    return out


def cpu_count():
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


def bounded_map(pool, function, jobs, capacity):
    """Ordered prefetch; capacity includes the current consumer's buffer."""
    pending, source = deque(), iter(jobs)
    try:
        for _ in range(capacity):
            job = next(source, None)
            if job is None:
                break
            pending.append((job, pool.submit(function, *job[2:])))
        while pending:
            job, future = pending.popleft()
            yield job, future.result()
            del future
            job = next(source, None)
            if job is not None:
                pending.append((job, pool.submit(function, *job[2:])))
    finally:
        for _, future in pending:
            future.cancel()


def make_plan(data, args):
    c, z, y, x = data.shape
    item = np.dtype(data.first.dtype).itemsize
    bx, by = args.block_xy
    budget = int(args.read_memory_gib * 1024**3)
    align = by
    for stack in data.channels:
        for plane in stack:
            if not plane.memmap:
                candidate = math.lcm(align, plane.segment_rows)
                if candidate * x * item <= budget // 4:
                    align = candidate
    rows = min(math.ceil(y / by)*by, max(by, align))
    scratch = max((p.scratch_bytes for stack in data.channels for p in stack if not p.memmap), default=0)
    job_bytes = rows * x * item + scratch
    capacity = min(args.readers * 2, (budget - bx * by * item) // job_bytes)
    if capacity < 1:
        raise ValueError(f"Reader budget too small: one band plus codec scratch needs approximately "
                         f"{(job_bytes + bx*by*item)/1024**3:.3f} GiB. Increase --read-memory-gib, "
                         "reduce --block-xy height, or retile oversized compressed TIFF strips.")
    return dict(shape_czyx=data.shape, dtype=data.first.dtype, channels=data.names,
                input_bytes=c*z*y*x*item, band_rows=rows, prefetch=capacity,
                readers=min(args.readers, capacity), writer_threads=args.writer_threads,
                estimated_reader_bytes=capacity*job_bytes + bx*by*item,
                compression=args.compression, voxel_size_um=args.voxel_size,
                input_block_xy=args.block_xy, z_range=[data.z_origin, data.z_origin + z])


def jobs_for(data, rows):
    for c, stack in enumerate(data.channels):
        for z, plane in enumerate(stack):
            for y0 in range(0, plane.height, rows):
                yield c, z, plane, y0, min(y0+rows, plane.height)


def load_writer():
    """Check the Linux loader before an expensive header scan or output creation."""
    try:
        from PyImarisWriter import PyImarisWriter as pw
    except ImportError as exc:
        raise RuntimeError("Install PyImarisWriter and its native library; see IMARIS_CONVERSION.md") from exc
    if sys.platform.startswith("linux"):
        try:
            ctypes.CDLL("libbpImarisWriter96.so")
        except OSError as exc:
            package_dir = Path(pw.__file__).parent
            raise RuntimeError(
                f"Cannot load native ImarisWriter: {exc}. Python package: {package_dir}. "
                "The pip package alone may not include the Linux .so. Install/build libbpImarisWriter96.so "
                "and set LD_LIBRARY_PATH to its directory and dependency directories BEFORE starting Python. "
                "With run_imaris.sbatch, set IMARIS_LIB_DIR to the directory containing the .so. "
                "Run ldd on the .so to identify missing dependencies. See IMARIS_CONVERSION.md."
            ) from exc
    return pw


def convert(data, args, plan, pw=None):
    if pw is None:
        pw = load_writer()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    lock = output.with_name(output.name + ".lock")
    partial = output.with_name(output.stem + f".{uuid.uuid4().hex}.partial.ims")
    with lock.open("x") as file:
        file.write(f"pid={os.getpid()}\npartial={partial}\n")
    converter = None
    try:
        options = pw.Options()
        options.mNumberOfThreads = args.writer_threads
        options.mCompressionAlgorithmType = getattr(pw, "eCompressionAlgorithmNone" if args.compression == "none"
                                                    else "eCompressionAlgorithmGzipLevel" + args.compression[4:])
        options.mEnableLogProgress = False
        options.mForceFileBlockSizeZ = False
        c, z, y, x = data.shape
        bx, by = args.block_xy

        class Progress(pw.CallbackClass):
            def __init__(self):
                self.last = time.monotonic()

            def RecordProgress(self, progress, total_bytes_written):
                now = time.monotonic()
                if now-self.last >= 30:
                    LOG.info("Writer %.1f%%; %.3f GiB written", progress*100, total_bytes_written/1024**3)
                    self.last = now

        converter = pw.ImageConverter(data.first.dtype, pw.ImageSize(x=x, y=y, z=z, c=c, t=1),
                                      pw.ImageSize(x=1, y=1, z=1, c=1, t=1),
                                      pw.DimensionSequence("x", "y", "z", "c", "t"),
                                      pw.ImageSize(x=bx, y=by, z=1, c=1, t=1),
                                      str(partial), options, "StitchedTiffToImaris", "2.0", Progress())
        block = np.zeros((by, bx), dtype=data.first.dtype)
        start = last = time.monotonic()
        copied = 0
        with ThreadPoolExecutor(max_workers=plan["readers"]) as pool:
            stream = bounded_map(pool, read_region, jobs_for(data, plan["band_rows"]), plan["prefetch"])
            try:
                for (ci, zi, plane, y0, y1), band in stream:
                    for yy in range(y0, y1, by):
                        height = min(by, y1-yy)
                        for xx in range(0, x, bx):
                            width = min(bx, x-xx)
                            block.fill(0)
                            block[:height, :width] = band[yy-y0:yy-y0+height, xx:xx+width]
                            converter.CopyBlock(block, pw.ImageSize(x=xx//bx, y=yy//by, z=zi, c=ci, t=0))
                    copied += band.nbytes
                    del band
                    now = time.monotonic()
                    if now-last >= 30:
                        LOG.info("Input %.2f%%; %.1f MiB/s", copied/plan["input_bytes"]*100,
                                 copied/(now-start)/1024**2)
                        last = now
            finally:
                stream.close()
        params = pw.Parameters()
        params.set_value("Image", "Unit", "um")
        params.set_value("Image", "ImageSizeInMB", plan["input_bytes"]/1024**2)
        colors = []
        palette = [(0, 1, 0, 1), (1, 0, 1, 1), (0, 1, 1, 1), (1, 1, 0, 1)]
        for ci, name in enumerate(data.names):
            params.set_channel_name(ci, name)
            color = pw.ColorInfo()
            color.set_base_color(pw.Color(*palette[ci % len(palette)]))
            colors.append(color)
        vx, vy, vz = args.voxel_size
        LOG.info("Finalizing pyramid, histograms, and metadata")
        converter.Finish(pw.ImageExtents(0, 0, data.z_origin*vz, x*vx, y*vy, (data.z_origin+z)*vz), params,
                         [datetime.now()], colors, True)
        converter.Destroy()
        converter = None
        # Windows rename refuses replacement; POSIX rename would overwrite.
        if os.name == "nt":
            partial.rename(output)
        else:
            # Same-filesystem atomic publication without a PB-sized second copy.
            os.link(partial, output)
            partial.unlink()
        LOG.info("Wrote %s in %.1f seconds", output, time.monotonic()-start)
    except BaseException:
        LOG.error("Conversion failed; incomplete output (if created): %s", partial)
        raise
    finally:
        try:
            if converter is not None:
                converter.Destroy()
        finally:
            lock.unlink(missing_ok=True)


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def positive_float(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return number


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", type=Path, help="TIFF directory, or root with one directory per channel")
    p.add_argument("output", type=Path, help="new .ims file; existing files are never overwritten")
    layout = p.add_mutually_exclusive_group()
    layout.add_argument("--channel-regex", help="filename regex with named group (?P<c>...)")
    layout.add_argument("--channels", type=positive_int, help="for unlabeled multi-channel file sequences")
    p.add_argument("--order", choices=["channel-major", "z-major"], default="channel-major")
    cpus = cpu_count()
    readers = min(4, max(1, cpus//4))
    p.add_argument("--readers", type=positive_int, default=readers)
    p.add_argument("--writer-threads", type=positive_int, default=max(1, cpus-readers))
    p.add_argument("--read-memory-gib", type=positive_float, default=2,
                   help="reader-buffer budget; excludes native writer/OS caches (default: 2)")
    p.add_argument("--block-xy", nargs=2, type=positive_int, default=[512, 512], metavar=("X", "Y"),
                   help="input transfer blocks; native writer chooses output 3D chunks")
    p.add_argument("--compression", choices=["none"]+[f"gzip{i}" for i in range(1, 10)], default="gzip2")
    p.add_argument("--voxel-size", nargs=3, type=positive_float, default=[1., 1., 1.], metavar=("X", "Y", "Z"),
                   help="voxel spacing in micrometres; default 1 1 1, not inferred")
    p.add_argument("--dry-run", action="store_true", help="validate all TIFF headers and print plan without writer")
    p.add_argument("--z-range", nargs=2, type=int, metavar=("START", "STOP"),
                   help="convert a zero-based half-open Z range, preserving physical Z offset")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        if not args.input.is_dir():
            raise ValueError(f"Not a directory: {args.input}")
        if args.output.suffix.lower() != ".ims":
            raise ValueError("Output filename must end in .ims")
        if args.order != "channel-major" and not args.channels:
            raise ValueError("--order requires --channels")
        pw = None if args.dry_run else load_writer()
        LOG.info("Scanning TIFF headers (all files/pages validated before conversion)")
        data = inspect_dataset(discover(args.input, args.channel_regex, args.channels, args.order))
        if args.z_range:
            start, stop = args.z_range
            if not 0 <= start < stop <= data.shape[1]:
                raise ValueError(f"--z-range must satisfy 0 <= START < STOP <= {data.shape[1]}")
            data = Dataset([stack[start:stop] for stack in data.channels], data.names, start)
        plan = make_plan(data, args)
        print(json.dumps(plan, indent=2), flush=True)
        if not args.dry_run:
            convert(data, args, plan, pw)
        return 0
    except (ValueError, OSError, RuntimeError, ImportError, re.error) as exc:
        LOG.error("%s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
