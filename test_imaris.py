from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys

import numpy as np
import pytest
import tifffile

import Imaris as im


def test_linux_native_failure_before_scan(tmp_path, monkeypatch, caplog):
    pytest.importorskip('PyImarisWriter')
    monkeypatch.setattr(im.sys, 'platform', 'linux')
    def unavailable(name):
        raise OSError('libbpImarisWriter96.so: cannot open shared object file')
    monkeypatch.setattr(im.ctypes, 'CDLL', unavailable)
    def unexpected_scan(*args):
        pytest.fail('must fail before scanning TIFFs')
    monkeypatch.setattr(im, 'discover', unexpected_scan)
    assert im.main([str(tmp_path), str(tmp_path/'out.ims')]) == 1
    assert 'LD_LIBRARY_PATH' in caplog.text
    assert 'IMARIS_LIB_DIR' in caplog.text
    assert not list(tmp_path.iterdir())


def dataset(root):
    return im.inspect_dataset(im.discover(root))


@pytest.mark.parametrize('storage', ['mapped', 'strips', 'tiles'])
@pytest.mark.parametrize('layout', ['gray', 'rgb', 'planar'])
@pytest.mark.parametrize('dtype', ['uint8', 'uint16', 'uint32', 'float32'])
def test_region_pixels(tmp_path, storage, layout, dtype):
    shape = (37, 53) if layout == 'gray' else (37, 53, 3) if layout == 'rgb' else (3, 37, 53)
    a = np.arange(np.prod(shape)).reshape(shape).astype(dtype)
    kw = dict(photometric='minisblack' if layout == 'gray' else 'rgb')
    if layout == 'planar':
        kw['planarconfig'] = 'separate'
    if storage != 'mapped':
        kw.update(compression='deflate', predictor=True)
        kw.update(tile=(16, 32)) if storage == 'tiles' else kw.update(rowsperstrip=7)
    tifffile.imwrite(tmp_path / 'z0.tif', a, **kw)
    data = dataset(tmp_path)
    for c, stack in enumerate(data.channels):
        expected = a if layout == 'gray' else a[:, :, c] if layout == 'rgb' else a[c]
        for y0, y1 in [(0, 9), (9, 32), (32, 37)]:
            actual = im.read_region(stack[0], y0, y1)
            np.testing.assert_array_equal(actual, expected[y0:y1])
            assert actual.flags.c_contiguous and actual.dtype.isnative


def test_big_endian_multipage(tmp_path):
    a = np.arange(3*19*23, dtype='uint16').reshape(3, 19, 23)
    tifffile.imwrite(tmp_path / 'stack.tif', a, byteorder='>', photometric='minisblack', metadata={'axes': 'ZYX'})
    data = dataset(tmp_path)
    assert data.shape == (1, 3, 19, 23)
    for z, p in enumerate(data.channels[0]):
        np.testing.assert_array_equal(im.read_region(p, 0, 19), a[z])


def test_discover_channels_natural_order(tmp_path):
    for c in (10, 2):
        for z in (10, 2, 1):
            tifffile.imwrite(tmp_path / f'z{z}_ch{c}.TIFF', np.zeros((5, 7), 'uint8'))
    groups = im.discover(tmp_path)
    assert [g[0] for g in groups] == ['Channel 2', 'Channel 10']
    assert [p.name for p in groups[0][1]] == ['z1_ch2.TIFF', 'z2_ch2.TIFF', 'z10_ch2.TIFF']
    assert dataset(tmp_path).shape == (2, 3, 5, 7)
    (tmp_path / 'z2_ch2.TIFF').unlink()
    with pytest.raises(ValueError, match='matching Z'):
        im.discover(tmp_path)


def test_validation_and_budget(tmp_path):
    tifffile.imwrite(tmp_path / 'z0.tif', np.zeros((17, 33), 'uint16'))
    data = dataset(tmp_path)
    args = im.parser().parse_args([str(tmp_path), str(tmp_path/'out.ims'), '--read-memory-gib', '0.000001'])
    with pytest.raises(ValueError, match='budget too small'):
        im.make_plan(data, args)
    tifffile.imwrite(tmp_path / 'z1.tif', np.zeros((18, 33), 'uint16'))
    with pytest.raises(ValueError, match='differs'):
        dataset(tmp_path)


def test_bounded_prefetch():
    submitted = []
    def source():
        for i in range(10):
            submitted.append(i)
            yield 0, 0, i
    with ThreadPoolExecutor(2) as pool:
        stream = im.bounded_map(pool, lambda x: x, source(), 3)
        assert next(stream)[1] == 0
        assert len(submitted) == 3
        assert next(stream)[1] == 1
        assert len(submitted) == 4
        stream.close()


def test_folders_and_manual_order(tmp_path):
    for channel in ('green', 'red'):
        folder = tmp_path/channel
        folder.mkdir()
        for z in range(2):
            tifffile.imwrite(folder/f'z{z}.tif', np.zeros((8, 8), 'uint8'))
    assert dataset(tmp_path).shape == (2, 2, 8, 8)
    root = tmp_path/'green'
    assert len(im.discover(root, channels=2, order='z-major')) == 2
    with pytest.raises(ValueError, match='divisible'):
        im.discover(root, channels=3)


def test_cli_dry_run_and_invalid_range(tmp_path, capsys):
    tifffile.imwrite(tmp_path/'z0.tif', np.zeros((8, 8), 'uint8'))
    argv = [str(tmp_path), str(tmp_path/'out.ims'), '--dry-run']
    assert im.main(argv + ['--z-range', '0', '1']) == 0
    assert 'shape_czyx' in capsys.readouterr().out
    assert not (tmp_path/'out.ims').exists()
    assert im.main(argv + ['--z-range', '0', '2']) == 1
    assert im.main(argv + ['--channel-regex', '[']) == 1


def test_native_pyramid(tmp_path):
    pytest.importorskip('PyImarisWriter')
    subprocess.run([sys.executable, '-c',
                    'from pathlib import Path; from test_imaris import native_pyramid; '
                    'import sys; native_pyramid(Path(sys.argv[1]))', str(tmp_path)],
                   check=True, timeout=60)


def native_pyramid(tmp_path):
    import h5py
    a = np.arange(33*257*259, dtype='uint16').reshape(33, 257, 259)
    tifffile.imwrite(tmp_path/'stack.tif', a, photometric='minisblack',
                     compression='deflate', tile=(32, 32), metadata={'axes': 'ZYX'})
    assert im.main([str(tmp_path), str(tmp_path/'out.ims'), '--writer-threads', '2',
                    '--readers', '2', '--block-xy', '64', '32']) == 0
    with h5py.File(tmp_path/'out.ims') as f:
        assert len(f['DataSet']) > 1
        for level in f['DataSet'].values():
            channel = level['TimePoint 0/Channel 0']
            assert 'Histogram' in channel
            assert channel['Data'].chunks is not None
        np.testing.assert_array_equal(f['DataSet/ResolutionLevel 0/TimePoint 0/Channel 0/Data'][:33, :257, :259], a)


def test_writer_failure_cleanup(tmp_path):
    tifffile.imwrite(tmp_path/'z0.tif', np.zeros((7, 9), 'uint16'))
    data = dataset(tmp_path)
    args = im.parser().parse_args([str(tmp_path), str(tmp_path/'out.ims')])
    destroyed = []
    class Converter:
        def __init__(self, *args):
            Path(args[5]).write_bytes(b'partial')
        def CopyBlock(self, *args):
            raise RuntimeError('injected write failure')
        def Destroy(self):
            destroyed.append(True)
    pw = SimpleNamespace(Options=SimpleNamespace, CallbackClass=object,
                         eCompressionAlgorithmGzipLevel2=2, ImageSize=SimpleNamespace,
                         DimensionSequence=lambda *a: a, ImageConverter=Converter)
    with pytest.raises(RuntimeError, match='injected'):
        im.convert(data, args, im.make_plan(data, args), pw)
    assert destroyed == [True]
    assert not args.output.exists()
    assert not Path(str(args.output)+'.lock').exists()
    assert len(list(tmp_path.glob('*.partial.ims'))) == 1


@pytest.mark.parametrize('dtype', ['uint8', 'uint16', 'uint32', 'float32'])
@pytest.mark.parametrize('compression', ['none', 'gzip2'])
def test_native_roundtrip(tmp_path, dtype, compression):
    pytest.importorskip('PyImarisWriter')
    # The vendor DLL can hang when multiple converters are created sequentially
    # in one process. Exercise the supported one-conversion-per-CLI-process model.
    subprocess.run([sys.executable, '-c',
                    'from pathlib import Path; from test_imaris import native_roundtrip; '
                    'import sys; native_roundtrip(Path(sys.argv[1]), sys.argv[2], sys.argv[3])',
                    str(tmp_path), dtype, compression], check=True, timeout=60)


def native_roundtrip(tmp_path, dtype, compression):
    h5py = pytest.importorskip('h5py')
    a = np.arange(2*3*37*53).reshape(2, 3, 37, 53).astype(dtype)
    for c in range(2):
        for z in range(3):
            tifffile.imwrite(tmp_path / f'z{z}_ch{c}.tif', a[c, z], compression='deflate', rowsperstrip=7)
    args = im.parser().parse_args([str(tmp_path), str(tmp_path/'out.ims'), '--block-xy', '16', '16',
                                  '--readers', '2', '--writer-threads', '2', '--compression', compression,
                                  '--voxel-size', '0.5', '0.75', '2'])
    data = dataset(tmp_path)
    data = im.Dataset([stack[1:] for stack in data.channels], data.names, 1)
    im.convert(data, args, im.make_plan(data, args))
    with h5py.File(args.output) as f:
        for c in range(2):
            raw = f[f'DataSet/ResolutionLevel 0/TimePoint 0/Channel {c}/Data']
            np.testing.assert_array_equal(raw[:2, :37, :53], a[c, 1:])
            assert raw.dtype == np.dtype(dtype)
            assert raw.compression == (None if compression == 'none' else 'gzip')
        attrs = f['DataSetInfo/Image'].attrs
        assert float(attrs['ExtMax0'].tobytes()) == 53*0.5
        assert float(attrs['ExtMax2'].tobytes()) == 3*2
        assert float(attrs['ExtMin2'].tobytes()) == 2
    assert not list(tmp_path.glob('*.partial.ims'))
    with pytest.raises(FileExistsError):
        im.convert(data, args, im.make_plan(data, args))
