import json
from types import SimpleNamespace

import numpy as np
import pytest
import tifffile

import stitch_nd2 as s


def test_camera_rotation():
    theta = 0.2
    camera = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    expected = np.array([[0, 0], [90, 0], [0, 90]])
    points = (expected * 0.6) @ camera.T + [42, -300]
    np.testing.assert_allclose(s.stage_offsets(points, camera, [0.6, 0.6, 1]), expected, atol=1e-12)


@pytest.mark.parametrize('actual', [(70, 3), (3, 70), (-70, 3), (3, -70)])
def test_overlap_shift_sign(actual):
    rng = np.random.default_rng(123)
    scene = rng.normal(size=(300, 300))
    dx, dy = actual
    a = scene[100:200, 100:200]
    b = scene[100+dy:200+dy, 100+dx:200+dx]
    predicted = np.array(actual) + [2, -1]
    correction, score = s.overlap_shift(a, b, predicted, 5, 0.8)
    np.testing.assert_equal(correction, [-2, 1])
    assert score > 0.99


def test_blank_registration_rejected():
    assert s.overlap_shift(np.zeros((100, 100)), np.zeros((100, 100)), [70, 0], 5, 0.6) is None


@pytest.mark.parametrize('shape,expected', [((512, 512), (25, 51)),
                                           ((2048, 2048), (102, 204)),
                                           ((512, 1024), (25, 51)),
                                           ((8, 8), (1, 1))])
def test_size_dependent_shift_defaults(shape, expected):
    args = SimpleNamespace(max_shift=None, max_global_shift=None)
    s.resolve_shift_defaults(args, dict(zip(('Y', 'X'), shape)))
    assert (args.max_shift, args.max_global_shift) == expected


def test_explicit_shift_limits_override_defaults():
    args = SimpleNamespace(max_shift=12, max_global_shift=80)
    s.resolve_shift_defaults(args, {'Y': 512, 'X': 512})
    assert (args.max_shift, args.max_global_shift) == (12, 80)


def test_narrow_overlap_with_size_dependent_search_radius():
    scene = np.random.default_rng(4).normal(size=(512, 1024))
    correction, score = s.overlap_shift(scene[:, :512], scene[:, 456:968],
                                       [461, 0], 25, 0.8)
    np.testing.assert_equal(correction, [-5, 0])
    assert score > 0.99


def test_global_refinement_reduces_stage_error():
    rng = np.random.default_rng(8)
    scene = rng.integers(1, 1000, (100, 260), dtype=np.uint16)
    class Reader:
        sizes = {'P': 3, 'Y': 100, 'X': 100, 'C': 1, 'Z': 1}
        def read_frame(self, index):
            return scene[:, index*70:index*70+100]
    xy = np.array([[0., 0.], [72., -1.], [138., 1.]])
    args = SimpleNamespace(downsample=1, reference_slices=1, max_shift=6,
                           channel=0, min_score=0.8)
    result, report = s.refine(Reader(), {(0, i, 0): i for i in range(3)}, xy, args)
    assert report['accepted_edges'] == 2
    assert report['unmatched_tiles'] == []
    np.testing.assert_allclose(np.diff(result, axis=0), [[70, 0], [70, 0]], atol=0.15)


def test_subpixel_blend_and_holes():
    sums = np.zeros((2, 5, 12), dtype=np.float32)
    weights = np.zeros((5, 12), dtype=np.float32)
    frame = np.stack([np.full((4, 4), 100), np.full((4, 4), 700)])
    feather = s.feather_weights(4, 4, 2)
    s.splat(sums, weights, frame, [0.5, 0.5], feather)
    s.splat(sums, weights, frame, [3.5, 0.5], feather)
    result = np.divide(sums, weights[None], out=np.zeros_like(sums), where=weights[None] > 0)
    np.testing.assert_allclose(result[0][weights > 0], 100)
    np.testing.assert_allclose(result[1][weights > 0], 700)
    assert np.all(result[:, :, -2:] == 0)


@pytest.mark.parametrize('global_limit', [None, 6])
def test_accumulated_correction_has_independent_limit(tmp_path, monkeypatch, global_limit):
    class Reader:
        sizes = {'P': 20, 'Y': 100, 'X': 100, 'C': 1, 'Z': 1}
        def read_frame(self, index):
            return np.zeros((100, 100), dtype=np.uint16)
    # Consistent +5 px neighbor corrections accumulate along a long tile row.
    monkeypatch.setattr(s, 'overlap_shift', lambda *args: (np.array([5., 0.]), 1.0))
    xy = np.column_stack([np.arange(20)*70., np.zeros(20)])
    args = SimpleNamespace(downsample=1, reference_slices=1, max_shift=6,
                           max_global_shift=global_limit, channel=0, min_score=0.8,
                           plan=str(tmp_path/'layout.json'))
    table = {(0, i, 0): i for i in range(20)}
    if global_limit is None:
        result, report = s.refine(Reader(), table, xy, args)
        assert report['max_correction_px'] > args.max_shift
        assert np.all(np.diff(result[:, 0]) > 0)
    else:
        with pytest.raises(ValueError, match='--max-global-shift'):
            s.refine(Reader(), table, xy, args)
    diagnostics = json.loads((tmp_path/'layout.registration.json').read_text())
    assert diagnostics['max_correction_px'] > args.max_shift
    assert diagnostics['accepted_edges'] == 19
    assert len(diagnostics['tile_corrections_xy']) == 20
    assert diagnostics['residual_px']['p95'] >= 0


def test_frame_mapping_arbitrary_loop_order():
    f = SimpleNamespace(sizes={'P': 2, 'Z': 2}, loop_indices=[
        {'Z': 0, 'P': 0}, {'Z': 0, 'P': 1}, {'Z': 1, 'P': 0}, {'Z': 1, 'P': 1}])
    assert s.frame_table(f)[(0, 1, 0)] == 1
    f.loop_indices = f.loop_indices[:-1]
    with pytest.raises(ValueError, match='Incomplete'):
        s.frame_table(f)


def test_render_two_channels_and_resume(tmp_path, monkeypatch):
    class FakeND2:
        sizes = {'P': 2, 'Z': 1, 'C': 2, 'Y': 8, 'X': 8}
        loop_indices = [{'P': 0}, {'P': 1}]
        def __init__(self, path): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read_frame(self, index):
            return np.stack([np.full((8, 8), 100+index*100, dtype='uint16'),
                             np.full((8, 8), 500+index*100, dtype='uint16')])
    monkeypatch.setattr(s.nd2, 'ND2File', FakeND2)
    plan = {'input': 'fake.nd2', 'sizes': FakeND2.sizes, 'shape_yx': [8, 12],
            'positions_xy': [[0, 0], [4, 0]], 'feather': 4, 'dtype': 'uint16',
            'voxel_um': [0.6, 0.6, 1], 'channels': ['Green', 'Red'],
            'z_start_um': 10, 'signed_z_step_um': 1}
    job = (plan, str(tmp_path/'output'), str(tmp_path), 0, 0, False, 'abc')
    assert 'Finished' in s.render_plane(job)
    a = tifffile.imread(tmp_path/'output/ch00_z00000.tif')
    b = tifffile.imread(tmp_path/'output/ch01_z00000.tif')
    np.testing.assert_equal(b.astype(int)-a, 400)
    assert np.all(a[:, :4] == 100) and np.all(a[:, -4:] == 200)
    assert np.all(np.diff(a[3].astype(int)) >= 0)
    assert 'Skip' in s.render_plane((*job[:5], True, 'abc'))
    with pytest.raises(FileExistsError):
        s.render_plane(job)
    # A damaged/partial output is rebuilt on resume.
    (tmp_path/'output/ch00_z00000.tif').write_bytes(b'broken')
    assert 'Finished' in s.render_plane((*job[:5], True, 'abc'))
    marker = json.loads((tmp_path/'output/z00000.done.json').read_text())
    assert len(marker['bytes']) == 2
