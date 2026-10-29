"""Small real-TIFF integration tests; no acquisition data or cluster is needed."""
import tempfile
import csv
import json
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import tifffile
from scipy.ndimage import gaussian_filter

import auto_xyz_flexible_export as stitch


class StitchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='stitch-test-', dir=Path(__file__).resolve().parent)
        self.root = Path(self.temp.name)

    def tearDown(self):
        assert self.root.resolve().parent == Path(__file__).resolve().parent
        self.temp.cleanup()

    def config(self, **kwargs):
        values = dict(dir_path=str(self.root / 'input'), export_path=str(self.root / 'out'),
                      registration_dir=str(self.root / 'registration'), ch_num=1,
                      stack_shape=(128, 128, 40), voxel_size=(1., 1., 1.),
                      rigid_xy_downsample=2, rigid_z_downsample=2,
                      rigid_roi_px=128, rigid_roi_count=1,
                      rigid_max_xy_shift_px=12, rigid_max_z_shift_px=6,
                      rigid_min_ncc=.2, rigid_min_overlap=.2,
                      registration_downsample=1, registration_mesh_px=32,
                      registration_patch_px=64, min_patch_coverage=.5,
                      max_local_shift_px=8, field_smoothing_sigma=.5,
                      tiff_tile=32, compression=None, cache_mb=8,
                      if_downsample=False, flexible_stitch=False)
        values.update(kwargs)
        return stitch.Config(**values)

    def tile(self, name, data, channel='ch00.ome.tif'):
        dest = self.root / 'input' / name / '0'
        dest.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(dest / channel, data, photometric='minisblack')

    def test_parallel_export_matches_serial_and_repairs_resume(self):
        data = np.arange(4*35*37, dtype=np.uint16).reshape(4,35,37)
        name = 'X1000_Y2000_Z3000'
        self.tile(name, data)
        self.tile(name, data+300, 'ch01.ome.tif')
        labels = np.full(data.shape, 4000000001, np.uint32)
        dest = self.root/'input'/name/'segmentation'/'instances'
        dest.mkdir(parents=True)
        tifffile.imwrite(dest/'instances.tif', labels, photometric='minisblack')
        profile = self.root/'profile.tif'
        tifffile.imwrite(profile, np.linspace(1,2,35*37,dtype=np.float32).reshape(35,37))
        c = self.config(workflow_stage='export', stack_shape=(37,35,4), ch_num=2,
                        rigid_stitch=False, segmented_ch=('instances',),
                        segmentation_mode='preserve_labels', compression='lzw',
                        if_standardization=True, stand_img_path=str(profile),
                        if_downsample=True, downsample_factor=(2,3,1))
        stitch.run(c)
        parallel = replace(c, export_path=str(self.root/'parallel'), export_workers=2)
        stitch.run(parallel)
        outputs = sorted((self.root/'out').glob('*.tif'))
        self.assertEqual(len(outputs), 12)
        for path in outputs:
            np.testing.assert_array_equal(tifffile.imread(path),
                                          tifffile.imread(self.root/'parallel'/path.name))
        self.assertEqual(json.loads((self.root/'out'/'export_metadata.json').read_text()),
                         json.loads((self.root/'parallel'/'export_metadata.json').read_text()))
        intact = self.root/'parallel'/outputs[0].name
        mtime = intact.stat().st_mtime_ns
        broken = self.root/'parallel'/outputs[-1].name
        broken.write_bytes(broken.read_bytes()[:32])
        # Worker counts and compression thread counts do not invalidate resume.
        stitch.run(replace(parallel, export_workers=3, tiff_maxworkers=1))
        self.assertEqual(intact.stat().st_mtime_ns, mtime)
        np.testing.assert_array_equal(tifffile.imread(broken), tifffile.imread(outputs[-1]))
        self.assertFalse(list((self.root/'parallel').glob('*.partial')))

    def test_parallel_export_propagates_input_failure(self):
        self.tile('X1000_Y2000_Z3000', np.ones((2,8,8), np.uint16))
        c = self.config(workflow_stage='export', stack_shape=(8,8,2),
                        rigid_stitch=False, ch_num=2, export_workers=2)
        with self.assertRaisesRegex(OSError, 'Cannot read'):
            stitch.run(c)  # The second channel is deliberately missing.
        self.assertFalse((self.root/'out'/'ch01_z000000.tif').exists())
        with self.assertRaisesRegex(ValueError, 'export_workers'):
            stitch.run(replace(c, export_workers=0))

    def test_nearby_z_coordinates_share_layer_and_source_pages(self):
        # Same XY and slightly different folder Z: retain both source stacks.
        for z in ('05378.000', '05378.196', '05378.900', '05379.000', '05379.800'):
            data = np.broadcast_to(np.arange(8, dtype=np.uint16)[:, None, None], (8, 4, 4))
            self.tile(f'X154599.900_Y54873.200_Z{z}', data)
        c = self.config(stack_shape=(4, 4, 8), voxel_size=(.1489, .1489, .1))
        idx = stitch.load_index(c)
        np.testing.assert_array_equal(idx.layer_ids, [0, 0, 0, 1, 1])
        np.testing.assert_array_equal(idx.positions[:, 2], [0, 0, 0, 10, 10])
        np.testing.assert_array_equal(idx.layer_starts, [0, 10])
        self.assertEqual([len(ids) for ids in idx.layer_tiles], [3, 2])
        # Every stack in a grouped layer contributes the same local Z page.
        with stitch.PageCache(8) as cache:
            image, valid = stitch.layer_mosaic(c, idx, 0, 3, 1, cache)
        np.testing.assert_array_equal(image[valid], 3)
        offsets = np.array([[0, 0, 0], [2, 3, -1]])
        shifted = stitch.load_index(c, offsets, normalize=True)
        corrected = stitch.corrected_index(c, idx, offsets)
        np.testing.assert_array_equal(shifted.positions, corrected.positions)
        np.testing.assert_array_equal(shifted.layer_starts, [0, 9])
        # Zero restores legacy voxel grouping; changed grouping invalidates caches.
        legacy = replace(c, layer_z_tolerance_um=0)
        self.assertEqual(len(stitch.load_index(legacy).layer_starts), 5)
        for kind in ('geometry', 'rigid', 'flexible'):
            self.assertNotEqual(stitch.config_signature(c, idx, kind),
                                stitch.config_signature(legacy, idx, kind))
        for tolerance in (-1, float('nan'), float('inf')):
            with self.assertRaisesRegex(ValueError, 'layer_z_tolerance_um'):
                stitch.load_index(replace(c, layer_z_tolerance_um=tolerance))

    def scene(self):
        rng = np.random.default_rng(123)
        return (1000 + 1500 * gaussian_filter(rng.random((110, 160, 160)), (.6, 1, 1))).astype(np.uint16)

    def test_rigid_recovery_checkpoint_and_crop(self):
        scene = self.scene()
        truth = np.array([[0, 0, 0], [4, -2, 1], [-2, 4, -1]])
        for nominal, (dx, dy, dz) in zip((0, 20, 40), truth):
            self.tile(f'X1000.000_Y2000.000_Z{3000+nominal:.3f}',
                      scene[10+nominal+dz:50+nominal+dz, 10+dy:138+dy, 10+dx:138+dx])
        c = self.config(registration_only=True, flexible_stitch=True)
        real_pair = stitch.rigid_pair
        def interrupted(*args, **kwargs):
            if args[3] == 2:
                raise RuntimeError('simulated interruption')
            return real_pair(*args, **kwargs)
        with patch.object(stitch, 'rigid_pair', side_effect=interrupted):
            with self.assertRaisesRegex(RuntimeError, 'simulated interruption'):
                stitch.run(c)
        with np.load(self.root / 'registration' / 'rigid_state.npz') as state:
            self.assertEqual(int(state['completed']), 1)
        stitch.run(c)
        offsets = np.load(self.root / 'registration' / 'rigid_offsets_raw_xyz.npy')
        np.testing.assert_allclose(stitch.accumulate_rigid_offsets(offsets), truth, atol=1)
        with patch.object(stitch, 'rigid_pair', side_effect=AssertionError('cache missed')):
            stitch.run(c)
        out = replace(c, registration_only=False, if_subregion=True,
                      subregion_range=((16, 48), (16, 48), (25, 27)),
                      if_downsample=True, downsample_factor=(2, 2, 1))
        stitch.run(out)
        files = sorted((self.root / 'out').glob('ch*_z*.tif'))
        self.assertEqual(len(files), 2)
        self.assertEqual(tifffile.imread(files[0]).shape, (16, 16))
        expected = tifffile.imread(files[0])
        # Check pixel content against the known scene in the corrected frame.
        truth_crop = scene[35,24:56,24:56].astype(np.float32).reshape(16,2,16,2).mean(axis=(1,3))
        self.assertLess(float(np.mean(np.abs(expected-truth_crop))), 8.)
        # Resume must detect truncated pixel payloads and rebuild.
        with files[0].open('r+b') as stream:
            stream.truncate(300)
        stitch.run(out)
        np.testing.assert_array_equal(tifffile.imread(files[0]), expected)
        with self.assertRaisesRegex(RuntimeError, 'different export'):
            stitch.run(replace(out, if_xy_blend=False))

    def test_rigid_cumulative_drift_exceeds_pair_limits_and_resumes(self):
        rng = np.random.default_rng(567)
        scene = (1000 + 1500 * gaussian_filter(rng.random((130, 190, 190)),
                                              (.6, 1, 1))).astype(np.uint16)
        truth = np.arange(4)[:, None] * np.array([8, -4, 2])
        for k, (dx, dy, dz) in enumerate(truth):
            nominal = 20 * k
            self.tile(f'X1000_Y2000_Z{3000+nominal}',
                      scene[10+nominal+dz:50+nominal+dz, 30+dy:158+dy, 10+dx:138+dx])
        c = self.config(rigid_max_xy_shift_px=10, rigid_max_z_shift_px=3,
                        rigid_z_downsample=1, rigid_refine_z=False,
                        save_registration_previews=False, registration_only=True)
        real_pair = stitch.rigid_pair
        def interrupted(cfg, work, prev, cur, cache):
            if cur == 2:
                raise RuntimeError('simulated interruption')
            return real_pair(cfg, work, prev, cur, cache)
        with patch.object(stitch, 'rigid_pair', side_effect=interrupted):
            with self.assertRaisesRegex(RuntimeError, 'simulated interruption'):
                stitch.run(c)
        with np.load(self.root / 'registration' / 'rigid_state.npz') as state:
            self.assertEqual(int(state['completed']), 1)
            np.testing.assert_allclose(state['offsets'][1], truth[1], atol=1)
        stitch.run(c)
        offsets = np.load(self.root / 'registration' / 'rigid_offsets_raw_xyz.npy')
        np.testing.assert_allclose(offsets, [[0,0,0],[8,-4,2],[8,-4,2],[8,-4,2]], atol=1)
        with np.load(self.root/'registration'/'rigid_state.npz') as state:
            np.testing.assert_allclose(state['offsets'], offsets)
            self.assertEqual(str(state['offset_convention'].item()),stitch.RIGID_OFFSET_CONVENTION)
        with (self.root/'registration'/'rigid_offsets.csv').open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        np.testing.assert_allclose([[float(r[key]) for key in ('dx_raw','dy_raw','dz_raw')]
                                    for r in rows], offsets)
        self.assertEqual([int(r['prev_layer']) for r in rows],[-1,0,1,2])
        manifest = json.loads((self.root/'registration'/'registration_manifest.json').read_text())
        np.testing.assert_allclose(manifest['rigid_offsets'], offsets)
        self.assertEqual(manifest['rigid_offset_convention'],stitch.RIGID_OFFSET_CONVENTION)
        with (self.root/'registration'/'rigid_registration_qc.csv').open(newline='') as stream:
            qc = list(csv.DictReader(stream))
        np.testing.assert_allclose([[float(r[key]) for key in
                                    ('correction_x_raw','correction_y_raw','correction_z_raw')]
                                    for r in qc], offsets[1:])
        absolute = stitch.accumulate_rigid_offsets(offsets)
        np.testing.assert_allclose(absolute, truth, atol=1)
        self.assertGreater(absolute[-1, 0], c.rigid_max_xy_shift_px)
        self.assertGreater(absolute[-1, 2], c.rigid_max_z_shift_px)
        with patch.object(stitch, 'rigid_pair', side_effect=AssertionError('cache missed')):
            stitch.run(c)
        # Large accumulated drift is allowed; excessive adjacent residuals are not.
        for name, limits in (('xy', dict(rigid_max_xy_shift_px=6)),
                             ('z', dict(rigid_max_z_shift_px=1))):
            strict = replace(c, registration_dir=str(self.root / f'limited-{name}'),
                             rigid_skip_on_failure=False, **limits)
            with self.assertRaises(stitch.RigidRegistrationError):
                stitch.run(strict)

    def test_flexible_cumulative_shift_and_memory_bound(self):
        scene = self.scene()
        for k in range(3):
            self.tile(f'X1000_Y2000_Z{3000+20*k}',
                      scene[10+20*k:50+20*k, 10:138, 10+3*k:138+3*k])
        c = self.config(rigid_stitch=False, flexible_stitch=True,
                        max_local_shift_px=4, registration_only=True,
                        max_registration_megapixels=.01)
        # A whole 128x128 mosaic exceeds .01 MP; a 64-pixel patch fits.
        with patch.object(stitch, 'layer_mosaic', side_effect=AssertionError('full mosaic')):
            real_matches = stitch.streaming_local_matches
            def interrupted(*args):
                if args[3] == 2:
                    raise RuntimeError('simulated interruption')
                return real_matches(*args)
            with patch.object(stitch, 'streaming_local_matches', side_effect=interrupted):
                with self.assertRaisesRegex(RuntimeError, 'simulated interruption'):
                    stitch.run(c)
            with np.load(self.root / 'registration' / 'flexible_state.npz') as state:
                self.assertEqual(int(state['completed']), 1)
            stitch.run(c)
        with np.load(self.root / 'registration' / 'flexible_displacement_fields.npz') as state:
            med = np.median(state['fields'], axis=(1, 2))
        np.testing.assert_allclose(med[:, 0], [0, -3, -6], atol=.7)
        np.testing.assert_allclose(med[:, 1], 0, atol=.5)
        candidates = list((self.root/'registration'/'overlap_previews'/'flexible').rglob('candidate_aligned/metadata.json'))
        self.assertTrue(candidates)
        folder = candidates[0].parent
        before = folder.parent/'before_residual'
        ref = tifffile.imread(folder/'reference.tif')
        mov = tifffile.imread(folder/'moving.tif')
        unaligned = tifffile.imread(before/'moving.tif')
        common = ((tifffile.imread(folder/'reference_valid.tif') > 0) &
                  (tifffile.imread(folder/'moving_valid.tif') > 0) &
                  (tifffile.imread(before/'moving_valid.tif') > 0))
        self.assertLess(float(np.mean((ref[common]-mov[common])**2)),
                        float(np.mean((ref[common]-unaligned[common])**2)))
        with self.assertRaisesRegex(RuntimeError, 'cache is incompatible'):
            stitch.run(replace(c, registration_mesh_px=64))

    def test_nondivisible_downsampling_multichannel_labels_and_lzw(self):
        pixels = np.arange(35, dtype=np.uint16).reshape(5,7)
        data = np.stack([pixels, pixels+100])
        name = 'X1000_Y2000_Z3000'
        self.tile(name, data)
        self.tile(name, data+200, 'ch01.ome.tif')
        labels = np.full(data.shape, 4000000001, np.uint32)
        dest = self.root/'input'/name/'segmentation'/'instances'
        dest.mkdir(parents=True)
        tifffile.imwrite(dest/'instances.tif',labels,photometric='minisblack')
        c = self.config(stack_shape=(7,5,2), ch_num=2, rigid_stitch=False,
                        segmented_ch=('instances',), segmentation_mode='preserve_labels',
                        if_downsample=True, downsample_factor=(3,2,1), compression='lzw')
        stitch.run(c)
        expected = np.array([[np.rint(pixels[y:y+2,x:x+3].mean()) for x in range(0,7,3)]
                             for y in range(0,5,2)], np.uint16)
        np.testing.assert_array_equal(tifffile.imread(self.root/'out'/'ch00_z000000.tif'), expected)
        np.testing.assert_array_equal(tifffile.imread(self.root/'out'/'ch01_z000000.tif'), expected+200)
        actual_labels = tifffile.imread(self.root/'out'/'ch02_z000000.tif')
        self.assertEqual(actual_labels.dtype,np.uint32)
        self.assertTrue((actual_labels == 4000000001).all())

    def test_signs_and_strict_input(self):
        self.tile('X150700.000_Y39097.500_Z03676.904', np.ones((40,128,128), np.uint16))
        self.tile('X150699.000_Y39095.500_Z03696.904', np.ones((40,128,128), np.uint16))
        c = self.config()
        idx = stitch.load_index(c)
        byname = {p.name: idx.positions[k] for k,p in enumerate(idx.dirs)}
        np.testing.assert_array_equal(byname['X150700.000_Y39097.500_Z03676.904'], [0,0,0])
        np.testing.assert_array_equal(byname['X150699.000_Y39095.500_Z03696.904'], [1,2,20])
        np.testing.assert_allclose(idx.stage_origin_um, [-150700.,-39097.5,3676.904])
        cache = stitch.PageCache(1)
        with self.assertRaises(OSError):
            cache.read(self.root / 'missing.tif', 0, (1,1), False)
        cache.close()

    def edit_rigid_csv(self, layer, xyz):
        path = self.root/'registration'/'rigid_offsets.csv'
        with path.open(newline='') as stream:
            reader = csv.DictReader(stream)
            headings = reader.fieldnames
            rows = list(reader)
        row = next(row for row in rows if int(row['layer']) == layer)
        row.update(dict(zip(('dx_raw','dy_raw','dz_raw'),map(str,xyz))))
        row['status'] = 'manual'
        row['note'] = 'Measured in Fiji'
        with path.open('w',newline='') as stream:
            writer = csv.DictWriter(stream,fieldnames=headings)
            writer.writeheader(); writer.writerows(rows)

    def test_independent_stages_manual_fiji_offsets_and_stale_fields(self):
        for z in (3000,3020):
            self.tile(f'X1000_Y2000_Z{z}',np.zeros((40,128,128),np.uint16))
        c = self.config(workflow_stage='rigid',flexible_stitch=True,
                        stop_on_registration_failure=False,registration_only=True,
                        if_subregion=True,subregion_range=((0,16),(100,116),(0,1)))
        with patch.object(stitch,'estimate_fields',side_effect=AssertionError('rigid stage called flexible')):
            stitch.run(c)
        self.assertFalse((self.root/'out').exists())
        self.assertFalse((self.root/'registration'/'flexible_state.npz').exists())
        self.edit_rigid_csv(1,[100,-100,10])
        edited_bytes = (self.root/'registration'/'rigid_offsets.csv').read_bytes()
        with patch.object(stitch,'rigid_pair',side_effect=AssertionError('unexpected rigid calculation')):
            stitch.run_rigid(c)
        self.assertEqual((self.root/'registration'/'rigid_offsets.csv').read_bytes(),edited_bytes)
        with patch.object(stitch,'estimate_rigid',side_effect=AssertionError('unexpected rigid rerun')):
            stitch.run_flexible(c)
        offsets = np.load(self.root/'registration'/'rigid_offsets_raw_xyz.npy')
        np.testing.assert_array_equal(offsets,[[0,0,0],[100,-100,10]])
        with np.load(self.root/'registration'/'tile_index.npz') as state:
            pos = state['corrected_positions_xyz']
        np.testing.assert_array_equal(pos[1]-pos[0],[100,-100,30])
        flexible_bytes = (self.root/'registration'/'flexible_state.npz').read_bytes()
        with (patch.object(stitch,'estimate_rigid',side_effect=AssertionError('unexpected rigid rerun')),
              patch.object(stitch,'estimate_fields',side_effect=AssertionError('unexpected flexible rerun'))):
            stitch.run_export(c)
        self.assertTrue((self.root/'out'/'ch00_z000000.tif').is_file())
        self.assertEqual((self.root/'registration'/'flexible_state.npz').read_bytes(),flexible_bytes)
        edited_bytes = (self.root/'registration'/'rigid_offsets.csv').read_bytes()
        with (patch.object(stitch,'estimate_rigid',side_effect=AssertionError('preview reran rigid')),
              patch.object(stitch,'estimate_fields',side_effect=AssertionError('preview reran flexible'))):
            stitch.run(replace(c,workflow_stage='preview'))
        self.assertEqual((self.root/'registration'/'rigid_offsets.csv').read_bytes(),edited_bytes)
        self.assertTrue(list((self.root/'registration'/'overlap_previews'/'checkpoint_review').rglob('metadata.json')))
        self.edit_rigid_csv(1,[101,-100,10])
        with self.assertRaisesRegex(RuntimeError,'Flexible checkpoint is stale'):
            stitch.run_export(c)
        with self.assertRaisesRegex(RuntimeError,'cache is incompatible'):
            stitch.run_flexible(c)
        # Force flexible only: preserve the manually edited rigid CSV.
        csv_before = (self.root/'registration'/'rigid_offsets.csv').read_bytes()
        with patch.object(stitch,'estimate_rigid',side_effect=AssertionError('unexpected rigid rerun')):
            stitch.run_flexible(replace(c,force_registration=True))
        self.assertEqual((self.root/'registration'/'rigid_offsets.csv').read_bytes(),csv_before)
        with self.assertRaisesRegex(RuntimeError,'different export'):
            stitch.run_export(c)

    def test_export_saved_rigid_only_honors_edits_and_ignores_flexible(self):
        for z, value in ((3000, 100), (3002, 200)):
            self.tile(f'X1000_Y2000_Z{z}', np.full((4, 8, 8), value, np.uint16))
        c = self.config(stack_shape=(8, 8, 4), flexible_stitch=False,
                        save_registration_previews=False, workflow_stage='rigid')
        stitch.run(c)
        self.edit_rigid_csv(1, [2, 0, 1])
        regdir = self.root / 'registration'
        # Even an unreadable flexible checkpoint must be ignored by rigid-only export.
        (regdir / 'flexible_state.npz').write_bytes(b'unused flexible checkpoint')
        before = {p.name: p.read_bytes() for p in regdir.iterdir() if p.is_file()}
        with (patch.object(stitch, 'estimate_rigid', side_effect=AssertionError('rigid rerun')),
              patch.object(stitch, 'estimate_fields', side_effect=AssertionError('flexible rerun')),
              patch.object(stitch, 'load_flexible_checkpoint', side_effect=AssertionError('flexible read'))):
            stitch.run_export(c)
        metadata = json.loads((self.root / 'out' / 'export_metadata.json').read_text())
        self.assertTrue(metadata['rigid_stitch'])
        self.assertFalse(metadata['flexible_stitch'])
        self.assertEqual(metadata['rigid_offsets_xyz_raw'], [[0, 0, 0], [2, 0, 1]])
        self.assertEqual(metadata['shape_xyz'], [10, 8, 7])
        # At global Z=4 only the moving stack remains; its X origin is now 2.
        image = tifffile.imread(self.root / 'out' / 'ch00_z000004.tif')
        np.testing.assert_array_equal(image[:, :2], 0)
        np.testing.assert_array_equal(image[:, 2:], 200)
        self.assertEqual({p.name: p.read_bytes() for p in regdir.iterdir() if p.is_file()}, before)

    def test_pairwise_csv_edit_propagates_through_flexible_preview_and_export(self):
        for k in range(4):
            self.tile(f'X1000_Y2000_Z{3000+2*k}', np.full((4,8,8),100+50*k,np.uint16))
        c = self.config(stack_shape=(8,8,4), save_registration_previews=False,
                        flexible_stitch=True, stop_on_registration_failure=False)
        stitch.run_rigid(c)
        for layer, shift in enumerate(([0,0,0],[1,0,0],[2,0,0],[3,0,0])):
            self.edit_rigid_csv(layer, shift)
        stitch.run_flexible(c)
        pairwise = np.array([[0,0,0],[1,0,0],[2,0,0],[3,0,0]])
        direct = stitch.load_index(c,pairwise,normalize=True)
        corrected = stitch.corrected_index(c,stitch.load_index(c),pairwise)
        np.testing.assert_array_equal(direct.positions,[[0,0,0],[1,0,2],[3,0,4],[6,0,6]])
        np.testing.assert_array_equal(direct.positions,corrected.positions)
        with np.load(self.root/'registration'/'tile_index.npz') as saved:
            np.testing.assert_array_equal(saved['corrected_positions_xyz'],
                                          [[0,0,0],[1,0,2],[3,0,4],[6,0,6]])
        stitch.run_export(c)
        image = tifffile.imread(self.root/'out'/'ch00_z000008.tif')
        np.testing.assert_array_equal(image[:,:6],0)
        np.testing.assert_array_equal(image[:,6:],250)
        # Edit only the first pair: every subsequent layer changes, without editing its row.
        self.edit_rigid_csv(1,[2,1,1])
        with self.assertRaisesRegex(RuntimeError, 'Flexible checkpoint is stale'):
            stitch.run_export(c)
        edited = replace(c, flexible_stitch=False, export_path=str(self.root/'edited'))
        stitch.run_export(edited)
        metadata = json.loads((self.root/'edited'/'export_metadata.json').read_text())
        self.assertEqual(metadata['rigid_offset_convention'],stitch.RIGID_OFFSET_CONVENTION)
        self.assertEqual(metadata['rigid_offsets_xyz_raw'],[[0,0,0],[2,1,1],[2,0,0],[3,0,0]])
        self.assertEqual(metadata['shape_xyz'],[15,9,11])
        image = tifffile.imread(self.root/'edited'/'ch00_z000010.tif')
        np.testing.assert_array_equal(image[0],0)
        np.testing.assert_array_equal(image[1:,:7],0)
        np.testing.assert_array_equal(image[1:,7:],250)
        with patch.object(stitch,'save_checkpoint_overlap_previews') as preview:
            stitch.run(replace(edited,workflow_stage='preview'))
        np.testing.assert_array_equal(preview.call_args.args[3],
                                      [[0,0,0],[2,1,1],[2,0,0],[3,0,0]])
        stitch.run(replace(edited,workflow_stage='preview',save_registration_previews=True))
        previews = list((self.root/'registration'/'overlap_previews'/'checkpoint_review').glob(
            'attempt_*/layer_0001_to_0002/roi_*/before_pair_correction/metadata.json'))
        self.assertTrue(previews)
        before = json.loads(previews[-1].read_text())
        self.assertEqual(before['adjacent_correction_xyz_raw'],[2,0,0])
        self.assertEqual(before['moving_placement_correction_xyz_raw'],[-2,0,0])

    def test_independent_stages_require_complete_checkpoints_and_valid_edits(self):
        for z in (3000,3020):
            self.tile(f'X1000_Y2000_Z{z}',np.zeros((40,128,128),np.uint16))
        c = self.config(workflow_stage='export')
        with patch.object(stitch,'estimate_rigid',side_effect=AssertionError('unexpected rigid rerun')):
            with self.assertRaisesRegex(RuntimeError,'Missing rigid checkpoint'):
                stitch.run(c)
        stitch.run_rigid(c)
        path = self.root/'registration'/'rigid_state.npz'
        with np.load(path,allow_pickle=False) as saved:
            state = {key:saved[key] for key in saved.files}
        stitch.atomic_savez(path,**dict(state,completed=np.int64(0)))
        with self.assertRaisesRegex(RuntimeError,'Rigid checkpoint is incomplete'):
            stitch.run_flexible(c)
        stitch.atomic_savez(path,**state)
        with self.assertRaisesRegex(RuntimeError,'Missing flexible checkpoint'):
            stitch.run_export(replace(c,flexible_stitch=True))
        flexible = replace(c,flexible_stitch=True)
        idx = stitch.corrected_index(c,stitch.load_index(c),np.zeros((2,3)))
        gx,gy = stitch.field_grid(c,idx)
        signature = stitch.config_signature(flexible,idx,'flexible',np.zeros((2,3)))
        stitch.atomic_savez(self.root/'registration'/'flexible_state.npz',
                            signature=np.asarray(signature),completed=np.int64(0),
                            fields=np.zeros((2,len(gy),len(gx),2),np.float32))
        with self.assertRaisesRegex(RuntimeError,'Flexible checkpoint is incomplete'):
            stitch.run_export(flexible)
        self.edit_rigid_csv(0,[1,0,0])
        with self.assertRaisesRegex(ValueError,'reference layer 0'):
            stitch.run_flexible(c)
        self.edit_rigid_csv(0,[0,0,0])
        self.edit_rigid_csv(1,[float('nan'),0,0])
        with self.assertRaisesRegex(ValueError,'finite XYZ'):
            stitch.run_flexible(c)
        # Deliberate rigid recomputation backs up the edited CSV before replacing it.
        previous = (self.root/'registration'/'rigid_offsets.csv').read_bytes()
        stitch.run_rigid(replace(c,force_registration=True))
        self.assertEqual((self.root/'registration'/'rigid_offsets.csv.bak').read_bytes(),previous)

    def test_signal_rois_find_off_diagonal_sparse_tissue(self):
        scene = self.scene()
        for layer in range(2):
            for x,y in ((0,0),(384,0),(0,384),(384,384)):
                data = np.zeros((40,128,128),np.uint16)
                if (x,y) == (0,384):
                    data = scene[10+20*layer:50+20*layer,10:138,10+4*layer:138+4*layer]
                self.tile(f'X{1000-x}_Y{2000-y}_Z{3000+20*layer}',data)
        c = self.config(rigid_roi_px=64,rigid_roi_count=1,registration_only=True)
        idx = stitch.load_index(c)
        with stitch.PageCache(8) as cache:
            centers = stitch.signal_rigid_roi_centers(c,idx,0,1,cache)
        self.assertEqual(len(centers),1)
        self.assertLess(centers[0][0],128)
        self.assertGreater(centers[0][1],384)
        stitch.run(c)
        offsets = np.load(self.root/'registration'/'rigid_offsets_raw_xyz.npy')
        np.testing.assert_allclose(offsets[1],[4,0,0],atol=1)

    def test_overlap_preview_stacks_match_global_z_and_xyz_correction(self):
        z,y,x = np.indices((10,6,8))
        self.tile('X1000_Y2000_Z3000',(100*z+10*y+x).astype(np.uint16))
        # Moving tile's true origin is X=2, Z=7; nominal Z is 6.
        self.tile('X1000_Y2000_Z3006',(100*(z+7)+10*y+x+2).astype(np.uint16))
        c = self.config(stack_shape=(8,6,10),rigid_roi_px=8,
                        rigid_xy_downsample=1,rigid_z_downsample=1)
        idx = stitch.load_index(c)
        folder = self.root/'registration'/'overlap_previews'/'test'
        with stitch.PageCache(8) as cache:
            stitch.save_rigid_overlap_preview(c,idx,0,1,4,4,cache,folder,[2,0,1],status='test_aligned')
        a = tifffile.imread(folder/'reference.tif')
        b = tifffile.imread(folder/'moving.tif')
        common = (tifffile.imread(folder/'reference_valid.tif') > 0) & (tifffile.imread(folder/'moving_valid.tif') > 0)
        self.assertEqual(a.shape,(3,8,8))
        self.assertTrue(common.any())
        np.testing.assert_allclose(a[common],b[common],atol=1e-3)
        meta = json.loads((folder/'metadata.json').read_text())
        self.assertEqual(meta['sampled_global_z_raw'],[7,8,9])
        self.assertEqual(meta['reference_local_z_indices'],[7,8,9])
        self.assertEqual(meta['moving_local_z_indices'],[0,1,2])
        for axis in ('xy','xz','yz'):
            self.assertTrue((folder/f'overlay_{axis}.png').is_file())
        with stitch.PageCache(8) as cache:
            stitch.save_rigid_overlap_preview(c,idx,0,1,4,4,cache,folder/'none',[0,0,20])
        self.assertEqual(json.loads((folder/'none'/'metadata.json').read_text())['status'],'no_z_overlap')
        disabled = replace(c,save_registration_previews=False)
        with stitch.PageCache(8) as cache:
            stitch.save_rigid_overlap_preview(disabled,idx,0,1,4,4,cache,folder/'disabled',[0,0,0])
        self.assertFalse((folder/'disabled').exists())

    def test_blank_rigid_pair_skips_and_exports(self):
        for z in (3000,3020):
            self.tile(f'X1000_Y2000_Z{z}',np.zeros((40,128,128),np.uint16))
        c = self.config(if_subregion=True,subregion_range=((0,16),(0,16),(20,21)))
        stitch.run(c)
        np.testing.assert_array_equal(np.load(self.root/'registration'/'rigid_offsets_raw_xyz.npy'),0)
        with (self.root/'registration'/'rigid_registration_qc.csv').open(newline='') as stream:
            row = list(csv.DictReader(stream))[0]
        self.assertEqual(row['status'],'skipped_inherited')
        self.assertIn('signal-bearing',row['reason'])
        self.assertFalse(tifffile.imread(self.root/'out'/'ch00_z000020.tif').any())
        with patch.object(stitch,'rigid_pair',side_effect=AssertionError('cache missed')):
            stitch.run(c)
        strict = replace(c,rigid_skip_on_failure=False,force_registration=True)
        with self.assertRaises(stitch.RigidRegistrationError):
            stitch.run(strict)

    def test_skipped_rigid_layer_saves_zero_pair_shift_and_continues(self):
        for k in range(4):
            self.tile(f'X1000_Y2000_Z{3000+20*k}',np.ones((40,128,128),np.uint16))
        c = self.config(registration_only=True)
        idx = stitch.load_index(c)
        def pair(cfg,work,prev,cur,cache):
            if cur == 1:
                return np.array([4.,2.,1.]),1,.9
            if cur == 2:
                raise stitch.RigidRegistrationError('sparse overlap')
            np.testing.assert_array_equal(work.positions[work.layer_tiles[2]][0],[4,2,41])
            return np.array([1.,2.,0.]),1,.8
        with patch.object(stitch,'rigid_pair',side_effect=pair) as mocked:
            stitch.run(c)
        self.assertEqual(mocked.call_count,3)
        offsets = np.load(self.root/'registration'/'rigid_offsets_raw_xyz.npy')
        np.testing.assert_array_equal(offsets,[[0,0,0],[4,2,1],[0,0,0],[1,2,0]])
        np.testing.assert_array_equal(stitch.accumulate_rigid_offsets(offsets),
                                      [[0,0,0],[4,2,1],[4,2,1],[5,4,1]])
        # A real input error is not converted to a skipped alignment.
        with patch.object(stitch,'rigid_pair',side_effect=OSError('unreadable TIFF')):
            with self.assertRaisesRegex(OSError,'unreadable TIFF'):
                stitch.run(replace(c,force_registration=True))

    def test_tiff_standardization_before_downsampling_and_blending(self):
        # Rectangular/asymmetric profile catches accidental XY transposition.
        profile = np.array([[1,2,3,4,5,6,7], [2,3,4,5,6,7,8],
                            [3,4,5,6,7,8,9], [2,3,4,5,6,7,8],
                            [1,2,3,4,5,6,7]], np.float32)
        path = self.root/'illumination.tif'
        tifffile.imwrite(path, profile, photometric='minisblack')
        for name in ('X1000_Y2000_Z3000', 'X996_Y2000_Z3000'):
            data = np.stack([profile*100, profile*200]).astype(np.uint16)
            self.tile(name,data)
            self.tile(name,data*2,'ch01.ome.tif')
            dest = self.root/'input'/name/'segmentation'/'instances'
            dest.mkdir(parents=True)
            tifffile.imwrite(dest/'instances.tif', np.full(data.shape,4000000001,np.uint32),
                             photometric='minisblack')
        c = self.config(stack_shape=(7,5,2),ch_num=2,rigid_stitch=False,
                        if_standardization=True,stand_img_path=str(path),
                        segmented_ch=('instances',),segmentation_mode='preserve_labels',
                        if_downsample=True,downsample_factor=(3,2,1))
        stitch.run(c)
        for ch in range(2):
            for z in range(2):
                actual = tifffile.imread(self.root/'out'/f'ch{ch:02d}_z{z:06d}.tif')
                expected = np.rint(float(profile.mean())*100*(ch+1)*(z+1))
                np.testing.assert_array_equal(actual,np.full((3,4),expected,np.uint16))
        labels = tifffile.imread(self.root/'out'/'ch02_z000000.tif')
        self.assertTrue((labels == 4000000001).all())
        # Native-resolution output must use exactly the same correction.
        native = replace(c, export_path=str(self.root/'native'),if_downsample=False)
        stitch.run(native)
        actual = tifffile.imread(self.root/'native'/'ch00_z000000.tif')
        self.assertTrue((actual == np.rint(float(profile.mean())*100)).all())
        # A calibration replaced in-place cannot silently reuse old output.
        tifffile.imwrite(path,profile[:,::-1],photometric='minisblack')
        with self.assertRaisesRegex(RuntimeError,'different export'):
            stitch.run(c)

    def test_standardization_in_rigid_flexible_and_checkpoint_previews(self):
        scene = self.scene().astype(np.float32)
        yy, xx = np.mgrid[:128, :128]
        profile = (0.25 + 0.75 * (xx / 127) * (yy / 127)).astype(np.float32)
        path = self.root/'illumination.tif'
        tifffile.imwrite(path, profile, photometric='minisblack')
        for nominal, dx in ((0, 0), (20, 4)):
            self.tile(f'X1000_Y2000_Z{3000+nominal}',
                      scene[10+nominal:50+nominal, 10:138, 10+dx:138+dx] * profile)
        c = self.config(workflow_stage='rigid', flexible_stitch=True,
                        if_standardization=True, stand_img_path=str(path),
                        save_registration_previews=True)
        real_read = stitch.PageCache.read
        counts = {}
        stage = 'rigid'
        def checked_read(cache, source, z, factors, nearest, standard_factor=None):
            self.assertIsNotNone(cache.standard_factor)
            actual = real_read(cache, source, z, factors, nearest, standard_factor)
            with tifffile.TiffFile(source) as handle:
                native = handle.pages[z].asarray().astype(np.float32)
            corrected = native * (profile.mean(dtype=np.float64) / profile).astype(np.float32)
            dx, dy = factors
            expected = np.array([[corrected[y:y+dy, x:x+dx].mean()
                                  for x in range(0, 128, dx)]
                                 for y in range(0, 128, dy)], np.float32)
            np.testing.assert_allclose(actual, expected, rtol=2e-6)
            counts[stage] = counts.get(stage, 0) + 1
            return actual
        with patch.object(stitch.PageCache, 'read', new=checked_read):
            with patch.object(stitch, 'load_standardization', wraps=stitch.load_standardization) as load:
                stitch.run(c)
                self.assertEqual(load.call_count, 1)
            offsets = np.load(self.root/'registration'/'rigid_offsets_raw_xyz.npy')
            np.testing.assert_allclose(offsets[1], [4, 0, 0], atol=1.1)
            stage = 'flexible'
            stitch.run_flexible(c)
            stage = 'preview'
            stitch.run(replace(c, workflow_stage='preview'))
        self.assertTrue(all(counts.get(name, 0) > 0 for name in ('rigid', 'flexible', 'preview')))
        idx = stitch.load_index(c)
        offsets = np.load(self.root/'registration'/'rigid_offsets_raw_xyz.npy')
        corrected_idx = stitch.corrected_index(c, idx, offsets)
        old_flex = stitch.config_signature(c, corrected_idx, 'flexible', offsets)
        # Replacing the calibration at the same path invalidates both stages.
        tifffile.imwrite(path, profile[:, ::-1], photometric='minisblack')
        self.assertNotEqual(old_flex, stitch.config_signature(c, corrected_idx, 'flexible', offsets))
        with self.assertRaisesRegex(RuntimeError, 'Rigid registration cache does not match'):
            stitch.run(c)
        with self.assertRaisesRegex(RuntimeError, 'cache is incompatible'):
            stitch.run_flexible(c)
        with self.assertRaisesRegex(RuntimeError, 'does not match'):
            stitch.run(replace(c, if_standardization=False))

    def test_reuse_rigid_offsets_with_changed_estimation_settings(self):
        for z in (3000,3020):
            self.tile(f'X1000_Y2000_Z{z}',np.zeros((40,128,128),np.uint16))
        c = self.config(flexible_stitch=True, stop_on_registration_failure=False)
        stitch.run_rigid(c)
        self.edit_rigid_csv(1,[2,-3,1])
        csv_path = self.root/'registration'/'rigid_offsets.csv'
        csv_before = csv_path.read_bytes()
        state_path = self.root/'registration'/'rigid_state.npz'
        state_before = state_path.read_bytes()
        profile = self.root/'illumination.tif'
        tifffile.imwrite(profile,np.ones((128,128),np.float32),photometric='minisblack')
        changed = replace(c, rigid_xy_downsample=4, rigid_z_downsample=1,
                          rigid_roi_px=96, rigid_refine_z=False,
                          if_standardization=True, stand_img_path=str(profile))
        with patch.object(stitch,'estimate_rigid',side_effect=AssertionError('rigid rerun')):
            stitch.run_flexible(changed)
            # Changing only rigid estimation settings must also allow flexible reuse.
            stitch.run_flexible(replace(changed,rigid_roi_px=80,rigid_min_ncc=.7))
            stitch.run_export(replace(changed,rigid_roi_px=80))
            self.assertEqual(csv_path.read_bytes(),csv_before)
            self.assertEqual(state_path.read_bytes(),state_before)
            with self.assertRaisesRegex(RuntimeError,'source geometry'):
                stitch.run_flexible(replace(changed,voxel_size=(2.,1.,1.)))
            self.edit_rigid_csv(1,[3,-3,1])
            with self.assertRaisesRegex(RuntimeError,'cache is incompatible'):
                stitch.run_flexible(changed)

    def test_rigid_checkpoint_rejects_changed_grouping_same_layer_count(self):
        for z in ('3000', '3000.6', '3001.2'):
            self.tile(f'X1000_Y2000_Z{z}', np.zeros((4, 8, 8), np.uint16))
        old = self.config(stack_shape=(8, 8, 4), layer_z_tolerance_um=0)
        current = replace(old, layer_z_tolerance_um=1)
        old_idx, idx = stitch.load_index(old), stitch.load_index(current)
        np.testing.assert_array_equal(old_idx.layer_starts, idx.layer_starts)
        self.assertFalse(np.array_equal(old_idx.layer_ids, idx.layer_ids))
        regdir = self.root / 'registration'
        regdir.mkdir()
        stitch.write_rigid_edit_checkpoint(old, old_idx, regdir,
                                           stitch.config_signature(old, old_idx, 'rigid'), np.zeros((2, 3)))
        with self.assertRaisesRegex(RuntimeError, 'source geometry'):
            stitch.load_rigid_checkpoint(current, idx, regdir,
                                         stitch.config_signature(current, idx, 'rigid'), reuse_completed=True)

    def test_standardization_profile_validation_and_disabled_mode(self):
        path = self.root/'illumination.tif'
        c = self.config(stack_shape=(7,5,2),if_standardization=True,stand_img_path=str(path))
        with self.assertRaisesRegex(ValueError,'requires stand_img_path'):
            stitch.load_standardization(replace(c,stand_img_path=None))
        cases = [(np.zeros((5,7),np.float32),'positive mean'),
                 (np.full((5,7),-1,np.float32),'nonnegative'),
                 (np.full((5,7),np.nan,np.float32),'finite'),
                 (np.ones((7,5),np.float32),'YX grayscale'),
                 (np.ones((2,5,7),np.float32),'exactly one')]
        for data,message in cases:
            with self.subTest(message=message):
                tifffile.imwrite(path,data,photometric='minisblack')
                with self.assertRaisesRegex(ValueError,message):
                    stitch.load_standardization(c)
        profile = np.ones((5,7),np.float32)
        profile[0,0] = 0
        tifffile.imwrite(path,profile,photometric='minisblack')
        factor = stitch.load_standardization(c)
        self.assertTrue(np.isfinite(factor).all())
        self.assertEqual(factor[0,0],100)
        # Disabled mode does not even open the calibration file.
        self.assertIsNone(stitch.load_standardization(replace(c,if_standardization=False,
                                                             stand_img_path='missing.tif')))

    def test_overlap_selection_and_z_coverage(self):
        # Two XY tiles in first layer; shifted XY extent in next layer.
        self.tile('X1000_Y2000_Z3000', np.full((40,128,128), 100, np.uint16))
        self.tile('X936_Y2000_Z3000', np.full((40,128,128), 300, np.uint16))
        self.tile('X904_Y2000_Z3020', np.full((40,128,128), 500, np.uint16))
        c = self.config(rigid_stitch=False)
        idx = stitch.load_index(c)
        gx,gy = stitch.field_grid(c,idx)
        fields = np.zeros((2,len(gy),len(gx),2),np.float32)
        cache = stitch.PageCache(8)
        def render(cfg,z,x):
            return stitch.render_output_tile(cfg,idx,fields,cache,z,0,0,x,128,224,None,idx.xy_min)
        # Z overlap endpoint: next layer's sole XY coverage remains visible.
        self.assertEqual(int(render(c,20,192)[8,8]),500)
        # No Z blend must fall back where the preferred layer has no XY tile.
        self.assertGreater(int(render(replace(c,if_z_blend=False),39,0)[8,8]),0)
        # Disabling XY blend selects one tile rather than averaging intensities.
        hard = render(replace(c,if_xy_blend=False),0,64)
        self.assertTrue(np.isin(hard, [100,300]).all())
        soft = render(c,0,64)
        self.assertTrue(((soft > 100) & (soft < 300)).any())
        cache.close()


if __name__ == '__main__':
    unittest.main(verbosity=2)
