from pathlib import Path
from dataclasses import replace
from unittest.mock import patch
import tempfile, json, contextlib, io
import numpy as np
import tifffile
import auto_xyz_flexible_export as s
from test_auto_xyz_flexible_export import StitchTests

c=s.Config(registration_downsample=1,registration_mesh_px=1,field_smoothing_sigma=0,max_local_shift_px=256)
p=[(0,0),(1,0),(0,1),(1,1),(.5,.5)]
v=np.array([[100,0],[-100,0],[0,100],[0,-100],[100,100]])
median=np.median(v,axis=0)
print('OUTLIERS: retained by criterion =',int(np.sum(np.linalg.norm(v-median,axis=1)<=64)), '; returned field =',s.interpolate_matches(p,v.tolist(),[1]*5,c,None,(2,2)).tolist())
print('SINGLE_MATCH: whole-grid X range =',s.interpolate_matches([(0,0)],[(200,0)],[.2],c,None,(10,10))[...,0].min(),s.interpolate_matches([(0,0)],[(200,0)],[.2],c,None,(10,10))[...,0].max())
idx=type('I',(),{'layer_starts':np.array([0.,10.])})()
c2=replace(c,stack_shape=(16,16,40),rigid_roi_px=16,rigid_xy_downsample=1,rigid_z_downsample=2,rigid_refine_z_planes=3,rigid_max_z_shift_px=10,rigid_min_ncc=.2)
with patch.object(s,'roi_mosaic',return_value=(np.ones((16,16),np.float32),np.ones((16,16),bool))), patch.object(s,'masked_ncc',return_value=(-.5,1.)), contextlib.redirect_stdout(io.StringIO()):
    refined=s.refine_rigid_z(c2,idx,0,1,8,8,np.array([0.,0.,0.]),None)
print('Z_REFINEMENT: NCC=-0.5 < required 0.2, result=',refined.tolist())
print('FRACTIONAL_Z: start=10.5, stack depth=4, exported local pages=',[round(z-10.5) for z in range(11,15)])

case=StitchTests();case.setUp()
try:
    case.tile('X0_Y0_Z0',np.full((2,12,12),100,np.uint16))
    case.tile('X1_Y0_Z0',np.full((2,12,12),100,np.uint16))
    for name in ['X0_Y0_Z0','X1_Y0_Z0']:
        path=case.root/'input'/name/'segmentation'/'instances';path.mkdir(parents=True)
        tifffile.imwrite(path/'instances.tif',np.full((2,12,12),7,np.uint32),photometric='minisblack')
    cfg=case.config(stack_shape=(12,12,2),flip_stage_xy=False,rigid_stitch=False,flexible_stitch=False,segmented_ch=('instances',),segmentation_mode='preserve_labels',if_downsample=True,downsample_factor=(4,4,1),save_registration_previews=False)
    with contextlib.redirect_stdout(io.StringIO()): s.run(cfg)
    sig=tifffile.imread(case.root/'out'/'ch00_z000000.tif');lab=tifffile.imread(case.root/'out'/'ch01_z000000.tif')
    print('LABEL_EDGE: signal last col=',sig[:,-1].tolist(),'; labels last col=',lab[:,-1].tolist())
    with contextlib.redirect_stdout(io.StringIO()): s.run(replace(cfg,resume=False,if_subregion=True,subregion_range=((0,8),(0,8),(0,1))))
    metadata=json.loads((case.root/'out'/'export_metadata.json').read_text())
    print('REEXPORT: metadata Z count=',metadata['shape_xyz'][2],'; remaining signal TIFFs=',[p.name for p in (case.root/'out').glob('ch00_z*.tif')])
finally: case.tearDown()
case=StitchTests();case.setUp()
try:
    for z in (0,1): case.tile(f'X0_Y0_Z{z}',np.full((2,12,12),100,np.uint16))
    cfg=case.config(stack_shape=(12,12,2),workflow_stage='rigid',save_registration_previews=False)
    with patch.object(s,'rigid_pair',return_value=(np.array([3.,0.,0.]),1,.9)), contextlib.redirect_stdout(io.StringIO()): s.run(cfg)
    with np.load(case.root/'registration'/'rigid_state.npz') as state: before=state['offsets'].copy()
    with contextlib.redirect_stdout(io.StringIO()): s.run(replace(cfg,rigid_stitch=False))
    with np.load(case.root/'registration'/'rigid_state.npz') as state: after=state['offsets'].copy()
    try:
        with contextlib.redirect_stdout(io.StringIO()): s.run(cfg)
        outcome='unexpectedly accepted'
    except RuntimeError as e: outcome=str(e)
    print('DISABLE_RIGID: saved offsets before=',before.tolist(),'; after=',after.tolist(),'; reenabling=',outcome)
finally: case.tearDown()
