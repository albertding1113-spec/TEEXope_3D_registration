#!/usr/bin/env python3
"""Edit the dataset paths and options, then run this file or use the Slurm launcher."""
import argparse
from dataclasses import replace
from auto_xyz_flexible_export import Config, run

c = Config(
    # CSV row i stores the correction of layer i relative to layer i-1, in raw XYZ voxels.
    # Editing one row also shifts all subsequent layers; export accumulates internally.
    # If no rigid checkpoint exists yet, first run with --stage rigid.
    workflow_stage="export",
    save_registration_previews=True,
    registration_preview_flexible_patches=5,
    # Acquisition folders: X..._Y..._Z.../0/ch00.ome.tif
    dir_path="/HWFS/zhaohu_lab/dingjiayi/data/260928_ER-mBaoJin_mito-mScarlet3H",
    export_path="",
    # Shared registration cache, independent of export_path.
    registration_dir="/HWFS/zhaohu_lab/dingjiayi/data/260928_ER-mBaoJin_mito-mScarlet3H/261028_registration_xyz_flexible",
    ch_num=2,
    stack_shape=(2048, 2048, 640),
    voxel_size=(0.1489, 0.1489, 0.5),
    flip_stage_xy=True,

    # 1) Coarse rigid alignment: automatically estimate large XYZ translations.
    rigid_stitch=True,
    registration_channel=0,  # choose one channel with features shared across layers
    rigid_xy_downsample=16,
    rigid_z_downsample=8,
    rigid_roi_px=4096,
    rigid_roi_count=5,
    rigid_signal_candidates=64,  # scout candidate regions for signal before reading 3D ROIs
    rigid_signal_z_samples=5,
    rigid_skip_on_failure=True,  # save zero pair shift and inherit preceding correction if alignment fails
    rigid_max_xy_shift_px=512,  # additional X/Y correction per adjacent layer, raw pixels
    rigid_max_z_shift_px=256,  # additional Z correction per adjacent layer, raw voxels
    rigid_min_ncc=0.10,
    rigid_min_overlap=0.05,
    rigid_refine_z=True,

    # 2) False exports only rigid XYZ corrections; no flexible checkpoint is needed.
    # Set True to apply flexible registration after computing it with --stage flexible.
    flexible_stitch=True,
    registration_downsample=16,
    registration_mesh_px=2048,
    registration_patch_px=8192,
    registration_z_samples=3,
    min_phase_response=0.20,
    min_patch_coverage=0.50,
    max_local_shift_px=512,
    field_smoothing_sigma=0.8,

    # 3) Blending, shared illumination correction and export options.
    if_xy_blend=True,
    if_z_blend=True,
    # One grayscale TIFF slice, same YX shape/orientation as a raw tile page.
    # Applied before rigid/flexible registration and signal export; labels are unchanged.
    if_standardization=True,
    stand_img_path="/HWFS/zhaohu_lab/dingjiayi/data/standardization_image.tif",  # e.g. "/path/to/average_illumination.tif"
    if_downsample=True,
    downsample_factor=(10, 10, 100),
    if_subregion=False,
    subregion_range=((41500, 44500), (23000, 26000), (3100, 3300)),
    compression="lzw",   # use None if imagecodecs is unavailable
    tiff_tile=256,
    tiff_maxworkers=2,
    cache_mb=512,
    cache_open_files=64,
    strict_input=True,  # fail on a missing/corrupt requested TIFF page

    # Legacy option used only with workflow_stage="all"; explicit stages take precedence.
    registration_only=False,
    # False reuses checkpoints. --force-registration recomputes only the selected stage.
    force_registration=False,
    resume=True,
    stop_on_registration_failure=True,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Independent registration/export stages")
    parser.add_argument('--stage', choices=('rigid','flexible','export','preview','all'), default=c.workflow_stage)
    parser.add_argument('--force-registration', action='store_true', default=None,
                        help='Recompute only the selected registration stage; rigid overwrites edited CSV (backup saved)')
    args = parser.parse_args()
    run(replace(c,workflow_stage=args.stage,
                force_registration=c.force_registration if args.force_registration is None
                else args.force_registration))
