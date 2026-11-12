"""Read acquisition metadata only (no image stack is loaded)."""
import json
import sys
import nd2

for path in sys.argv[1:]:
    with nd2.ND2File(path) as f:
        channels = f.metadata.channels
        loops = f.loop_indices
        points = [p for e in f.experiment if e.type == 'XYPosLoop'
                  for p in e.parameters.points]
        print(json.dumps({
            'path': path, 'sizes': dict(f.sizes), 'dtype': str(f.dtype),
            'voxel_um': list(f.voxel_size()), 'frames': len(loops),
            'first_indices': loops[:3], 'last_indices': loops[-3:],
            'first_positions': [list(p.stagePositionUm) for p in points[:3]],
            'last_positions': [list(p.stagePositionUm) for p in points[-3:]],
            'channels': [str(c.channel.name) for c in channels],
            'volume': str(channels[0].volume),
            'first_frame': str(f.frame_metadata(0)),
        }, indent=2), flush=True)
