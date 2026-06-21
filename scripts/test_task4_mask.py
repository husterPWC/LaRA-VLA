#!/usr/bin/env python
"""Quick test for task_04 mask filtering logic."""
import json, sys, numpy as np
sys.path.insert(0, 'str(_REPO.parent)/LaRA-VLA')
from lara_vla.data.spatial_cot_dataset import SpatialCoTDataset

SPATIAL = 'str(_REPO.parent)/LaRA-VLA/output/spatial_lara_libero'
ds = SpatialCoTDataset(
    SPATIAL, SPATIAL + '/spatial_lara_libero_index_cot.jsonl',
    'str(_REPO.parent)/datasets/lovejuly/libero_lerobot_all',
    SPATIAL + '/cot_spatial_alignment.json',
    enable_dynamic_mask=True)

TID = 4
# Get all demo IDs for task_4
demos = set()
for e in ds.entries:
    if e['suite'] == 'libero_10' and e['task_id'] == TID:
        demos.add(e['demo_id'])
demos = sorted(demos)

print(f"task_{TID}: {len(demos)} demos\n")

for did in demos[:8]:  # first 8 demos
    # Find subtask transitions
    cot_ep = None
    key_frames = set()
    for i, e in enumerate(ds.entries):
        if e['suite'] == 'libero_10' and e['task_id'] == TID and e['demo_id'] == did:
            if cot_ep is None:
                cot_ep = e['cot_episode_id']
            cf = e['cot_frame_idx']
            # Sample every 20 frames + first frame
            if cf == 0 or cf % 20 == 0:
                key_frames.add(cf)

    key_frames = sorted(key_frames)[:8]

    # Detect switches
    prev_held = None
    issues = []
    for cf in key_frames:
        for i, e in enumerate(ds.entries):
            if (e['suite'] == 'libero_10' and e['task_id'] == TID and
                e['demo_id'] == did and e['cot_frame_idx'] == cf):
                s = ds[i]
                held = ds._get_last_held_object('libero_10', cot_ep, cf)
                cmap = ds._last_container_cache.get(('libero_10', cot_ep), {})
                cont = cmap.get(cf, '')
                if held and held != prev_held:
                    key_frames.append(cf)
                prev_held = held
                break

    key_frames = sorted(set(key_frames))[:10]

    print(f"demo_{did:06d} (cot_ep={cot_ep}):")
    prev_held = None
    for cf in key_frames:
        for i, e in enumerate(ds.entries):
            if (e['suite'] == 'libero_10' and e['task_id'] == TID and
                e['demo_id'] == did and e['cot_frame_idx'] == cf):
                s = ds[i]
                held = ds._get_last_held_object('libero_10', cot_ep, cf)
                cmap = ds._last_container_cache.get(('libero_10', cot_ep), {})
                cont = cmap.get(cf, '')
                n_rel = s.get('num_relevant_objects', 0)
                grip = s.get('cot_gripper_state', -1)
                mask_px = s['affordance_mask_agentview'].sum()

                # Detect issues
                status = ""
                if prev_held and held != prev_held:
                    status = " ← SWITCH"
                prev_held = held

                print(f"  CoT{cf:4d}: grip={grip} held={held:25s} cont={cont:15s} n_rel={n_rel} mask={mask_px:5d}px{status}")
                break
    print()

print("Done. Check: held switches once (porcelain→white_yellow), cont switches once (plate_1→plate_2), n_rel=2 throughout.")
