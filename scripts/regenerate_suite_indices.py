#!/usr/bin/env python
"""Regenerate index files for v2 suites from existing NPZ metadata."""
import json
from pathlib import Path

SPATIAL = Path('/home/robot/codePWC/lara_repro/LaRA-VLA/output/spatial_lara_libero')
FK = 8

for suite in ['libero_spatial', 'libero_object', 'libero_goal']:
    v2_dir = SPATIAL / f'{suite}_v2'
    if not v2_dir.exists():
        print(f'{suite}: v2 dir not found, skipping')
        continue

    lines = []
    for meta_path in sorted(v2_dir.glob('task_*/demo_*/episode_*_meta.json')):
        with open(meta_path) as f:
            m = json.load(f)
        tid = m['task_id']
        did = m['demo_id']
        lr_ep = m.get('lerobot_episode', 0)
        T = m['T']
        objs = m.get('objects_of_interest', [])
        ep_rel = f'{suite}_v2/task_{tid:02d}/demo_{did:06d}/episode_{did:06d}.npz'
        for cf in range(T):
            lines.append(json.dumps({
                'suite': suite, 'task_id': tid, 'demo_id': did,
                'cot_episode_id': lr_ep, 'hdf5_demo_id': did,
                'cot_frame_idx': cf, 'cot_future_idx': min(cf + FK, T - 1),
                'hdf5_frame_idx': cf, 'hdf5_future_idx': min(cf + FK, T - 1),
                'T_cot': T, 'T_hdf5': T,
                'episode_path': ep_rel,
                'meta_path': ep_rel.replace('.npz', '_meta.json'),
                'primary_object': objs[0] if objs else '',
                'objects_of_interest': objs,
                'camera_names': ['agentview', 'robot0_eye_in_hand'],
                'alignment_method': 'identity_no_noops_hdf5',
            }, ensure_ascii=False))

    out = SPATIAL / f'spatial_lara_libero_index_{suite}_v2.jsonl'
    with open(out, 'w') as f:
        for line in lines:
            f.write(line + '\n')
    n_demos = len(set(json.loads(l)['demo_id'] for l in lines))
    print(f'{suite}: {n_demos} demos, {len(lines)} frames')
    print(f'  -> {out}')
