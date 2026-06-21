#!/usr/bin/env python
"""Rebuild any LIBERO suite from no-noops HDF5 (identity mapping, no DTW)."""
import os, sys, json, numpy as np, h5py, pandas as pd, imageio
from pathlib import Path; from PIL import Image

_REPO = Path(__file__).resolve().parents[1]
_LARA_REPRO = _REPO.parent

NOOP_DIR = Path(os.environ.get('NOOP_HDF5_ROOT', str(_LARA_REPRO / 'datasets/clip-rt/modified_libero_hdf5')))
LEROBOT = Path(os.environ.get('LEROBOT_ROOT', str(_LARA_REPRO / 'datasets/lovejuly/libero_lerobot_all')))
OUT_BASE = _REPO / 'output' / 'spatial_lara_libero'
RES, FK = 224, 8
CAMS = ['agentview', 'robot0_eye_in_hand']

def resolve_seg_id(obj_name, inst_to_id):
    """Match obj name to seg ID with prefix fallback (for region names)."""
    if obj_name in inst_to_id:
        return inst_to_id[obj_name]
    for inst_name, sid in inst_to_id.items():
        if obj_name.startswith(inst_name):
            return sid
    for inst_name, sid in inst_to_id.items():
        if inst_name.startswith(obj_name):
            return sid
    return None

_LIBERO = os.environ.get('LIBERO_HOME', str(_LARA_REPRO / 'LIBERO'))
os.environ.setdefault('LIBERO_HOME', _LIBERO)
os.environ.setdefault('LIBERO_CONFIG_PATH', str(Path(_LIBERO) / 'libero'))
if _LIBERO not in sys.path:
    sys.path.insert(0, _LIBERO)
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--suite', required=True)
args = parser.parse_args()
SUITE = args.suite
OUT_DIR = OUT_BASE / f'{SUITE}_v2'
OUT_DIR.mkdir(parents=True, exist_ok=True)

ts = benchmark.get_benchmark_dict()[SUITE]()

# Load LeRobot episodes for this suite
lr_dir = LEROBOT / f'{SUITE}_no_noops_1.0.0_lerobot'
lr_by_task = {}
with open(lr_dir / 'meta' / 'episodes.jsonl') as f:
    for line in f:
        ep = json.loads(line)
        pq = lr_dir / 'data' / 'chunk-000' / f'episode_{ep["episode_index"]:06d}.parquet'
        T = len(pd.read_parquet(pq))
        lr_by_task.setdefault(ep['tasks'][0], []).append((ep['episode_index'], T))

noop_suite_dir = NOOP_DIR / f'{SUITE}_no_noops'
new_index, total_ep, total_fr = [], 0, 0

for tid in range(ts.n_tasks):
    t = ts.get_task(tid); desc = t.language
    bddl = Path(get_libero_path('bddl_files')) / t.problem_folder / t.bddl_file
    nf = t.bddl_file.replace('.bddl', '') + '_demo.hdf5'
    noop_path = noop_suite_dir / nf
    if not noop_path.exists():
        print(f'task_{tid}: no file'); continue

    # Get no-noops demos
    with h5py.File(noop_path, 'r') as f:
        noop_demos = {dk: f[f'data/{dk}/actions'].shape[0] for dk in f['data'] if dk.startswith('demo_')}

    lr_eps = sorted(lr_by_task.get(desc, []), key=lambda x: x[0])
    if not lr_eps:
        print(f'task_{tid}: no LR episodes'); continue

    # Pre-load instance mapping
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=RES, camera_widths=RES,
                             camera_names=CAMS, camera_segmentations='instance')
    env.reset()
    inst = {n: i + 1 for i, n in enumerate(env.env.model.instances_to_ids)}
    objs = list(env.obj_of_interest)
    env.close()

    matched = 0
    for lr_ep, lr_T in lr_eps:
        candidates = [(dk, T) for dk, T in noop_demos.items() if T == lr_T]
        if not candidates:
            continue
        if len(candidates) == 1:
            dk = candidates[0][0]
        else:
            lr_rgb = imageio.get_reader(
                str(lr_dir / 'videos' / 'chunk-000' / 'observation.images.image' / f'episode_{lr_ep:06d}.mp4')
            ).get_data(0)
            lr_s = np.array(Image.fromarray(lr_rgb).resize((64, 64))).astype(float) / 255
            best_dk, best_d = None, 1.0
            for dk_c, _ in candidates:
                with h5py.File(noop_path, 'r') as f:
                    h5_rgb = f[f'data/{dk_c}/obs/agentview_rgb'][0][::-1, ::-1]
                h5_s = np.array(Image.fromarray(h5_rgb).resize((64, 64))).astype(float) / 255
                d = np.abs(lr_s - h5_s).mean()
                if d < best_d: best_dk, best_d = dk_c, d
            dk = best_dk
        did = int(dk.split('_')[1]); matched += 1

        # State replay
        env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=RES, camera_widths=RES,
                                 camera_names=CAMS, camera_segmentations='instance')
        env.reset()
        with h5py.File(noop_path, 'r') as f:
            states = f[f'data/{dk}/states'][:]
            actions = f[f'data/{dk}/actions'][:].astype(np.float32)
        T = len(states)
        rgb_a = np.zeros((T, RES, RES, 3), np.uint8); rgb_w = np.zeros_like(rgb_a)
        seg_a = np.zeros((T, RES, RES), np.int32); seg_w = np.zeros_like(seg_a)
        aff_a = np.zeros((T, RES, RES), np.uint8); aff_w = np.zeros_like(aff_a)
        eef_p = np.zeros((T, 3), np.float32); eef_q = np.zeros((T, 4), np.float32)
        gr = np.zeros((T, 2), np.float32); jt = np.zeros((T, 7), np.float32); ft = np.zeros(T, np.int64)

        for ti in range(T):
            if ti % 200 == 0:
                print(f'\r  task_{tid} demo_{did}: {ti}/{T}', end='', flush=True)
            env.sim.set_state_from_flattened(states[ti]); env.sim.forward()
            env._update_observables(force=True); obs = env.env._get_observations()
            rk_a, rk_w = f'{CAMS[0]}_image', f'{CAMS[1]}_image'
            sk_a, sk_w = f'{CAMS[0]}_segmentation_instance', f'{CAMS[1]}_segmentation_instance'
            if rk_a in obs: rgb_a[ti] = obs[rk_a][::-1, ::-1]
            if rk_w in obs: rgb_w[ti] = obs[rk_w][::-1, ::-1]
            if sk_a in obs:
                s = obs[sk_a][::-1, ::-1]; seg_a[ti] = s[..., 0] if s.ndim == 3 else s
            if sk_w in obs:
                s = obs[sk_w][::-1, ::-1]; seg_w[ti] = s[..., 0] if s.ndim == 3 else s
            ua = np.zeros((RES, RES), bool); uw = np.zeros_like(ua)
            for obj in objs:
                sid = resolve_seg_id(obj, inst)
                if sid: ua |= seg_a[ti] == sid; uw |= seg_w[ti] == sid
            aff_a[ti] = ua.astype(np.uint8); aff_w[ti] = uw.astype(np.uint8)
            if 'robot0_eef_pos' in obs: eef_p[ti] = obs['robot0_eef_pos'].astype(np.float32)
            if 'robot0_eef_quat' in obs: eef_q[ti] = obs['robot0_eef_quat'].astype(np.float32)
            if 'robot0_gripper_qpos' in obs: gr[ti] = obs['robot0_gripper_qpos'].astype(np.float32)
            if 'robot0_joint_pos' in obs: jt[ti] = obs['robot0_joint_pos'].astype(np.float32)
            ft[ti] = min(ti + FK, T - 1)

        od = OUT_DIR / f'task_{tid:02d}' / f'demo_{did:06d}'; od.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(od / f'episode_{did:06d}.npz',
            rgb_agentview=rgb_a, rgb_wrist=rgb_w, seg_agentview=seg_a, seg_wrist=seg_w,
            affordance_mask_agentview=aff_a, affordance_mask_wrist=aff_w,
            primary_pose_world=np.zeros((T, 7), np.float32), primary_pose_eef=np.zeros((T, 7), np.float32),
            interest_poses_world=np.zeros((T, len(objs), 7), np.float32),
            interest_poses_eef=np.zeros((T, len(objs), 7), np.float32),
            robot0_eef_pos=eef_p, robot0_eef_quat=eef_q, robot0_gripper_qpos=gr, robot0_joint_pos=jt,
            actions=actions, future_indices=ft)
        json.dump({'suite': SUITE, 'task_id': tid, 'demo_id': did, 'lerobot_episode': lr_ep, 'T': T,
            'objects_of_interest': objs, 'camera_names': CAMS,
            'instance_to_id': {str(k): int(v) for k, v in inst.items()},
            'cot_frame==hdf5_frame': True, 'source': 'clip-rt/no_noops'},
            open(od / f'episode_{did:06d}_meta.json', 'w'), indent=2)
        ep_rel = f'{SUITE}_v2/task_{tid:02d}/demo_{did:06d}/episode_{did:06d}.npz'
        for cf in range(T):
            new_index.append(json.dumps({'suite': SUITE, 'task_id': tid, 'demo_id': did,
                'cot_episode_id': lr_ep, 'hdf5_demo_id': did,
                'cot_frame_idx': cf, 'cot_future_idx': min(cf + FK, T - 1),
                'hdf5_frame_idx': cf, 'hdf5_future_idx': min(cf + FK, T - 1),
                'T_cot': T, 'T_hdf5': T, 'episode_path': ep_rel,
                'meta_path': ep_rel.replace('.npz', '_meta.json'),
                'primary_object': objs[0] if objs else '', 'objects_of_interest': objs,
                'camera_names': CAMS,
                'alignment_method': 'identity_no_noops_hdf5'}, ensure_ascii=False))
            total_fr += 1
        total_ep += 1; env.close()
    print(f'\rtask_{tid}: {matched}/{len(lr_eps)} demos{"":30s}')

# Save index (this suite only — will merge later)
idx_path = OUT_BASE / f'spatial_lara_libero_index_{SUITE}_v2.jsonl'
with open(idx_path, 'w') as f:
    for line in new_index: f.write(line + '\n')
print(f'Index: {idx_path} ({total_ep} episodes, {total_fr} frames)')
