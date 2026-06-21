#!/usr/bin/env python
"""Verify LeRobot ↔ Spatial alignment with four-panel comparison.

For libero_10: reads from no-noops NPZ (libero_10_v2/)
For other suites: reads from original NPZ via DTW-mapped index

Usage:
    python tools/verify_alignment.py --suite libero_10 --task-id 0 --demo-id 12
"""

import argparse, json, sys, numpy as np, imageio
from pathlib import Path; from PIL import Image, ImageDraw

_THIS = Path(__file__).resolve(); _REPO = _THIS.parents[1]; sys.path.insert(0, str(_REPO))

SPATIAL = _REPO / "output" / "spatial_lara_libero"
COT = os.environ.get("LEROBOT_ROOT", str(_REPO.parent / "datasets/lovejuly/libero_lerobot_all")))
OUT = _REPO / "output" / "alignment_verification"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", default="libero_10")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--demo-id", type=int, default=0)
    args = parser.parse_args()

    suite, tid, did = args.suite, args.task_id, args.demo_id

    # Determine NPZ path
    v2_dir = SPATIAL / f"{suite}_v2" / f"task_{tid:02d}" / f"demo_{did:06d}"
    orig_dir = SPATIAL / suite / f"task_{tid:02d}" / f"demo_{did:06d}"
    if v2_dir.exists():
        data_dir = v2_dir; version = "v2 (no-noops)"
    elif orig_dir.exists():
        data_dir = orig_dir; version = "original (DTW)"
    else:
        print(f"❌ Demo not found: {v2_dir} or {orig_dir}"); return

    npz_path = data_dir / f"episode_{did:06d}.npz"
    meta_path = data_dir / f"episode_{did:06d}_meta.json"
    if not npz_path.exists():
        print(f"❌ NPZ not found: {npz_path}"); return

    with open(meta_path) as f: meta = json.load(f)
    lr_ep = meta.get("lerobot_episode", 0)
    T = meta.get("T", 0)
    objs = meta.get("objects_of_interest", [])
    inst = meta.get("instance_to_id", {})
    is_v2 = meta.get("cot_frame==hdf5_frame", False)

    print(f"PAIR: LeRobot ep {lr_ep} ↔ HDF5 demo {did} (T={T}, {version})")
    print(f"cot_frame == hdf5_frame: {is_v2}")

    # Load LeRobot video
    lr_vid = (COT_ROOT / f"{suite}_no_noops_1.0.0_lerobot" / "videos" / "chunk-000"
              / "observation.images.image" / f"episode_{lr_ep:06d}.mp4")
    lr_reader = imageio.get_reader(str(lr_vid)) if lr_vid.exists() else None
    lr_T = lr_reader.count_frames() if lr_reader else 0
    print(f"LeRobot video: {lr_vid.exists()}, {lr_T} frames")

    # Load CoT annotations
    cot_steps = {}
    annot_path = (COT_ROOT / f"{suite}_no_noops_1.0.0_lerobot"
                  / "annotations" / "episode_dense_captions_full.jsonl")
    if annot_path.exists():
        with open(annot_path) as f:
            for line in f:
                ep = json.loads(line)
                if ep["episode_index"] == lr_ep:
                    cot_steps = ep.get("steps", {})
                    break

    # Load HDF5 NPZ
    npz = np.load(npz_path)
    T_h5 = npz["rgb_agentview"].shape[0]

    # Get key frames: subtask changes + gripper changes
    key_frames = set()
    prev_sub, prev_grip = None, None
    for s_str in sorted(cot_steps.keys(), key=int):
        s = int(s_str); info = cot_steps[s_str]
        sub = info.get("subtask",""); grip = info.get("gripper_state",-1)
        if sub != prev_sub or grip != prev_grip:
            key_frames.add(s)
        prev_sub, prev_grip = sub, grip
    # Add evenly sampled
    for t in range(0, T, max(1, T//8)):
        key_frames.add(t)
    key_frames = sorted(key_frames)[:20]

    out_dir = OUT / suite / f"task_{tid:02d}" / f"demo_{did:06d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    for cf in key_frames:
        if cf >= T or cf >= T_h5: continue
        h5f = cf if is_v2 else cf  # for v2, cot_frame == hdf5_frame

        # LeRobot RGB
        lr_rgb = lr_reader.get_data(cf) if lr_reader and cf < lr_T else np.zeros((256,256,3), dtype=np.uint8)

        # HDF5 RGB
        h5_rgb = npz["rgb_agentview"][h5f].copy()

        # HDF5 mask (rebuild from seg using union of all objs)
        seg = npz["seg_agentview"][h5f]
        mask = np.zeros(seg.shape[:2], dtype=bool)
        for obj in objs:
            sid = inst.get(obj);
            if sid is not None: mask |= (seg == sid)

        # Four-panel
        H = 256
        panel = np.zeros((H*2+60, H*2, 3), dtype=np.uint8)
        # Top-left: LeRobot
        lr_s = np.array(Image.fromarray(lr_rgb).resize((H,H)))
        panel[:H,:H] = lr_s
        # Top-right: HDF5
        h5_s = np.array(Image.fromarray(h5_rgb).resize((H,H)))
        panel[:H,H:] = h5_s
        # Bottom-left: mask overlay
        m_s = np.array(Image.fromarray((mask*255).astype(np.uint8)).resize((H,H)))
        overlay = h5_s.copy().astype(float)
        mb = m_s > 128
        overlay[mb,1]=np.clip(overlay[mb,1]+120,0,255); overlay[mb,0]*=0.3; overlay[mb,2]*=0.3
        panel[H:2*H,:H] = overlay.astype(np.uint8)
        # Bottom-right: info
        info_lines = [
            f"suite={suite} task={tid} demo={did}",
            f"lerobot_ep={lr_ep} hdf5_demo={did}",
            f"cot_frame={cf} hdf5_frame={h5f}",
            f"T_cot={T} T_hdf5={T_h5} cot==h5: {is_v2}",
        ]
        if str(cf) in cot_steps:
            si = cot_steps[str(cf)]
            info_lines.append(f"grip: {si.get('gripper_state','?')}")
            info_lines.append(f"sub: {si.get('subtask','')[:80]}")
        info_img = np.zeros((H,H,3), dtype=np.uint8)
        ip = Image.fromarray(info_img); d = ImageDraw.Draw(ip)
        for i, line in enumerate(info_lines):
            d.text((10,10+i*25), line, fill=(255,255,255))
        panel[H:2*H,H:] = np.array(ip)

        # Labels
        lb = np.zeros((60,H*2,3), dtype=np.uint8)
        lp = Image.fromarray(lb); ld = ImageDraw.Draw(lp)
        ld.text((10,2),"LeRobot RGB", fill=(255,200,0))
        ld.text((H+10,2),"HDF5 RGB", fill=(255,200,0))
        ld.text((10,32),"HDF5 Mask", fill=(0,255,0))
        ld.text((H+10,32),"Info", fill=(200,200,200))
        full = np.vstack([np.array(lp)[:30], panel[:H], np.array(lp)[30:], panel[H:]])

        fname = f"frame_{cf:06d}_lr{lr_ep}_h5d{did}.png"
        imageio.imwrite(str(out_dir/fname), full)
        si = cot_steps.get(str(cf), {})
        print(f"  CoT{cf:4d}→H5{h5f:4d}: grip={si.get('gripper_state','?')}  \"{si.get('subtask','')[:55]}\"")

    npz.close()
    if lr_reader: lr_reader.close()
    print(f"\nSaved: {out_dir}")
    print("Top-left=LeRobot, Top-right=HDF5. Same scene? → alignment correct.")


if __name__ == "__main__":
    main()
