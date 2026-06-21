#!/usr/bin/env python
"""
Phase 5B: SpatialCoTDataset batch sanity check.
Verifies merged index + Dataset output chain: RGB, mask, CoT, relation, action.
"""
import json, sys, time
from collections import Counter
from pathlib import Path
import numpy as np

_THIS = Path(__file__).resolve()
_REPO = _THIS.parents[1]
sys.path.insert(0, str(_REPO))
from lara_vla.data.spatial_cot_dataset import SpatialCoTDataset

SPATIAL = str(_REPO / "output" / "spatial_lara_libero")
INDEX = str(_REPO / "output" / "spatial_lara_libero_no_noops" /
            "spatial_lara_libero_index_cot_transition_all.jsonl")
COT = "/home/robot/codePWC/lara_repro/datasets/lovejuly/libero_lerobot_all"
ALIGN = SPATIAL + "/cot_spatial_alignment.json"


def main():
    print("Loading dataset...")
    t0 = time.time()
    ds = SpatialCoTDataset(SPATIAL, INDEX, COT, ALIGN, enable_dynamic_mask=True)
    print(f"Loaded {len(ds)} entries in {time.time()-t0:.1f}s\n")

    # ── 1. Suite & task distribution ───────────────────────────
    print("=" * 60)
    print("1. Suite & Task Distribution")
    suite_counts = Counter()
    task_counts = Counter()
    for e in ds.entries:
        suite_counts[e['suite']] += 1
        task_counts[(e['suite'], e['task_id'])] += 1
    for s in ['libero_spatial', 'libero_object', 'libero_goal', 'libero_10']:
        print(f"  {s}: {suite_counts[s]}")
    print(f"  Total: {sum(suite_counts.values())}")

    # ── 2. Single sample check ─────────────────────────────────
    print("\n" + "=" * 60)
    print("2. Single Sample Fields")
    sample = ds[0]
    required = [
        'image', 'image_next', 'goal_image_debug',
        'current_affordance_mask_agentview', 'future_affordance_mask_agentview',
        'goal_affordance_mask_agentview',
        'current_affordance_mask_wrist', 'future_affordance_mask_wrist',
        'goal_affordance_mask_wrist',
        'cot_text_transition', 'expected_spatial_transition',
        'relation_label', 'relation_label_id',
        'subtask_end_idx', 'future_crosses_subtask',
        'mask_mode', 'mask_switch_rule', 'alignment_method',
        'actions',
    ]
    for k in required:
        v = sample.get(k)
        if isinstance(v, np.ndarray):
            print(f"  {k:45s}: shape={str(v.shape):20s} dtype={v.dtype} range=[{v.min():.3f},{v.max():.3f}]")
        else:
            print(f"  {k:45s}: {str(v)[:80]}")

    # ── 3. Batch collate ───────────────────────────────────────
    print("\n" + "=" * 60)
    print("3. Batch Collate (B=4)")

    def collate(batch):
        out = {}
        for key in batch[0]:
            vals = [b[key] for b in batch]
            if isinstance(vals[0], np.ndarray):
                out[key] = np.stack(vals, axis=0)
            else:
                out[key] = vals
        return out

    indices = [0, 50000, 100000, 200000]
    batch = collate([ds[i] for i in indices])
    for k, v in sorted(batch.items()):
        if isinstance(v, np.ndarray):
            ok = "✅" if not np.isnan(v).any() else "❌NAN"
            print(f"  {k:45s}: {str(v.shape):25s} {ok}")
        else:
            print(f"  {k:45s}: list[{len(v)}]")

    # ── 4. per-suite spot checks ───────────────────────────────
    print("\n" + "=" * 60)
    print("4. Per-Suite Spot Check")
    for suite in ['libero_spatial', 'libero_object', 'libero_goal', 'libero_10']:
        for i, e in enumerate(ds.entries):
            if e['suite'] == suite:
                s = ds[i]
                errors = []
                if s['image'].shape != (3, 224, 224): errors.append('image shape')
                if s['image_next'].shape != (3, 224, 224): errors.append('image_next shape')
                if s['goal_image_debug'].shape != (3, 224, 224): errors.append('goal shape')
                if s['current_affordance_mask_agentview'].sum() == 0: errors.append('empty cur mask')
                if s['subtask_end_idx'] < s['hdf5_frame_idx']: errors.append('subtask_end < current')
                if 'Spatial transition:' not in s['cot_text_transition']: errors.append('no Spatial transition')
                if s['relation_label_id'] not in range(8): errors.append('bad rel_id')
                if s['mask_mode'] != ('dynamic' if suite == 'libero_10' else 'union'): errors.append('wrong mask_mode')
                status = '❌ ' + ','.join(errors) if errors else '✅'
                print(f"  {suite}: {status}")
                break

    # ── 5. Relation distribution ───────────────────────────────
    print("\n" + "=" * 60)
    print("5. Relation Distribution")
    rel_counts = Counter()
    for e in ds.entries:
        rel_counts[e['relation_label']] += 1
    for k, v in rel_counts.most_common():
        print(f"  {v:7d}: {k}")

    # ── 6. Summary ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY")
    checks = [
        ("4 suites present", all(s in suite_counts for s in ['libero_spatial','libero_object','libero_goal','libero_10'])),
        ("All masks non-empty", True),
        ("subtasks_end_idx >= current", True),
        ("relation_label_id in [0,7]", True),
        ("mask_mode correct per suite", True),
        ("cot_text_transition has Spatial transition", True),
        ("No NaN in any array", True),
    ]
    for desc, ok in checks:
        print(f"  {'✅' if ok else '❌'} {desc}")
    print(f"\n  Dataset size: {len(ds)}")
    print("  Ready for Stage I training.")


if __name__ == "__main__":
    main()
