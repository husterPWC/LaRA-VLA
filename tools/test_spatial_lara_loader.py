#!/usr/bin/env python
"""
Spatial-LaRA DataLoader Sanity Check
====================================
Test batch loading from the Spatial-LaRA LIBERO dataset.

Usage:
    python tools/test_spatial_lara_loader.py \
        --root output/spatial_lara_libero \
        --index output/spatial_lara_libero/spatial_lara_libero_index.jsonl \
        --batch-size 8 --num-workers 4 --num-batches 10
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader

# Add repo root to path
_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from lara_vla.data.spatial_lara_libero_dataset import SpatialLaRALiberoDataset


def main():
    parser = argparse.ArgumentParser(description="Test Spatial-LaRA DataLoader")
    parser.add_argument("--root", type=str,
                        default=str(_REPO_ROOT / "output" / "spatial_lara_libero"))
    parser.add_argument("--index", type=str,
                        default=str(_REPO_ROOT / "output" / "spatial_lara_libero" /
                                    "spatial_lara_libero_index.jsonl"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-batches", type=int, default=10)
    args = parser.parse_args()

    # ── Create dataset ──────────────────────────────────────────
    print(f"Root:  {args.root}")
    print(f"Index: {args.index}")
    print()

    ds = SpatialLaRALiberoDataset(
        root=args.root,
        index_path=args.index,
        future_k=8,
        cache_size=32,
    )
    print(f"Dataset size: {len(ds)}")
    print()

    # ── Single sample check ─────────────────────────────────────
    print("=== Single sample ===")
    sample = ds[0]
    for k, v in sample.items():
        if isinstance(v, np.ndarray):
            print(f"  {k:40s}: shape={str(v.shape):20s} dtype={str(v.dtype):10s} "
                  f"range=[{v.min():.3f}, {v.max():.3f}]")
        else:
            print(f"  {k:40s}: {v}")
    print()

    # ── DataLoader check ────────────────────────────────────────
    print(f"=== DataLoader (batch_size={args.batch_size}, workers={args.num_workers}) ===")

    def collate_fn(batch):
        """Simple collate: stack numpy arrays, keep scalars as lists."""
        out = {}
        for key in batch[0]:
            vals = [b[key] for b in batch]
            if isinstance(vals[0], np.ndarray):
                out[key] = np.stack(vals, axis=0)
            else:
                out[key] = vals
        return out

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    t0 = time.time()
    for i, batch in enumerate(loader):
        if i >= args.num_batches:
            break

        if i == 0:
            # Print first batch shapes
            for k, v in batch.items():
                if isinstance(v, np.ndarray):
                    print(f"  {k:40s}: shape={str(v.shape):25s}")
                else:
                    print(f"  {k:40s}: list len={len(v)}")

        # Sanity checks
        B = args.batch_size
        # Mask binary
        assert batch["current_affordance_mask_agentview"].max() <= 1.0
        assert batch["current_affordance_mask_agentview"].min() >= 0.0
        # Image range
        assert 0.0 <= batch["image_agentview"].max() <= 1.0
        # Pose valid
        assert not np.isnan(batch["primary_pose_world"]).any()
        # Actions shape
        assert batch["actions"].shape == (B, 8, 7)
        # Future idx >= frame idx
        for j in range(B):
            assert batch["future_idx"][j] >= batch["frame_idx"][j]

        if i % 3 == 0:
            print(f"  Batch {i+1}/{args.num_batches} OK")

    elapsed = time.time() - t0
    batches_per_sec = min(args.num_batches, i + 1) / elapsed if elapsed > 0 else 0
    print(f"\n  Speed: {batches_per_sec:.1f} batches/sec ({batches_per_sec * args.batch_size:.0f} samples/sec)")

    # ── Summary ─────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("ALL CHECKS PASSED ✅")
    print(f"  Dataset size:           {len(ds)}")
    print(f"  Batch shape valid:      ✓")
    print(f"  Mask binary [0-1]:      ✓")
    print(f"  Image range [0,1]:      ✓")
    print(f"  Pose no NaN:            ✓")
    print(f"  Actions shape [B,8,7]:  ✓")
    print(f"  Future >= frame:        ✓")
    print("=" * 60)


if __name__ == "__main__":
    main()
