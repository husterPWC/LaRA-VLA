#!/usr/bin/env python
"""
Add fixed-τ (physical time) future index to V3 JSONL.
======================================================
Current hdf5_future_idx is CoT-semantic future (alignment-based),
which may have inconsistent frame gaps across samples. LaWAM stresses
fixed physical intervals for stable latent subgoal learning.

This script adds hdf5_tau_future_idx at a configurable fixed offset
(e.g. 0.4s = 8 frames at 20fps), clamped to subtask_end_idx and
episode length.

New fields added:
    hdf5_tau_future_idx   int   fixed-τ future frame index
    tau_future_gap        int   actual gap (tau_idx - cur_idx)
    tau_future_valid      bool  gap >= min_gap AND doesn't cross subtask end

Usage:
    python tools/add_tau_future_idx.py

    # Custom τ:
    python tools/add_tau_future_idx.py --tau 0.8 --fps 20 --min-gap 4

    # Single suite for testing:
    python tools/add_tau_future_idx.py --suite libero_spatial
"""

import argparse, json, os, sys
from pathlib import Path
from collections import defaultdict
import numpy as np

_THIS = Path(__file__).resolve()
_REPO = _THIS.parents[1] if _THIS.parents[1].name != "tools" else _THIS.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

INDEX_V3 = str(_REPO / "output" / "spatial_lara_libero_no_noops" /
               "spatial_lara_libero_index_cot_transition_all_fixed_v3.jsonl")
INDEX_V4 = str(_REPO / "output" / "spatial_lara_libero_no_noops" /
               "spatial_lara_libero_index_cot_transition_all_fixed_v4_tau.jsonl")
SPATIAL = str(_REPO / "output" / "spatial_lara_libero")


def compute_tau_future(
    hdf5_frame_idx: int,
    subtask_end_idx: int,
    episode_len: int,
    offset: int,
    min_gap: int,
) -> dict:
    """
    Compute fixed-τ future frame index.

    Args:
        hdf5_frame_idx:  current frame index
        subtask_end_idx: subtask boundary (don't cross)
        episode_len:     total frames in episode
        offset:          fixed frame offset = round(fps * tau)
        min_gap:         minimum gap for valid flag

    Returns:
        dict with tau_future_idx, gap, valid
    """
    raw = hdf5_frame_idx + offset
    # Clamp: don't cross subtask boundary or episode end
    tau_idx = min(raw, subtask_end_idx, episode_len - 1)
    gap = tau_idx - hdf5_frame_idx
    valid = gap >= min_gap
    return {
        "hdf5_tau_future_idx": int(tau_idx),
        "tau_future_gap": int(gap),
        "tau_future_valid": bool(valid),
    }


def load_episode_lengths(spatial_root: str, entries: list) -> dict:
    """Pre-scan NPZ file lengths (cached per episode path)."""
    lengths = {}
    unique_eps = set(e["episode_path"] for e in entries)
    for ep_path in sorted(unique_eps):
        full = Path(spatial_root) / ep_path
        try:
            data = np.load(full)
            lengths[ep_path] = int(data["rgb_agentview"].shape[0])
            data.close()
        except Exception as exc:
            print(f"  WARNING: Cannot read {ep_path}: {exc}")
            lengths[ep_path] = 0
    return lengths


def main():
    parser = argparse.ArgumentParser(description="Add fixed-τ future index to V3 JSONL")
    parser.add_argument("--input", type=str, default=INDEX_V3,
                        help="Input V3 JSONL path")
    parser.add_argument("--output", type=str, default=INDEX_V4,
                        help="Output V4 JSONL path")
    parser.add_argument("--spatial-root", type=str, default=SPATIAL,
                        help="NPZ data root")
    parser.add_argument("--fps", type=float, default=20.0,
                        help="LIBERO frame rate (default 20Hz)")
    parser.add_argument("--tau", type=float, default=0.4,
                        help="Physical time interval in seconds (default 0.4s)")
    parser.add_argument("--min-gap", type=int, default=2,
                        help="Minimum frame gap for tau_future_valid=True")
    parser.add_argument("--suite", type=str, default=None,
                        help="Filter to single suite (e.g. libero_spatial)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print stats only, don't write output")
    args = parser.parse_args()

    offset = int(round(args.fps * args.tau))

    print("=" * 60)
    print("Add fixed-τ future index (Step 4)")
    print(f"  Input:  {args.input}")
    print(f"  Output: {args.output}")
    print(f"  FPS: {args.fps}  τ: {args.tau}s  offset: {offset} frames")
    print(f"  min_gap: {args.min_gap}")
    if args.suite:
        print(f"  Suite filter: {args.suite}")
    print("=" * 60)

    # ── Load V3 entries ─────────────────────────────────────
    print("\nLoading V3 entries...")
    entries = []
    with open(args.input, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))

    if args.suite:
        entries = [e for e in entries if e.get("suite") == args.suite]

    print(f"  Total: {len(entries)} entries")

    # ── Load episode lengths ────────────────────────────────
    print("Loading episode lengths...")
    ep_lengths = load_episode_lengths(args.spatial_root, entries)
    print(f"  Loaded {len(ep_lengths)} episode lengths")

    # ── Compute tau future for each entry ───────────────────
    print("Computing tau future indices...")
    stats = defaultdict(int)
    new_entries = []
    gap_dist = []

    for entry in entries:
        ep_len = ep_lengths.get(entry["episode_path"], 0)
        if ep_len <= 0:
            # Fallback: use subtask_end_idx as estimate
            ep_len = entry.get("subtask_end_idx", 0) + 10

        tau_info = compute_tau_future(
            hdf5_frame_idx=entry["hdf5_frame_idx"],
            subtask_end_idx=entry["subtask_end_idx"],
            episode_len=ep_len,
            offset=offset,
            min_gap=args.min_gap,
        )

        gap_dist.append(tau_info["tau_future_gap"])
        stats["total"] += 1
        if tau_info["tau_future_valid"]:
            stats["valid"] += 1
        else:
            stats["invalid"] += 1
        if tau_info["tau_future_gap"] == offset:
            stats["exact_offset"] += 1
        if tau_info["tau_future_gap"] < offset:
            stats["clamped"] += 1

        new_entry = dict(entry)
        new_entry["hdf5_tau_future_idx"] = tau_info["hdf5_tau_future_idx"]
        new_entry["tau_future_gap"] = tau_info["tau_future_gap"]
        new_entry["tau_future_valid"] = tau_info["tau_future_valid"]
        new_entries.append(new_entry)

    # ── Stats ────────────────────────────────────────────────
    gap_arr = np.array(gap_dist)
    print(f"\n{'='*60}")
    print("Statistics")
    print(f"{'='*60}")
    print(f"  Total entries:        {stats['total']}")
    print(f"  Valid (gap >= {args.min_gap}):  {stats['valid']} ({100*stats['valid']/max(1,stats['total']):.1f}%)")
    print(f"  Invalid (too short):  {stats['invalid']} ({100*stats['invalid']/max(1,stats['total']):.1f}%)")
    print(f"  Exact offset ({offset}):      {stats['exact_offset']} ({100*stats['exact_offset']/max(1,stats['total']):.1f}%)")
    print(f"  Clamped (<{offset}):          {stats['clamped']} ({100*stats['clamped']/max(1,stats['total']):.1f}%)")
    print(f"  gap mean: {gap_arr.mean():.1f}  median: {np.median(gap_arr):.1f}")
    print(f"  gap min: {gap_arr.min():.0f}  max: {gap_arr.max():.0f}")

    # Per-suite breakdown
    suite_stats = defaultdict(lambda: {"total": 0, "valid": 0, "exact": 0, "clamped": 0})
    for e in new_entries:
        s = e["suite"]
        suite_stats[s]["total"] += 1
        if e["tau_future_valid"]:
            suite_stats[s]["valid"] += 1
        if e["tau_future_gap"] == offset:
            suite_stats[s]["exact"] += 1
        if e["tau_future_gap"] < offset:
            suite_stats[s]["clamped"] += 1

    print(f"\n  Per-suite validity rate:")
    for s in sorted(suite_stats.keys()):
        st = suite_stats[s]
        vr = 100 * st["valid"] / max(1, st["total"])
        print(f"    {s}: {st['valid']}/{st['total']} = {vr:.1f}% valid  "
              f"(exact={st['exact']}, clamped={st['clamped']})")

    # ── Write output ─────────────────────────────────────────
    if args.dry_run:
        print(f"\n  DRY RUN — no output written.")
    else:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            for e in new_entries:
                f.write(json.dumps(e) + "\n")
        print(f"\n  ✅ Wrote {len(new_entries)} entries to {output_path}")
        print(f"  New fields: hdf5_tau_future_idx, tau_future_gap, tau_future_valid")


if __name__ == "__main__":
    main()
