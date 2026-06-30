#!/usr/bin/env python
"""
Fix subtask boundary issues in the transition index.
=====================================================
LaRA-VLA CoT annotations have systematically truncated "place" / "grasp"
subtasks (1-2 frames), causing subtask boundaries to shift too early.

Fix: merge short subtask segments (≤3 frames) backward into the preceding
segment, adjusting subtask_end_idx accordingly.

Strategy (per demo):
  1. Read all entries for the demo, sorted by frame
  2. Identify subtask segments (contiguous frames with same subtask_end_idx)
  3. For short segments (≤3 frames):
     - If preceded by a longer segment: merge by extending prev segment's end
     - Adjust subtask_end_idx for all affected frames
  4. Rewrite entries with corrected subtask_end_idx

Output: new index with _original_subtask_end_idx preserved for comparison.

Usage:
    python scripts/fix_subtask_boundaries.py
    python scripts/fix_subtask_boundaries.py --dry-run  # preview changes only
"""

import argparse, json, os, sys
from pathlib import Path
from collections import defaultdict

_THIS = Path(__file__).resolve()
_REPO = _THIS.parents[1]

INDEX_IN = str(_REPO / "output" / "spatial_lara_libero_no_noops" / "spatial_lara_libero_index_cot_transition_all.jsonl")
INDEX_OUT = str(_REPO / "output" / "spatial_lara_libero_no_noops" / "spatial_lara_libero_index_cot_transition_all_fixed.jsonl")
REPORT_PATH = str(_REPO / "output" / "audit_libero10" / "boundary_fix_report.txt")

MIN_SEGMENT_LEN = 3  # Segments ≤ this length get merged


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--min-len", type=int, default=MIN_SEGMENT_LEN)
    args = parser.parse_args()

    # ── Load all entries ──────────────────────────────────────
    print(f"Loading index: {INDEX_IN}")
    entries = []
    with open(INDEX_IN) as f:
        for line in f:
            entries.append(json.loads(line))
    print(f"  {len(entries)} total entries")

    # ── Group by (suite, task_id, demo_id) ────────────────────
    groups = defaultdict(list)
    for i, e in enumerate(entries):
        key = (e["suite"], e["task_id"], e["demo_id"])
        groups[key].append((i, e))

    # ── Process each demo ─────────────────────────────────────
    total_fixes = 0
    fix_details = []

    for key, group in sorted(groups.items()):
        suite, task_id, demo_id = key
        group.sort(key=lambda x: x[1]["cot_frame_idx"])

        # Build segments: groups of entries with same subtask_end_idx
        segments = []  # list of [(start_idx, end_idx), subtask_end_idx, frame_range]
        seg_start = 0
        for j in range(1, len(group)):
            if group[j][1]["subtask_end_idx"] != group[seg_start][1]["subtask_end_idx"]:
                segments.append((seg_start, j - 1))
                seg_start = j
        segments.append((seg_start, len(group) - 1))

        # Fix: merge short segments backward
        fixed = False
        j = 0
        while j < len(segments):
            s_start, s_end = segments[j]
            seg_len = s_end - s_start + 1
            frame_start = group[s_start][1]["cot_frame_idx"]
            frame_end = group[s_end][1]["cot_frame_idx"]
            goal_idx = group[s_start][1]["subtask_end_idx"]

            if seg_len <= args.min_len and j > 0 and not (j == len(segments) - 1):
                # This segment is too short. Merge it backward:
                # extend the PREVIOUS segment to cover these frames,
                # and set these frames' subtask_end_idx to the next segment's goal.
                prev_start, prev_end = segments[j - 1]
                next_start = segments[j + 1][0] if j + 1 < len(segments) else s_end + 1
                prev_goal = group[prev_start][1]["subtask_end_idx"]
                next_goal = group[next_start][1]["subtask_end_idx"] if j + 1 < len(segments) else goal_idx

                # Extend previous segment to include this short segment
                segments[j - 1] = (prev_start, s_end)

                # Update entries in the short segment:
                # Give them the NEXT non-short segment's goal
                for k in range(s_start, s_end + 1):
                    idx, e = group[k]
                    old_goal = e["subtask_end_idx"]
                    new_goal = next_goal
                    if old_goal != new_goal:
                        entries[idx]["_original_subtask_end_idx"] = old_goal
                        entries[idx]["subtask_end_idx"] = new_goal
                        entries[idx]["_boundary_fixed"] = True
                        total_fixes += 1

                # Remove the merged segment
                segments.pop(j)
                fixed = True
                continue
            j += 1

        if fixed:
            # Rebuild segments after modifications
            pass

    # ── Per-suite/per-task fix detail ──────────────────────────
    # Collect stats
    suite_task_fixes = defaultdict(lambda: {"total": 0, "demos": set()})
    for i, e in enumerate(entries):
        if e.get("_boundary_fixed"):
            key = (e["suite"], e["task_id"])
            suite_task_fixes[key]["total"] += 1
            suite_task_fixes[key]["demos"].add(e["demo_id"])

    print(f"\n=== Fix Summary ===")
    print(f"  Total entries fixed: {total_fixes}")
    print(f"  Affected demos: {sum(len(v['demos']) for v in suite_task_fixes.values())}")
    for (suite, tid), info in sorted(suite_task_fixes.items()):
        print(f"  {suite} task_{tid}: {info['total']} entries in {len(info['demos'])} demos")

    if args.dry_run:
        print("\n[Dry run] No changes written.")
        return

    # ── Write fixed index ─────────────────────────────────────
    print(f"\nWriting fixed index: {INDEX_OUT}")
    with open(INDEX_OUT, "w") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")
    print(f"  {len(entries)} entries written")

    # ── Save report ───────────────────────────────────────────
    Path(REPORT_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        f.write(f"Subtask Boundary Fix Report\n")
        f.write(f"{'='*60}\n")
        f.write(f"Min segment length threshold: {args.min_len}\n")
        f.write(f"Total entries fixed: {total_fixes}\n\n")
        for (suite, tid), info in sorted(suite_task_fixes.items()):
            f.write(f"{suite} task_{tid}: {info['total']} entries in {len(info['demos'])} demos\n")
            f.write(f"  Demos: {sorted(info['demos'])}\n")

    print(f"\nReport: {REPORT_PATH}")
    print(f"\nNext: update config to use the fixed index:")
    print(f"  --data.index_path={INDEX_OUT}")


if __name__ == "__main__":
    main()
