#!/usr/bin/env python
"""
Fix subtask boundaries using gripper state transitions (v2).
=============================================================
Root cause: LaRA-VLA CoT annotations have subtask text changes (e.g. "place X"
→ "reach Y") while gripper_state is still 0 (closed/holding). Visual inspection
confirms the robot is still holding the first object.

Fix: use gripper_state 0→1 (open) as the TRUE boundary for object handoff.
     - grip stays 0 = robot is still holding/manipulating current object
     - grip changes to 1 = robot released object, ready for next subtask

Algorithm:
  1. For each demo, load CoT annotation's per-step gripper_state
  2. Map each CoT frame to its CoT step
  3. When a subtask changes from "put/place X" to "reach Y" but grip is
     still 0, postpone the boundary: extend previous subtask_end_idx
     forward until grip actually opens (0→1).
  4. Update subtask_end_idx for all affected frames.

Output: new index with fixed subtask_end_idx

Usage:
    python scripts/fix_subtask_boundaries_v2.py --dry-run
    python scripts/fix_subtask_boundaries_v2.py
"""

import argparse, json, os, sys
from pathlib import Path
from collections import defaultdict

_THIS = Path(__file__).resolve()
_REPO = _THIS.parents[1]

INDEX_IN = str(_REPO / "output" / "spatial_lara_libero_no_noops" / "spatial_lara_libero_index_cot_transition_all.jsonl")
INDEX_OUT = str(_REPO / "output" / "spatial_lara_libero_no_noops" / "spatial_lara_libero_index_cot_transition_all_fixed_v2.jsonl")
COT_ROOT = os.environ.get("LEROBOT_ROOT", str(_REPO.parent / "datasets" / "lovejuly" / "libero_lerobot_all"))
REPORT_PATH = str(_REPO / "output" / "audit_libero10" / "boundary_fix_v2_report.txt")


def load_cot_grip_states(cot_root):
    """Load per-step gripper_state for all CoT episodes."""
    grip_map = {}  # (suite, cot_ep) → {step: gripper_state}
    for suite in ["libero_10", "libero_spatial", "libero_object", "libero_goal"]:
        cot_path = Path(cot_root) / f"{suite}_no_noops_1.0.0_lerobot" / "annotations" / "episode_dense_captions_full.jsonl"
        if not cot_path.exists():
            continue
        with open(cot_path) as f:
            for line in f:
                ann = json.loads(line)
                ep = ann["episode_index"]
                step_grips = {}
                for step_str, info in ann.get("steps", {}).items():
                    step_grips[int(step_str)] = info.get("gripper_state", -1)
                grip_map[(suite, ep)] = step_grips
    print(f"  Loaded grip states for {len(grip_map)} CoT episodes")
    return grip_map


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-gap", type=int, default=50,
                        help="Max frames to extend boundary when grip hasn't changed")
    args = parser.parse_args()

    # ── Load grip states ──────────────────────────────────────
    print("Loading CoT grip states...")
    grip_map = load_cot_grip_states(COT_ROOT)

    # ── Load entries ──────────────────────────────────────────
    print(f"Loading index: {INDEX_IN}")
    entries = []
    with open(INDEX_IN) as f:
        for line in f:
            entries.append(json.loads(line))
    print(f"  {len(entries)} entries")

    # ── Group by (suite, task_id, demo_id) ────────────────────
    groups = defaultdict(list)
    for i, e in enumerate(entries):
        key = (e["suite"], e["task_id"], e["demo_id"])
        groups[key].append((i, e))

    total_fixes = 0
    fix_demos = set()
    suite_task_fixes = defaultdict(lambda: {"entries": 0, "demos": set()})

    for key, group in sorted(groups.items()):
        suite, task_id, demo_id = key
        group.sort(key=lambda x: x[1]["cot_frame_idx"])

        # Get grip states for this episode
        cot_ep = group[0][1].get("cot_episode_id", -1)
        grip_states = grip_map.get((suite, cot_ep), {})
        if not grip_states:
            continue

        # Build per-frame grip state by mapping cot_frame_idx → cot step
        # cot_frame_idx == CoT step (identity mapping in no-noops data)
        frame_grips = {}
        for idx, e in group:
            cf = e["cot_frame_idx"]
            # Find the grip state at or before this frame
            grip = -1
            for s in sorted(grip_states.keys()):
                if s <= cf:
                    grip = grip_states[s]
                else:
                    break
            frame_grips[cf] = grip

        # Find grip transition points (0→1: released, 1→0: grasped)
        frames_sorted = sorted(frame_grips.keys())
        grip_transitions = []  # [(frame, from_grip, to_grip)]
        prev_g = -1
        for cf in frames_sorted:
            g = frame_grips[cf]
            if g != prev_g and prev_g != -1:
                grip_transitions.append((cf, prev_g, g))
            prev_g = g

        # Find subtask boundary frames where grip hasn't changed
        prev_goal = None
        fixed_this_demo = 0
        for j, (idx, e) in enumerate(group):
            goal = e["subtask_end_idx"]
            cf = e["cot_frame_idx"]

            if prev_goal is not None and goal != prev_goal:
                # Subtask boundary at cf. Check grip state.
                cur_grip = frame_grips.get(cf, -1)
                prev_grip = frame_grips.get(max(0, cf - 1), -1)

                # Scenario: subtask changes from put/place → reach, but grip
                # is still 0 (holding). The boundary is premature.
                # Find next grip 0→1 transition within max_gap frames.
                if cur_grip == 0 and prev_grip == 0:
                    # grip hasn't changed — find when it does
                    new_boundary = None
                    for t_cf, g_from, g_to in grip_transitions:
                        if t_cf > cf and g_from == 0 and g_to == 1:
                            if t_cf - cf <= args.max_gap:
                                new_boundary = t_cf
                            break

                    if new_boundary is not None:
                        # Extend the previous goal to new_boundary
                        # This means frames from cf to new_boundary-1
                        # should have subtask_end_idx = new_boundary
                        # (they're still in the previous subtask)
                        new_goal = new_boundary
                        for k in range(j, len(group)):
                            k_idx, k_e = group[k]
                            k_cf = k_e["cot_frame_idx"]
                            if k_cf >= new_boundary:
                                break
                            if k_e["subtask_end_idx"] == goal:
                                old_goal = k_e["subtask_end_idx"]
                                entries[k_idx]["_original_subtask_end_idx"] = old_goal
                                entries[k_idx]["subtask_end_idx"] = new_goal
                                entries[k_idx]["_boundary_fixed_v2"] = True
                                fixed_this_demo += 1
                                total_fixes += 1

            prev_goal = goal

        if fixed_this_demo > 0:
            fix_demos.add(key)
            skey = (suite, task_id)
            suite_task_fixes[skey]["entries"] += fixed_this_demo
            suite_task_fixes[skey]["demos"].add(demo_id)

    # ── Report ────────────────────────────────────────────────
    print(f"\n=== Fix Summary ===")
    print(f"  Total entries fixed: {total_fixes}")
    print(f"  Affected demos: {len(fix_demos)}")
    for (suite, tid), info in sorted(suite_task_fixes.items()):
        if info["entries"] > 0:
            print(f"  {suite} task_{tid}: {info['entries']} entries in {len(info['demos'])} demos")

    if args.dry_run:
        print("\n[Dry run] No changes written.")
        return

    # ── Write ─────────────────────────────────────────────────
    print(f"\nWriting fixed index: {INDEX_OUT}")
    with open(INDEX_OUT, "w") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")
    print(f"  Done.")

    # Save report
    Path(REPORT_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        f.write(f"Boundary Fix Report v2 (grip-state based)\n")
        f.write(f"{'='*60}\n")
        f.write(f"Total entries fixed: {total_fixes}\n")
        f.write(f"Affected demos: {len(fix_demos)}\n\n")
        for (suite, tid), info in sorted(suite_task_fixes.items()):
            if info["entries"] > 0:
                f.write(f"{suite} task_{tid}: {info['entries']} entries in {len(info['demos'])} demos\n")

    print(f"\nDone. Use this index for training:")
    print(f"  --data.index_path={INDEX_OUT}")


if __name__ == "__main__":
    main()
