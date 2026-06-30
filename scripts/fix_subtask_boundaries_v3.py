#!/usr/bin/env python
"""
Fix subtask boundaries V3: grip-based subtask_end_idx + CoT text correction.
============================================================================
V2 fixed subtask_end_idx only — goal mask now points to correct frame,
but the CoT text (subtask, reasoning, relation) at transition frames still
references the NEXT object, causing goal mask to include too many objects.

V3 additionally fixes CoT text: when grip is still 0 (holding previous object)
but subtask text already says "reach towards <next>", replace the text with
the PREVIOUS subtask's text. This ensures dynamic mask filtering produces
the correct mask (only current object, not both).

Algorithm:
  1. For each demo, load CoT annotation step-level data
  2. Find grip 0→1 transitions (release points)
  3. For frames BEFORE the release where subtask is "reach/approach <next obj>":
     - Replace cot_subtask, cot_text, cot_text_transition with previous subtask's
     - Fix relation_label to match the corrected subtask
  4. Also fix subtask_end_idx (same as V2)

Output: spatial_lara_libero_index_cot_transition_all_fixed_v3.jsonl
"""

import argparse, json, os, re, sys
from pathlib import Path
from collections import defaultdict

_THIS = Path(__file__).resolve()
_REPO = _THIS.parents[1]

INDEX_IN = str(_REPO / "output" / "spatial_lara_libero_no_noops" / "spatial_lara_libero_index_cot_transition_all.jsonl")
INDEX_OUT = str(_REPO / "output" / "spatial_lara_libero_no_noops" / "spatial_lara_libero_index_cot_transition_all_fixed_v3.jsonl")
COT_ROOT = os.environ.get("LEROBOT_ROOT", str(_REPO.parent / "datasets" / "lovejuly" / "libero_lerobot_all"))


def load_cot_data(cot_root):
    """Load per-step {subtask, gripper_state} for all CoT episodes."""
    data = {}  # (suite, cot_ep) → {step: {subtask, grip}}
    for suite in ["libero_10", "libero_spatial", "libero_object", "libero_goal"]:
        cot_path = Path(cot_root) / f"{suite}_no_noops_1.0.0_lerobot" / "annotations" / "episode_dense_captions_full.jsonl"
        if not cot_path.exists():
            continue
        with open(cot_path) as f:
            for line in f:
                ann = json.loads(line)
                ep = ann["episode_index"]
                step_data = {}
                for step_str, info in ann.get("steps", {}).items():
                    step_data[int(step_str)] = {
                        "subtask": info.get("subtask", ""),
                        "grip": info.get("gripper_state", -1),
                        "reasoning": info.get("reasoning", ""),
                    }
                data[(suite, ep)] = step_data
    print(f"  Loaded CoT data for {len(data)} episodes")
    return data


def classify_relation(subtask_text: str) -> str:
    """Simple relation classifier matching build_transition_index.py logic."""
    t = subtask_text.lower()
    if any(w in t for w in ["grasp", "pick up", "lift", "grip"]):
        return "grasp_object"
    if any(w in t for w in ["reach", "approach", "move toward", "move closer"]):
        return "approach_object"
    if any(w in t for w in ["put into", "place into", "inside"]):
        return "place_inside"
    if any(w in t for w in ["put on", "place on", "put the", "place the", "on top"]):
        return "place_on_top"
    if any(w in t for w in ["open", "close", "pull"]):
        return "open_articulated_object"
    if any(w in t for w in ["press", "push", "turn on", "turn off"]):
        return "object_moves_toward_target"
    return "no_salient_change"


_RELATION_ID_MAP = {
    "approach_object": 0, "grasp_object": 1, "release_object": 2,
    "place_inside": 3, "place_on_top": 4, "open_articulated_object": 5,
    "object_moves_toward_target": 6, "no_salient_change": 7,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    # ── Load CoT data ────────────────────────────────────────
    print("Loading CoT data...")
    cot_data = load_cot_data(COT_ROOT)

    # ── Load entries ─────────────────────────────────────────
    print(f"Loading index: {INDEX_IN}")
    entries = []
    with open(INDEX_IN) as f:
        for line in f:
            entries.append(json.loads(line))
    print(f"  {len(entries)} entries")

    # ── Group by demo ────────────────────────────────────────
    groups = defaultdict(list)
    for i, e in enumerate(entries):
        key = (e["suite"], e["task_id"], e["demo_id"])
        groups[key].append((i, e))

    total_endidx_fixes = 0
    total_text_fixes = 0
    stats = defaultdict(lambda: {"endidx": 0, "text": 0, "demos": set()})

    for key, group in sorted(groups.items()):
        suite, task_id, demo_id = key
        group.sort(key=lambda x: x[1]["cot_frame_idx"])

        cot_ep = group[0][1].get("cot_episode_id", -1)
        step_data = cot_data.get((suite, cot_ep), {})
        if not step_data:
            continue

        # ── Map frames to CoT step data ──────────────────────
        sorted_steps = sorted(step_data.keys())

        def get_step_info(cot_frame):
            """Get {subtask, grip, reasoning} at or before cot_frame."""
            info = {"subtask": "", "grip": -1, "reasoning": ""}
            for s in sorted_steps:
                if s <= cot_frame:
                    info = step_data[s]
                else:
                    break
            return info

        # ── Find grip 0→1 transitions (release points) ───────
        prev_grip = -1
        release_frames = []  # frames where grip changes 0→1
        for cf in sorted(set(e["cot_frame_idx"] for _, e in group)):
            info = get_step_info(cf)
            g = info["grip"]
            if g == 1 and prev_grip == 0:
                release_frames.append(cf)
            prev_grip = g

        # ── Fix subtask_end_idx (V2) + CoT text (V3) ─────────
        prev_goal = None
        prev_subtask_text = ""  # track the "real" subtask before boundary
        prev_cot_original = ""
        prev_cot_transition = ""

        for j, (idx, e) in enumerate(group):
            goal = e["subtask_end_idx"]
            cf = e["cot_frame_idx"]
            info = get_step_info(cf)
            cur_subtask = info["subtask"]
            cur_grip = info["grip"]

            if prev_goal is not None and goal != prev_goal:
                # ── V2: fix subtask_end_idx ──────────────────
                prev_grip_at_boundary = get_step_info(cf - 1)["grip"] if cf > 0 else -1
                if cur_grip == 0 and prev_grip_at_boundary == 0:
                    # Find next grip 0→1 (release)
                    new_boundary = None
                    for rf in release_frames:
                        if rf > cf:
                            new_boundary = rf
                            break
                    if new_boundary is not None:
                        for k in range(j, len(group)):
                            k_idx, k_e = group[k]
                            k_cf = k_e["cot_frame_idx"]
                            if k_cf >= new_boundary:
                                break
                            if k_e["subtask_end_idx"] == goal:
                                if "_original_subtask_end_idx" not in entries[k_idx]:
                                    entries[k_idx]["_original_subtask_end_idx"] = entries[k_idx]["subtask_end_idx"]
                                entries[k_idx]["subtask_end_idx"] = new_boundary
                                entries[k_idx]["_boundary_fixed_v3"] = True
                                total_endidx_fixes += 1
                                stats[(suite, task_id)]["endidx"] += 1
                                stats[(suite, task_id)]["demos"].add(demo_id)

                # Save the correct subtask text (before boundary)
                prev_subtask_text = get_step_info(cf - 1)["subtask"] if cf > 0 else ""
                prev_cot_original = ""
                prev_cot_transition = ""
                # Get the last entry's CoT text before the boundary
                if j > 0:
                    prev_entry = entries[group[j-1][0]]
                    prev_cot_original = prev_entry.get("cot_text_original", "")
                    prev_cot_transition = prev_entry.get("cot_text_transition", "")

            # ── V3: fix CoT text in transition zone ───────────
            # If grip=0 (still holding) and current subtask mentions "reach/approach"
            # AND we have a valid previous subtask, replace the CoT text
            if (cur_grip == 0
                and prev_subtask_text
                and ("reach" in cur_subtask.lower() or "approach" in cur_subtask.lower())
                and ("put" in prev_subtask_text.lower()
                     or "place" in prev_subtask_text.lower()
                     or "grasp" in prev_subtask_text.lower())):

                # Fix cot_text_original: replace subtask part
                old_cot_original = entries[idx].get("cot_text_original", "")
                old_cot_transition = entries[idx].get("cot_text_transition", "")
                old_expected = entries[idx].get("expected_spatial_transition", "")
                old_relation = entries[idx].get("relation_label", "")

                # Build corrected CoT text using previous subtask
                prev_reasoning = get_step_info(cf)["reasoning"]
                new_cot_original = f"Subtask: {prev_subtask_text} Reasoning: {prev_reasoning}"

                # Fix relation
                new_relation = classify_relation(prev_subtask_text)
                new_relation_id = _RELATION_ID_MAP.get(new_relation, 7)

                # Fix expected_spatial_transition based on prev subtask
                new_expected = old_expected  # keep if reasonable

                # Fix cot_text_transition
                if prev_cot_transition:
                    new_cot_transition = prev_cot_transition
                else:
                    new_cot_transition = new_cot_original

                # Apply fixes (preserve originals)
                if "_original_cot_text_original" not in entries[idx]:
                    entries[idx]["_original_cot_text_original"] = old_cot_original
                if "_original_cot_text_transition" not in entries[idx]:
                    entries[idx]["_original_cot_text_transition"] = old_cot_transition

                entries[idx]["cot_text_original"] = new_cot_original
                entries[idx]["cot_text_transition"] = new_cot_transition
                entries[idx]["relation_label"] = new_relation
                entries[idx]["relation_label_id"] = new_relation_id
                entries[idx]["_cot_text_fixed_v3"] = True

                total_text_fixes += 1
                stats[(suite, task_id)]["text"] += 1

            # Update tracking
            if goal != prev_goal:
                prev_goal = goal
                # After boundary, track current subtask for next transition
                if j < len(group):
                    pass  # handled at next boundary

    # ── Report ───────────────────────────────────────────────
    print(f"\n=== V3 Fix Summary ===")
    print(f"  subtask_end_idx fixes: {total_endidx_fixes}")
    print(f"  CoT text fixes:       {total_text_fixes}")
    print(f"  Affected demos:        {sum(1 for v in stats.values() if v['demos'])}")
    for (suite, tid), info in sorted(stats.items()):
        if info["endidx"] > 0 or info["text"] > 0:
            print(f"  {suite} task_{tid}: endidx={info['endidx']} text={info['text']} demos={len(info['demos'])}")

    if args.dry_run:
        print("\n[Dry run] No changes written.")
        return

    # ── Write ────────────────────────────────────────────────
    print(f"\nWriting: {INDEX_OUT}")
    with open(INDEX_OUT, "w") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")
    print("Done.")
    print(f"\nUse: --data.index_path={INDEX_OUT}")


if __name__ == "__main__":
    main()
