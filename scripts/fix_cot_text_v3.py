#!/usr/bin/env python
"""
Post-process V2-fixed index: replace CoT text for transition-zone frames.
==========================================================================
For frames where subtask_end_idx was fixed (grip=0 holding phase),
replace the CoT text with the previous subtask's text to ensure
dynamic mask filtering produces correct masks.

Input:  spatial_lara_libero_index_cot_transition_all_fixed_v3.jsonl
Output: spatial_lara_libero_index_cot_transition_all_fixed_v3.jsonl (in-place)
"""

import argparse, json, os, sys
from pathlib import Path
from collections import defaultdict

_THIS = Path(__file__).resolve()
_REPO = _THIS.parents[1]

INDEX_IN = str(_REPO / "output" / "spatial_lara_libero_no_noops" / "spatial_lara_libero_index_cot_transition_all_fixed_v2.jsonl")
INDEX_OUT = str(_REPO / "output" / "spatial_lara_libero_no_noops" / "spatial_lara_libero_index_cot_transition_all_fixed_v3.jsonl")
COT_ROOT = os.environ.get("LEROBOT_ROOT", str(_REPO.parent / "datasets" / "lovejuly" / "libero_lerobot_all"))


def classify_relation(subtask_text: str) -> str:
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


_RELATION_ID = {"approach_object": 0, "grasp_object": 1, "release_object": 2,
                "place_inside": 3, "place_on_top": 4, "open_articulated_object": 5,
                "object_moves_toward_target": 6, "no_salient_change": 7}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    print(f"Loading: {INDEX_IN}")
    entries = []
    with open(INDEX_IN) as f:
        for line in f:
            entries.append(json.loads(line))
    print(f"  {len(entries)} entries")

    # Load CoT step data
    print("Loading CoT data...")
    cot_data = {}  # (suite, cot_ep) → {step: {subtask, grip, reasoning}}
    for suite in ["libero_10", "libero_spatial", "libero_object", "libero_goal"]:
        cot_path = Path(COT_ROOT) / f"{suite}_no_noops_1.0.0_lerobot" / "annotations" / "episode_dense_captions_full.jsonl"
        if not cot_path.exists():
            continue
        with open(cot_path) as f:
            for line in f:
                ann = json.loads(line)
                ep = ann["episode_index"]
                sd = {}
                for s_str, info in ann.get("steps", {}).items():
                    sd[int(s_str)] = {
                        "subtask": info.get("subtask", ""),
                        "grip": info.get("gripper_state", -1),
                        "reasoning": info.get("reasoning", ""),
                    }
                cot_data[(suite, ep)] = sd
    print(f"  {len(cot_data)} episodes")

    # Group entries by demo
    groups = defaultdict(list)
    for i, e in enumerate(entries):
        if e.get("_boundary_fixed_v2"):
            key = (e["suite"], e["task_id"], e["demo_id"])
            groups[key].append(i)

    print(f"  {len(groups)} demos with fixed entries")

    total_fixes = 0
    for key, indices in sorted(groups.items()):
        suite, task_id, demo_id = key
        # Only fix CoT text for libero_10 (dynamic mask filtering);
        # other suites use union mask and don't need text correction.
        if suite != "libero_10":
            continue
        indices.sort(key=lambda i: entries[i]["cot_frame_idx"])

        cot_ep = entries[indices[0]].get("cot_episode_id", -1)
        step_data = cot_data.get((suite, cot_ep), {})
        if not step_data:
            continue
        sorted_steps = sorted(step_data.keys())

        def get_step_info(cf):
            info = {"subtask": "", "grip": -1, "reasoning": ""}
            for s in sorted_steps:
                if s <= cf:
                    info = step_data[s]
                else:
                    break
            return info

        # For each fixed frame: find the previous subtask (before the grip-open)
        # and replace CoT text if current subtask is "reach/approach"
        for idx in indices:
            e = entries[idx]
            cf = e["cot_frame_idx"]
            info = get_step_info(cf)
            cur_sub = info["subtask"]
            cur_grip = info["grip"]

            # Find the subtask BEFORE the premature boundary
            # Backtrack to find last "put/place/grasp" subtask
            prev_sub = ""
            prev_reasoning = ""
            for s in reversed(sorted_steps):
                if s < cf:
                    sd = step_data[s]
                    sub = sd["subtask"]
                    if any(w in sub.lower() for w in ["put", "place", "grasp", "pick up"]):
                        prev_sub = sub
                        prev_reasoning = sd["reasoning"]
                        break

            if (prev_sub
                and ("reach" in cur_sub.lower() or "approach" in cur_sub.lower())):
                # Fix CoT text
                old_cot_original = e.get("cot_text_original", "")
                old_cot_transition = e.get("cot_text_transition", "")

                if "_original_cot_text_original" not in e:
                    e["_original_cot_text_original"] = old_cot_original
                if "_original_cot_text_transition" not in e:
                    e["_original_cot_text_transition"] = old_cot_transition

                new_cot_original = f"Subtask: {prev_sub} Reasoning: {prev_reasoning}"
                new_relation = classify_relation(prev_sub)

                e["cot_text_original"] = new_cot_original
                e["cot_text_transition"] = new_cot_original  # use same for transition
                e["relation_label"] = new_relation
                e["relation_label_id"] = _RELATION_ID.get(new_relation, 7)
                e["_cot_text_fixed_v3"] = True
                total_fixes += 1

    print(f"\n=== Summary ===")
    print(f"  CoT text fixes: {total_fixes}")

    if args.dry_run:
        print("\n[Dry run] No changes written.")
        return

    print(f"\nWriting: {INDEX_OUT}")
    with open(INDEX_OUT, "w") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")
    print("Done.")


if __name__ == "__main__":
    main()
