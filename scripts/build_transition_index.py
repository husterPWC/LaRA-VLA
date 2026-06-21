#!/usr/bin/env python
"""
Build unified transition index with subtask_end_idx, relation_label,
expected_spatial_transition, and cot_text_transition.
Merges all 4 suites.
"""
import json, os, sys
from collections import defaultdict, Counter
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_LARA_REPRO = _REPO.parent
SPATIAL = _REPO / 'output' / 'spatial_lara_libero'
LEROBOT = Path(os.environ.get('LEROBOT_ROOT', str(_LARA_REPRO / 'datasets/lovejuly/libero_lerobot_all')))
OUT = SPATIAL / 'spatial_lara_libero_index_cot_transition_all.jsonl'
SUITES = ['libero_spatial', 'libero_object', 'libero_goal', 'libero_10']

# ── Relation label mapping ─────────────────────────────────────
RELATION_LABELS = {
    'approach_object': 0,
    'grasp_object': 1,
    'release_object': 2,
    'place_inside': 3,
    'place_on_top': 4,
    'open_articulated_object': 5,
    'object_moves_toward_target': 6,
    'no_salient_change': 7,
}

TRANSITION_TEMPLATES = {
    0: 'the gripper should move closer to the {subject}',  # approach_object
    1: 'the {subject} should become grasped by the gripper',  # grasp_object
    2: 'the {subject} should be released by the gripper',  # release_object
    3: 'the {subject} should become inside the {object}',  # place_inside
    4: 'the {subject} should become on top of the {object}',  # place_on_top
    5: 'the {subject} should become open',  # open_articulated_object
    6: 'the {subject} should move toward the {object}',  # object_moves_toward_target
    7: 'no salient spatial transition is expected',  # no_salient_change
}

# Keywords for container detection
CONTAINER_KW = ['basket', 'plate', 'stove', 'microwave', 'cabinet', 'caddy', 'rack', 'drawer']


def classify_relation(subtask_text: str, instruction: str) -> tuple:
    """Classify subtask into relation_label. Returns (label_id, label_name, subject, object_name)."""
    sub = subtask_text.lower()

    # Extract subject: the main object being manipulated
    # (from the first noun phrase after the verb)
    subject = 'object'
    # Try to find common object patterns
    object_patterns = [
        'black bowl', 'alphabet soup', 'tomato sauce', 'cream cheese', 'butter',
        'white mug', 'yellow and white mug', 'porcelain mug',
        'moka pot', 'chocolate pudding', 'salad dressing', 'wine bottle',
        'plate', 'basket', 'drawer', 'cabinet', 'stove', 'microwave',
        'book', 'block', 'soup', 'sauce', 'bowl', 'mug', 'pot',
    ]
    for pat in sorted(object_patterns, key=len, reverse=True):
        if pat in sub:
            subject = pat
            break

    # Determine relation
    if any(w in sub for w in ['grasp', 'pick up', 'grip']):
        return (1, 'grasp_object', subject, '')
    elif any(w in sub for w in ['open', 'close the', 'pull']):
        obj = 'drawer' if 'drawer' in sub else ('microwave' if 'microwave' in sub else 'object')
        return (5, 'open_articulated_object', subject, obj)
    elif any(w in sub for w in ['put into', 'place into', 'put in', 'inside', 'into the basket', 'into the bowl']):
        obj = find_container(sub)
        return (3, 'place_inside', subject, obj)
    elif any(w in sub for w in ['put on', 'place on', 'on top of', 'onto']):
        obj = find_container(sub)
        return (4, 'place_on_top', subject, obj)
    elif any(w in sub for w in ['put the', 'place the', 'place in']):
        if any(c in sub for c in ['basket', 'bowl', 'container', 'drawer', 'inside']):
            return (3, 'place_inside', subject, find_container(sub))
        else:
            return (4, 'place_on_top', subject, find_container(sub))
    elif any(w in sub for w in ['release', 'drop']):
        return (2, 'release_object', subject, '')
    elif any(w in sub for w in ['reach', 'approach', 'move toward']):
        return (0, 'approach_object', subject, '')
    elif any(w in sub for w in ['lift', 'raise']):
        return (1, 'grasp_object', subject, '')
    elif any(w in sub for w in ['carry', 'move', 'transfer', 'take']):
        obj = find_container(sub)
        return (6, 'object_moves_toward_target', subject, obj)
    elif any(w in sub for w in ['press', 'push', 'turn on', 'turn the']):
        return (6, 'object_moves_toward_target', subject, '')
    elif any(w in sub for w in ['turn', 'rotate', 'twist']):
        return (5, 'open_articulated_object', subject, '')
    else:
        return (7, 'no_salient_change', subject, '')


def find_container(text: str) -> str:
    for kw in CONTAINER_KW:
        if kw in text.lower():
            return kw
    return 'target'


def build_episode_data(suite: str, cot_ep: int) -> dict:
    """Build subtask_end_idx and relation data for one episode."""
    lr_dir = LEROBOT / f'{suite}_no_noops_1.0.0_lerobot'
    annot_path = lr_dir / 'annotations' / 'episode_dense_captions_full.jsonl'
    meta_path = lr_dir / 'meta' / 'episodes.jsonl'

    # Load CoT
    steps = {}
    instruction = ''
    with open(annot_path) as f:
        for line in f:
            ep = json.loads(line)
            if ep['episode_index'] == cot_ep:
                steps = ep['steps']
                break
    if not steps:
        return {}

    # Load instruction
    with open(meta_path) as f:
        for line in f:
            ep = json.loads(line)
            if ep['episode_index'] == cot_ep:
                instruction = ep.get('tasks', [''])[0]
                break

    T = max(int(k) for k in steps.keys()) + 1

    # Build subtask segments
    sorted_steps = sorted(steps.items(), key=lambda x: int(x[0]))
    segments = []
    cur_subtask = None
    cur_start = 0
    for s_str, info in sorted_steps:
        s = int(s_str)
        sub = info['subtask']
        if sub != cur_subtask:
            if cur_subtask is not None:
                segments.append({'start': cur_start, 'end': s - 1, 'subtask': cur_subtask})
            cur_subtask = sub
            cur_start = s
    if cur_subtask is not None:
        segments.append({'start': cur_start, 'end': T - 1, 'subtask': cur_subtask})

    # Build frame → subtask_end_idx mapping
    frame_to_end = {}
    for seg in segments:
        for s in range(seg['start'], seg['end'] + 1):
            frame_to_end[s] = seg['end']

    # Build frame → subtask text mapping
    frame_to_subtask = {}
    for s_str, info in sorted_steps:
        s = int(s_str)
        frame_to_subtask[s] = info['subtask']

    return {
        'T': T,
        'instruction': instruction,
        'segments': segments,
        'frame_to_end': frame_to_end,
        'frame_to_subtask': frame_to_subtask,
        'steps_raw': steps,  # full CoT step data with reasoning
    }


def main():
    all_lines = []
    stats = defaultdict(lambda: defaultdict(int))
    total = 0

    for suite in SUITES:
        idx_path = SPATIAL / f'spatial_lara_libero_index_{suite}_v2.jsonl'
        if not idx_path.exists():
            # libero_10 uses index_cot.jsonl directly
            if suite == 'libero_10':
                idx_path = SPATIAL / 'spatial_lara_libero_index_cot.jsonl'
            else:
                print(f'⚠ No index for {suite}'); continue

        # Pre-load episode data
        ep_data_cache = {}

        mask_mode = 'dynamic' if suite == 'libero_10' else 'union'
        mask_rule = 'grasp+gripper_open+longest_alias+container_prescan' if suite == 'libero_10' else 'none'

        suite_total = 0
        with open(idx_path) as f:
            for line in f:
                entry = json.loads(line)
                if entry.get('suite') != suite:
                    continue  # skip entries from other suites in cot.jsonl

                cot_ep = entry.get('cot_episode_id', 0)
                if cot_ep not in ep_data_cache:
                    ep_data_cache[cot_ep] = build_episode_data(suite, cot_ep)

                edata = ep_data_cache[cot_ep]
                if not edata:
                    continue

                cf = entry.get('cot_frame_idx', 0)
                T = entry.get('T_cot', edata['T'])

                # subtask_end_idx
                subtask_end = edata['frame_to_end'].get(cf, min(cf + 8, T - 1))
                subtask_text = edata['frame_to_subtask'].get(cf, '')

                # future_crosses_subtask
                cot_future = entry.get('cot_future_idx', min(cf + 8, T - 1))
                future_crosses = cot_future > subtask_end

                # relation
                rel_id, rel_name, subject, obj = classify_relation(subtask_text, edata.get('instruction', ''))
                relation_valid = rel_id != 7

                # expected_spatial_transition
                template = TRANSITION_TEMPLATES.get(rel_id, '')
                sp_transition = template.format(subject=subject, object=obj)

                # cot_text_transition: Subtask + Reasoning + Spatial transition
                reasoning = ''  # from CoT annotation
                lr_steps = edata.get('steps_raw', {})
                step_info = lr_steps.get(str(cf), {})
                reasoning = step_info.get('reasoning', '')
                cot_orig = f"Subtask: {subtask_text}"
                if reasoning:
                    cot_orig += f" Reasoning: {reasoning}."
                cot_text_orig = cot_orig
                cot_text_trans = f"{cot_orig} Spatial transition: {sp_transition}."

                # Build entry
                new_entry = {
                    'suite': suite,
                    'task_id': entry.get('task_id', 0),
                    'demo_id': entry.get('demo_id', 0),
                    'cot_episode_id': cot_ep,

                    'cot_frame_idx': cf,
                    'hdf5_frame_idx': entry.get('hdf5_frame_idx', cf),

                    'cot_future_idx': cot_future,
                    'hdf5_future_idx': entry.get('hdf5_future_idx', cot_future),

                    'subtask_end_idx': subtask_end,
                    'future_crosses_subtask': future_crosses,

                    'mask_mode': mask_mode,
                    'mask_switch_rule': mask_rule,

                    'relation_valid': relation_valid,
                    'relation_label': rel_name,
                    'relation_label_id': rel_id,
                    'relation_subject': subject,
                    'relation_object': obj,
                    'relation_change': 'becomes_true',

                    'expected_spatial_transition': sp_transition,

                    'cot_text_original': cot_text_orig,
                    'cot_text_transition': cot_text_trans,

                    'alignment_method': 'identity_no_noops_hdf5',
                    'episode_path': entry.get('episode_path', ''),
                    'meta_path': entry.get('meta_path', ''),
                    'primary_object': entry.get('primary_object', ''),
                    'objects_of_interest': entry.get('objects_of_interest', []),
                    'camera_names': entry.get('camera_names', []),
                    'T_cot': T,
                    'T_hdf5': entry.get('T_hdf5', T),
                }
                all_lines.append(json.dumps(new_entry, ensure_ascii=False))
                stats[suite]['total'] += 1
                stats[suite][f'task_{entry.get("task_id", 0)}'] += 1
                stats['relation'][rel_name] += 1
                if future_crosses:
                    stats[suite]['crosses'] += 1
                if not relation_valid:
                    stats[suite]['invalid_rel'] += 1
                suite_total += 1
                total += 1

        print(f'{suite}: {suite_total} entries')

    # Write
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, 'w') as f:
        for line in all_lines:
            f.write(line + '\n')

    # Stats
    print(f'\n=== Statistics ===')
    print(f'Total entries: {total}')
    print(f'\nSuite distribution:')
    for s in SUITES:
        print(f'  {s}: {stats[s].get("total", 0)}')
    print(f'\nRelation distribution:')
    for k, v in sorted(stats['relation'].items(), key=lambda x: -x[1]):
        print(f'  {k}: {v}')
    print(f'\nFuture crosses subtask:')
    for s in SUITES:
        t = stats[s].get('total', 1)
        c = stats[s].get('crosses', 0)
        print(f'  {s}: {c}/{t} ({100*c/max(1,t):.1f}%)')
    print(f'\nInvalid relations:')
    for s in SUITES:
        print(f'  {s}: {stats[s].get("invalid_rel", 0)}')
    print(f'\nSaved: {OUT}')


if __name__ == '__main__':
    main()
