"""
Spatial-CoLaRa Dataset: Merged CoT + Spatial Labels
=====================================================
Combines LaRA-VLA CoT annotations (subtask, reasoning, bbox) with
Spatial-LaRA labels (mask, pose, future supervision), and applies
dynamic mask filtering based on current subtask.

Key feature: For multi-step tasks (e.g. libero_10), the affordance mask
only includes objects relevant to the CURRENT subtask, not all objects
of interest globally.

Returns per sample:
    All spatial fields (image, mask, pose, actions, etc.)
    + cot_subtask, cot_reasoning, cot_bbox
    + filtered_affordance_mask (object-level) if subtask available
    + cot_text (formatted for LaRA-VLA training)
"""

import json
import logging
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from lara_vla.data.spatial_lara_libero_dataset import SpatialLaRALiberoDataset

logger = logging.getLogger(__name__)

# ── Object name → common words mapping (for subtask parsing) ────
# Maps LIBERO instance names to words commonly found in subtask descriptions
_OBJECT_ALIASES = {
    # libero_spatial / libero_goal
    "akita_black_bowl_1": ["black bowl", "bowl"],
    "akita_black_bowl_2": ["black bowl", "bowl"],
    "plate_1": ["left plate", "plate"],
    "plate_2": ["right plate", "plate"],
    "cookies_1": ["cookie", "cookies"],
    "glazed_rim_porcelain_ramekin_1": ["ramekin"],
    "wooden_cabinet_1": ["cabinet", "drawer"],
    "flat_stove_1": ["stove", "burner"],
    "wine_bottle_1": ["wine bottle", "bottle"],
    "cream_cheese_1": ["cream cheese"],
    "wine_rack_1": ["rack", "wine rack"],
    # libero_object
    "alphabet_soup_1": ["alphabet soup", "soup"],
    "tomato_sauce_1": ["tomato sauce", "sauce"],
    "basket_1": ["basket"],
    "salad_dressing_1": ["salad dressing"],
    "ketchup_1": ["ketchup"],
    "bbq_sauce_1": ["bbq sauce", "sauce"],
    "chocolate_pudding_1": ["chocolate pudding", "pudding"],
    "milk_1": ["milk"],
    "butter_1": ["butter"],
    "orange_juice_1": ["orange juice"],
    # libero_10
    "porcelain_mug_1": ["white mug", "porcelain mug", "mug"],
    "white_yellow_mug_1": ["yellow and white mug", "white and yellow mug", "mug"],
    "black_book_1": ["book"],
    "desk_caddy_1": ["caddy", "back compartment", "compartment"],
    "moka_pot_1": ["second moka pot", "moka pot", "pot"],
    "moka_pot_2": ["first moka pot", "moka pot", "pot"],
    "white_cabinet_1": ["cabinet", "bottom drawer", "drawer"],
    "microwave_1": ["microwave"],
    "chocolate_pudding_1": ["chocolate pudding", "pudding"],
}


def _resolve_seg_id(obj_name: str, instance_to_id: dict):
    """Resolve segmentation ID with prefix fallback (same as build script)."""
    if obj_name in instance_to_id:
        return instance_to_id[obj_name]
    for inst_name, seg_id in instance_to_id.items():
        if obj_name.startswith(inst_name):
            return seg_id
    for inst_name, seg_id in instance_to_id.items():
        if inst_name.startswith(obj_name):
            return seg_id
    return None


def _objects_in_subtask(subtask: str, objects_of_interest: list[str],
                        gripper_state: int = -1, last_held_obj: str = "") -> set[str]:
    """Determine which objects are relevant to a given subtask description.

    Args:
        subtask: Current subtask description text.
        objects_of_interest: All task-relevant objects.
        gripper_state: 0=closed (holding), 1=open, -1=unknown.
        last_held_obj: Object name that was last grasped (from previous subtask).
    """
    relevant = set()
    subtask_lower = subtask.lower()
    for obj in objects_of_interest:
        aliases = _OBJECT_ALIASES.get(obj, [obj])
        for alias in aliases:
            if alias in subtask_lower:
                relevant.add(obj)
                break

    # If gripper is closed and we know which object was grasped,
    # keep that object in the mask even if the current subtask text
    # has already moved on (LaRA-VLA annotation boundary issue).
    if gripper_state == 0 and last_held_obj:
        held_found = False
        for obj in objects_of_interest:
            aliases = _OBJECT_ALIASES.get(obj, [obj])
            for alias in aliases:
                if alias == last_held_obj or obj == last_held_obj:
                    relevant.add(obj)
                    held_found = True
                    break
            if held_found:
                break

    # Fallback
    if not relevant and objects_of_interest:
        relevant.add(objects_of_interest[0])
    return relevant


class SpatialCoTDataset(SpatialLaRALiberoDataset):
    """Merged dataset with CoT annotations and dynamic mask filtering.

    Args:
        spatial_root, index_path, future_k, cache_size: Passed to SpatialLaRALiberoDataset.
        cot_root: Root of LeRobot-format CoT data (e.g. datasets/lovejuly/libero_lerobot_all).
        alignment_path: Path to cot_spatial_alignment.json.
        enable_dynamic_mask: If True, filter mask per subtask. Default True.
    """

    def __init__(
        self,
        spatial_root: str,
        index_path: str,
        cot_root: str,
        alignment_path: str,
        future_k: int = 8,
        cache_size: int = 32,
        enable_dynamic_mask: bool = True,  # Gripper-based filtering
    ):
        super().__init__(spatial_root, index_path, future_k, cache_size)
        self.cot_root = Path(cot_root)
        self.enable_dynamic_mask = enable_dynamic_mask

        # Load alignment (optional — index has cot_episode_id as primary source)
        try:
            with open(alignment_path) as f:
                self.alignment = json.load(f)
        except FileNotFoundError:
            self.alignment = {}

        # Pre-load CoT annotations + episode tasks
        self._cot_cache = {}  # (suite, cot_ep) → annotations dict
        self._cot_tasks = {}  # (suite, cot_ep) → task description string
        self._last_held_cache = {}  # (suite, cot_ep) → {frame_idx: last_held_object_name}
        self._last_container_cache = {}  # (suite, cot_ep) → {frame_idx: last_container_name}

        for suite in set(entry["suite"] for entry in self.entries):
            meta_path = self.cot_root / f"{suite}_no_noops_1.0.0_lerobot" / "meta" / "episodes.jsonl"
            if meta_path.exists():
                with open(meta_path) as f:
                    for line in f:
                        ep = json.loads(line)
                        key = (suite, ep["episode_index"])
                        self._cot_tasks[key] = ep.get("tasks", [""])[0]

        logger.info(f"Loaded alignment: {len(self.alignment)} episodes, "
                    f"{len(self._cot_tasks)} task descriptions")

    def _get_held_at_frame(self, suite: str, cot_ep: int, frame_idx: int) -> str:
        """Get held object at a specific frame."""
        return self._get_last_held_object(suite, cot_ep, max(0, frame_idx))

    def _get_held_before_switch(self, suite: str, cot_ep: int, frame_idx: int) -> str:
        """Get the held object before the most recent switch.

        Scans backwards to find the previous held object value.
        """
        current = self._get_last_held_object(suite, cot_ep, frame_idx)
        if not current:
            return ""
        for s in range(frame_idx - 1, -1, -1):
            prev = self._get_last_held_object(suite, cot_ep, s)
            if prev and prev != current:
                return prev
        return ""

    def _get_last_held_object(self, suite: str, cot_ep: int, frame_idx: int) -> str:
        """Find currently held object based on 'grasp X' events in subtask text.

        Held object only changes when subtask contains 'grasp'/'pick up' of
        a DIFFERENT object. Gripper state is not used for switching.
        """
        cache_key = (suite, cot_ep)
        if cache_key not in self._last_held_cache:
            cot_data = self._load_cot_episode(suite, cot_ep)
            steps = cot_data.get("steps", {})
            held_map = {}
            container_map = {}
            last_held = ""
            last_container = ""
            _CONTAINER_KW = ["basket", "plate", "stove", "microwave", "cabinet", "caddy", "rack"]
            for step_str in sorted(steps.keys(), key=int):
                s = int(step_str)
                info = steps[step_str]
                sub = info.get("subtask", "").lower()
                # Track container from subtask (longest match, persistent)
                best_cont = None; best_clen = 0
                for obj_alias_key in _OBJECT_ALIASES:
                    if not any(kw in obj_alias_key for kw in _CONTAINER_KW):
                        continue
                    for alias in _OBJECT_ALIASES[obj_alias_key]:
                        if alias in sub and len(alias) > best_clen:
                            best_cont = obj_alias_key; best_clen = len(alias)
                if best_cont:
                    last_container = best_cont

                # Switch held object on grasp OR when gripper is open
                is_grasp = "grasp" in sub or "pick up" in sub or "grip" in sub
                grip = info.get("gripper_state", -1)
                if is_grasp or grip == 1:
                    # Find best match: longest alias wins (for disambiguation)
                    best_match = None
                    best_len = 0
                    for obj_alias_key in _OBJECT_ALIASES:
                        for alias in _OBJECT_ALIASES[obj_alias_key]:
                            if alias in sub and len(alias) > best_len:
                                best_match = obj_alias_key
                                best_len = len(alias)
                    if best_match and best_match != last_held:
                        # Held switched: try to find the next container from upcoming frames
                        last_held = best_match
                        old_container = last_container  # save before scanning
                        # Scan forward for next container mention (no limit)
                        fut_steps = sorted([int(k) for k in steps.keys() if int(k) > s])
                        for fs in fut_steps:
                            fsub = steps[str(fs)]['subtask'].lower()
                            for obj_ak in _OBJECT_ALIASES:
                                if not any(kw in obj_ak for kw in _CONTAINER_KW):
                                    continue
                                for alias in _OBJECT_ALIASES[obj_ak]:
                                    if alias in fsub:
                                        last_container = obj_ak
                                        break
                                if last_container != old_container:
                                    break
                            if last_container != old_container:
                                break
                container_map[s] = last_container
                held_map[s] = last_held
            self._last_held_cache[cache_key] = held_map
            self._last_container_cache[cache_key] = container_map

        held_map = self._last_held_cache[cache_key]
        container_map = self._last_container_cache[cache_key]
        if frame_idx in held_map:
            return held_map[frame_idx]
        for s in sorted(held_map.keys(), reverse=True):
            if s <= frame_idx:
                return held_map[s]
        return ""

    def _load_cot_episode(self, suite: str, cot_ep: int) -> dict:
        """Load CoT annotations for a specific episode."""
        cache_key = (suite, cot_ep)
        if cache_key in self._cot_cache:
            return self._cot_cache[cache_key]

        cot_dir = self.cot_root / f"{suite}_no_noops_1.0.0_lerobot"
        annot_path = cot_dir / "annotations" / "episode_dense_captions_full.jsonl"

        with open(annot_path) as f:
            for line in f:
                ep = json.loads(line)
                if ep["episode_index"] == cot_ep:
                    self._cot_cache[cache_key] = ep
                    return ep
        return {}

    def __getitem__(self, idx: int):
        # Get spatial sample
        sample = super().__getitem__(idx)

        suite = sample["suite"]
        task_id = sample["task_id"]
        demo_id = sample["demo_id"]
        frame_idx = sample["frame_idx"]  # CoT frame index (from index_cot.jsonl)

        # ── Load CoT annotation ────────────────────────────────
        # Use cot_episode_id from index (visual matching), fallback to alignment file
        index_entry = self.entries[idx]
        cot_ep = index_entry.get("cot_episode_id")
        if cot_ep is None:
            align_key = f"{suite}/{task_id}/{demo_id}"
            cot_ep = self.alignment.get(align_key)

        cot_subtask = ""
        cot_reasoning = ""
        cot_gripper_state = -1

        if cot_ep is not None:
            cot_data = self._load_cot_episode(suite, cot_ep)
            steps = cot_data.get("steps", {})
            step_key = str(frame_idx)

            if step_key in steps:
                step_info = steps[step_key]
                cot_subtask = step_info.get("subtask", "")
                cot_reasoning = step_info.get("reasoning", "")
                cot_gripper_state = step_info.get("gripper_state", -1)

            # ── Dynamic mask filtering: grasp-event based ─
            sample["num_relevant_objects"] = len(sample["objects_of_interest"])
            if self.enable_dynamic_mask and suite == "libero_10" and cot_subtask:
                objs = list(sample["objects_of_interest"])
                held_obj = self._get_last_held_object(suite, cot_ep, frame_idx)
                relevant_set = set()

                if held_obj:
                    relevant_set.add(held_obj)
                else:
                    # Before first grasp: use longest-match from subtask text
                    best_init = None; best_ilen = 0
                    for obj_alias_key in _OBJECT_ALIASES:
                        for alias in _OBJECT_ALIASES[obj_alias_key]:
                            if alias in cot_subtask.lower() and len(alias) > best_ilen:
                                best_init = obj_alias_key; best_ilen = len(alias)
                    if best_init:
                        relevant_set.add(best_init)
                    else:
                        relevant_set.update(objs)

                # Goal containers: from pre-computed container_map (persistent, no fallback)
                _CONTAINER_KW = ["basket", "plate", "stove", "microwave", "cabinet", "caddy", "rack"]
                container_objs = [o for o in objs if any(kw in o for kw in _CONTAINER_KW)]
                if container_objs:
                    container_map = self._last_container_cache.get((suite, cot_ep), {})
                    use_cont = container_map.get(frame_idx, "")
                    if use_cont and use_cont in objs:
                        relevant_set.add(use_cont)
                    elif container_map:
                        # Happens at frame 0 before any container mentioned
                        # Find first non-empty container from future frames
                        for s in sorted(container_map.keys()):
                            c = container_map[s]
                            if c and c in objs:
                                relevant_set.add(c)
                                break
                    else:
                        relevant_set.update(container_objs)

                self._filter_mask(sample, relevant_set, suite, idx)

        # ── Get instruction ────────────────────────────────────
        instruction = self._cot_tasks.get((suite, cot_ep), "") if cot_ep is not None else ""

        cot_text = self._format_cot_text(instruction, cot_subtask, cot_reasoning)

        sample["cot_subtask"] = cot_subtask
        sample["cot_reasoning"] = cot_reasoning
        sample["cot_gripper_state"] = cot_gripper_state
        sample["cot_text"] = cot_text
        sample["cot_episode_id"] = cot_ep if cot_ep is not None else -1
        sample["hdf5_demo_id"] = demo_id
        sample["cot_episode"] = cot_ep if cot_ep is not None else -1
        sample["instruction"] = instruction

        # Transition index fields (from build_transition_index.py)
        sample["subtask_end_idx"] = index_entry.get("subtask_end_idx", sample.get("subtask_end_idx", 0))
        sample["future_crosses_subtask"] = index_entry.get("future_crosses_subtask", False)
        sample["mask_mode"] = index_entry.get("mask_mode", "union")
        sample["mask_switch_rule"] = index_entry.get("mask_switch_rule", "none")
        sample["relation_valid"] = index_entry.get("relation_valid", True)
        sample["relation_label"] = index_entry.get("relation_label", "")
        sample["relation_label_id"] = index_entry.get("relation_label_id", -1)
        sample["relation_subject"] = index_entry.get("relation_subject", "")
        sample["relation_object"] = index_entry.get("relation_object", "")
        sample["expected_spatial_transition"] = index_entry.get("expected_spatial_transition", "")
        sample["cot_text_original"] = index_entry.get("cot_text_original", cot_text)
        sample["cot_text_transition"] = index_entry.get("cot_text_transition", "")
        sample["alignment_method"] = index_entry.get("alignment_method", "")

        return sample

    def _filter_mask(self, sample, relevant_objs, suite, idx):
        """Rebuild affordance mask to only show relevant objects."""
        all_objs = set(sample["objects_of_interest"])
        if not relevant_objs or relevant_objs == all_objs:
            sample["num_relevant_objects"] = len(all_objs)
            return

        # Reload episode to get seg and instance_to_id
        ep_path = self.entries[idx]["episode_path"]
        data = self._load_episode(ep_path)
        h5_frame = sample.get("hdf5_frame_idx", sample["frame_idx"])
        future_h5 = min(h5_frame + (sample["future_idx"] - sample["frame_idx"]),
                        data["seg_agentview"].shape[0] - 1)
        H, W = data["seg_agentview"].shape[1:3]

        # Get instance_to_id from meta (cached per episode)
        meta_key = f"__meta_{ep_path}"
        if meta_key not in self._cot_cache:
            meta_path = self.entries[idx].get("meta_path", "")
            if meta_path:
                full_meta = self.root / meta_path
                if full_meta.exists():
                    with open(full_meta) as f:
                        self._cot_cache[meta_key] = json.load(f)
                else:
                    self._cot_cache[meta_key] = {}
            else:
                self._cot_cache[meta_key] = {}
        meta = self._cot_cache.get(meta_key, {})
        inst_to_id = meta.get("instance_to_id", {})

        # Rebuild masks using HDF5 frame index (not CoT frame index!)
        seg = data["seg_agentview"][h5_frame]
        seg_w = data["seg_wrist"][h5_frame]

        mask_a = np.zeros((H, W), dtype=np.uint8)
        mask_w = np.zeros((H, W), dtype=np.uint8)
        for obj_name in relevant_objs:
            sid = _resolve_seg_id(obj_name, inst_to_id)
            if sid is not None:
                mask_a |= (seg == sid).astype(np.uint8)
                mask_w |= (seg_w == sid).astype(np.uint8)

        sample["current_affordance_mask_agentview"] = mask_a[np.newaxis, ...]
        sample["current_affordance_mask_wrist"] = mask_w[np.newaxis, ...]

        # Rebuild future masks (using HDF5 frame index)
        seg_f = data["seg_agentview"][future_h5]
        seg_wf = data["seg_wrist"][future_h5]
        mask_af = np.zeros((H, W), dtype=np.uint8)
        mask_wf = np.zeros((H, W), dtype=np.uint8)
        for obj_name in relevant_objs:
            sid = _resolve_seg_id(obj_name, inst_to_id)
            if sid is not None:
                mask_af |= (seg_f == sid).astype(np.uint8)
                mask_wf |= (seg_wf == sid).astype(np.uint8)

        sample["future_affordance_mask_agentview"] = mask_af[np.newaxis, ...]
        sample["future_affordance_mask_wrist"] = mask_wf[np.newaxis, ...]
        sample["num_relevant_objects"] = len(relevant_objs)

    @staticmethod
    def _format_cot_text(instruction: str, subtask: str, reasoning: str) -> str:
        """Format CoT text for LaRA-VLA Stage I training.

        Format: "Instruction @ Subtask: ... BBox: ... Reasoning: ..."
        This matches the training input format from LaRA-VLA.
        """
        parts = [instruction]
        if subtask:
            parts.append(f"Subtask: {subtask}")
        if reasoning:
            parts.append(f"Reasoning: {reasoning}")
        return " @ ".join(parts)
