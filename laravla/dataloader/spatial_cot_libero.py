"""
SpatialCoT LIBERO Dataset Adapter for LaRA-VLA build_dataloader.
==================================================================
Wraps SpatialCoTDataset into the format LaRA-VLA's Qwen_GR00T.forward()
expects. Registered as dataset_py="spatial_cot_libero" in build_dataloader.

Field mapping (SpatialCoTDataset → Qwen_GR00T batch dict):
    image[np]          → "image" [PIL.Image]
    image_wrist[np]    → "image_wrist" [PIL.Image]
    image_next[np]     → "image_next" [PIL.Image]
    instruction[str]   → "lang" str
    actions[np 8,7]    → "action" np[T,7]  (use as-is)
    robot_state[np 16] → "state" np[7]     (truncate to first 7 dims)

    cot_text_transition[str]         → "cot_text_transition" str
    current_affordance_mask_*[np]    → "current_affordance_mask_*" float32
    future_affordance_mask_*[np]     → "future_affordance_mask_*" float32
    goal_affordance_mask_*[np]       → "goal_affordance_mask_*" float32
    relation_label[str]              → "relation_label" str
    relation_label_id[int]           → "relation_label_id" int

Usage in config:
    datasets:
      vla_data:
        dataset_py: spatial_cot_libero
        spatial_root: output/spatial_lara_libero
        index_path: output/spatial_lara_libero_no_noops/spatial_lara_libero_index_cot_transition_all_fixed_v3.jsonl
        cot_root: datasets/lovejuly/libero_lerobot_all
        alignment_path: output/spatial_lara_libero/cot_spatial_alignment.json
        enable_dynamic_mask: true
        per_device_batch_size: 2
"""

import os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image


def numpy_to_pil(img_np):
    """Convert [3, H, W] float32 (0-1) numpy to PIL Image."""
    arr = (img_np.transpose(1, 2, 0) * 255).astype(np.uint8)
    return Image.fromarray(arr)


class SpatialCoTLiberoAdapter(Dataset):
    """Adapter: SpatialCoTDataset → LaRA-VLA Qwen_GR00T batch format."""

    def __init__(self, spatial_root, index_path, cot_root, alignment_path,
                 enable_dynamic_mask=True, future_k=8, cache_size=16,
                 state_dim=7):
        # Lazy import to avoid circular deps
        from lara_vla.data.spatial_cot_dataset import SpatialCoTDataset
        self._ds = SpatialCoTDataset(
            spatial_root=spatial_root,
            index_path=index_path,
            cot_root=cot_root,
            alignment_path=alignment_path,
            future_k=future_k,
            cache_size=cache_size,
            enable_dynamic_mask=enable_dynamic_mask,
        )
        self._state_dim = state_dim

    def __len__(self):
        return len(self._ds)

    def __getitem__(self, idx):
        s = self._ds[idx]

        # ── LaRA-VLA core fields ───────────────────────────
        img_pil = numpy_to_pil(s["image"])
        img_next_pil = numpy_to_pil(s["image_next"]) if s.get("image_next") is not None else None

        # State: truncate to state_dim (action model expects 7, our robot_state is 16)
        robot_state = s.get("robot_state", np.zeros(16, dtype=np.float32))
        state = robot_state[:self._state_dim].copy().astype(np.float32)

        sample = {
            # Core VLA fields
            "image": [img_pil],                         # List[PIL] — Qwen_GR00T expects list
            "lang": s["instruction"],                    # instruction string
            "action": s["actions"].copy(),               # [8, 7] float32
            "state": state,                              # [state_dim] float32

            # CoT fields
            "cot_text_transition": s.get("cot_text_transition", ""),
            "cot_text_original": s.get("cot_text_original", ""),
            "expected_spatial_transition": s.get("expected_spatial_transition", ""),

            # Current mask (input modality)
            "current_affordance_mask_agentview": s["current_affordance_mask_agentview"].astype(np.float32).squeeze(0),
            "current_affordance_mask_wrist": s["current_affordance_mask_wrist"].astype(np.float32).squeeze(0),

            # Future mask (supervision)
            "future_affordance_mask_agentview": s["future_affordance_mask_agentview"].astype(np.float32).squeeze(0),
            "future_affordance_mask_wrist": s["future_affordance_mask_wrist"].astype(np.float32).squeeze(0),

            # Goal mask (supervision)
            "goal_affordance_mask_agentview": s["goal_affordance_mask_agentview"].astype(np.float32).squeeze(0),
            "goal_affordance_mask_wrist": s["goal_affordance_mask_wrist"].astype(np.float32).squeeze(0),

            # Relation
            "relation_label": s.get("relation_label", ""),
            "relation_label_id": int(s.get("relation_label_id", -1)),
            "relation_subject": s.get("relation_subject", ""),
            "relation_object": s.get("relation_object", ""),

            # Image next (LaRA-VLA original)
            "image_next": [img_next_pil] if img_next_pil is not None else None,
            "image_next_fallback": img_next_pil is None,

            # Debug metadata
            "suite": s.get("suite", ""),
            "task_id": s.get("task_id", -1),
            "demo_id": s.get("demo_id", -1),
            "subtask_end_idx": s.get("subtask_end_idx", -1),
            "hdf5_frame_idx": s.get("hdf5_frame_idx", -1),
            "hdf5_future_idx": s.get("hdf5_future_idx", -1),
            "mask_mode": s.get("mask_mode", ""),
        }

        return sample


def collate_fn(batch):
    """Identity collate: Qwen_GR00T.forward() expects List[dict]."""
    return batch


def get_spatial_cot_dataset(data_cfg) -> SpatialCoTLiberoAdapter:
    """Build SpatialCoT dataset from config."""
    spatial_root = getattr(data_cfg, "spatial_root", None) or \
        os.path.join(os.path.dirname(__file__), "..", "..", "output", "spatial_lara_libero")
    index_path = getattr(data_cfg, "index_path", None) or \
        os.path.join(os.path.dirname(__file__), "..", "..", "output", "spatial_lara_libero_no_noops",
                     "spatial_lara_libero_index_cot_transition_all_fixed_v3.jsonl")
    cot_root = getattr(data_cfg, "cot_root", None) or \
        os.environ.get("LEROBOT_ROOT", "")
    alignment_path = getattr(data_cfg, "alignment_path", None) or \
        os.path.join(spatial_root, "cot_spatial_alignment.json")
    enable_dynamic_mask = getattr(data_cfg, "enable_dynamic_mask", True)
    state_dim = getattr(data_cfg, "state_dim", 7)

    ds = SpatialCoTLiberoAdapter(
        spatial_root=spatial_root,
        index_path=index_path,
        cot_root=cot_root,
        alignment_path=alignment_path,
        enable_dynamic_mask=enable_dynamic_mask,
        state_dim=state_dim,
    )
    return ds


def build_spatial_cot_dataloader(data_cfg, per_device_batch_size=2, num_workers=2):
    """Build a DataLoader for SpatialCoT data."""
    ds = get_spatial_cot_dataset(data_cfg)
    loader = DataLoader(
        ds,
        batch_size=per_device_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )
    return loader
