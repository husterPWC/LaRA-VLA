"""
Unified SpatialTransitionTargetBuilder.
========================================
Single entry point for P1 and P2 to read fixed-tau supervision targets.
No fallbacks to image_next, CoT future, or estimated indices.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import numpy as np
import torch


@dataclass
class SpatialTransitionTargets:
    """All spatial supervision targets for one training sample."""
    # Frame indices
    hdf5_frame_idx: int
    hdf5_tau_future_idx: int
    tau_future_gap: int
    tau_future_valid: bool

    # Masks (numpy arrays, [H,W] or [1,H,W])
    current_mask: np.ndarray
    future_tau_mask: np.ndarray
    goal_mask: np.ndarray

    # Relation
    relation_label_id: int

    # RGB images (PIL or numpy)
    image_current: np.ndarray
    image_tau_future: np.ndarray

    # Metadata
    episode_path: str
    suite: str
    task_id: int
    demo_id: int


def build_spatial_targets(
    entry: dict,
    episode_data: dict,
    tau_offset: int = 8,
    min_gap: int = 2,
) -> SpatialTransitionTargets:
    """
    Build spatial targets from index entry + loaded NPZ episode.

    Args:
        entry:        index JSONL entry (must have hdf5_tau_future_idx, tau_future_valid)
        episode_data: loaded NPZ dict (rgb_agentview, affordance_mask_agentview, ...)
        tau_offset:   fixed frame offset (default 8 = 0.4s at 20fps)
        min_gap:      minimum gap for tau_future_valid=True

    Returns:
        SpatialTransitionTargets with all supervision fields

    Raises:
        KeyError if required fields are missing (no silent fallback).
    """
    # Required fields — crash if missing
    cur_idx = entry["hdf5_frame_idx"]
    tau_idx = entry["hdf5_tau_future_idx"]
    goal_idx = entry["subtask_end_idx"]
    tau_valid = entry["tau_future_valid"]
    ep_T = episode_data["rgb_agentview"].shape[0]

    # Clamp to episode bounds
    tau_idx = min(tau_idx, ep_T - 1)
    goal_idx = min(goal_idx, ep_T - 1)
    gap = tau_idx - cur_idx
    tau_valid = tau_valid and (gap >= min_gap)

    # Images
    img_cur = episode_data["rgb_agentview"][cur_idx]       # [H,W,3] uint8
    img_tau = episode_data["rgb_agentview"][tau_idx]       # [H,W,3] uint8

    # Masks
    mask_cur = episode_data["affordance_mask_agentview"][cur_idx]   # [H,W] or [1,H,W]
    mask_tau = episode_data["affordance_mask_agentview"][tau_idx]
    mask_goal = episode_data["affordance_mask_agentview"][goal_idx]

    # Relation
    rel_id = entry.get("relation_label_id", -1)

    return SpatialTransitionTargets(
        hdf5_frame_idx=cur_idx,
        hdf5_tau_future_idx=tau_idx,
        tau_future_gap=gap,
        tau_future_valid=tau_valid,
        current_mask=mask_cur,
        future_tau_mask=mask_tau,
        goal_mask=mask_goal,
        relation_label_id=rel_id,
        image_current=img_cur,
        image_tau_future=img_tau,
        episode_path=entry.get("episode_path", ""),
        suite=entry.get("suite", ""),
        task_id=entry.get("task_id", -1),
        demo_id=entry.get("demo_id", -1),
    )
