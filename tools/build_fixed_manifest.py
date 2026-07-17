#!/usr/bin/env python
"""
Build a fixed, stratified validation manifest for P1→P2 quality checks.
=========================================================================
Samples 50 entries per suite (200 total) from the V4 index, saves sample
IDs and reference metrics computed by the trained P1 backbone.

Usage:
    python tools/build_fixed_manifest.py \
        --p1-ckpt results/P1_formal/best_student.pt \
        --output results/P1_formal/fixed_manifest.json
"""

import json, os, sys, hashlib, warnings; warnings.filterwarnings("ignore")
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn.functional as F

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

CKPT = os.environ.get("LARAVLA_CKPT",
    str(_REPO.parent / "models/LaRA-VLA-libero/checkpoints/steps_25000_pytorch_model.pt"))
SPATIAL = str(_REPO / "output/spatial_lara_libero")
IDX = str(_REPO / "output/spatial_lara_libero_no_noops" /
          "spatial_lara_libero_index_cot_transition_all_fixed_v4_tau.jsonl")
COT = os.environ.get("LEROBOT_ROOT",
    str(_REPO.parent / "datasets" / "lovejuly" / "libero_lerobot_all"))


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--p1-ckpt", type=str, default=str(_REPO / "results/P1_formal/best_student.pt"))
    parser.add_argument("--output", type=str, default=str(_REPO / "results/P1_formal/fixed_manifest.json"))
    parser.add_argument("--samples-per-suite", type=int, default=50)
    args = parser.parse_args()

    print("=" * 60)
    print("Building fixed validation manifest")
    print("=" * 60)

    # ── Build VLA + load P1 backbone ────────────────────────
    from laravla.model.tools import read_mode_config
    from omegaconf import OmegaConf
    model_cfg, _ = read_mode_config(Path(CKPT))
    model_cfg["framework"]["mask_conditioned_transition"] = {
        "enable": True, "num_transition_tokens": 6, "mask_res": 56,
        "num_relation_labels": 7, "transition_dim": 512,
        "loss_weights": {"future_mask":0.05,"goal_mask":0.10,"relation":0.05,
                         "current_mask":0.05,"dino_future":0.05,"slot_residual_gamma":1.5},
        "dino": {"model_name":"dinov2_vitb14","dino_dim":768,"num_patches":256},
    }
    from laravla.model.framework import build_framework
    vla = build_framework(OmegaConf.create(model_cfg))
    vla.load_state_dict(torch.load(CKPT, map_location="cpu"), strict=False)

    # Restore VLM contract
    raw_ckpt = torch.load(args.p1_ckpt, map_location="cpu")
    if "vlm_contract" in raw_ckpt:
        from laravla.model.framework.vlm_contract import restore_vlm_contract
        restore_vlm_contract(vla, raw_ckpt["vlm_contract"])
        print(f"  VLM contract restored")

    # Load backbone
    bk = {k.replace("backbone.", ""): v
          for k, v in raw_ckpt["p1_state_dict"].items() if k.startswith("backbone.")}
    vla.spatial_backbone.load_state_dict(bk, strict=True)
    vla = vla.to("cuda")
    vla.eval()

    from laravla.model.modules.spatial_transition import dino_future_cosine

    # ── Build dataloader ─────────────────────────────────────
    from laravla.dataloader import build_dataloader
    loader = build_dataloader(OmegaConf.create({"datasets": {"vla_data": {
        "dataset_py": "spatial_cot_libero", "spatial_root": SPATIAL,
        "index_path": IDX, "cot_root": COT,
        "alignment_path": SPATIAL + "/cot_spatial_alignment.json",
        "enable_dynamic_mask": True, "per_device_batch_size": 4,
        "num_workers": 0, "state_dim": 7,
    }}}), dataset_py="spatial_cot_libero")

    # ── Collect stratified samples ───────────────────────────
    suite_samples = defaultdict(list)
    target_per_suite = args.samples_per_suite
    suites_needed = {"libero_spatial", "libero_object", "libero_goal", "libero_10"}

    print(f"  Collecting {target_per_suite} samples per suite...")
    for batch in loader:
        for s in batch:
            suite = s.get("suite", "")
            if suite in suites_needed and len(suite_samples[suite]) < target_per_suite:
                # Store sample ID info
                sample_id = {
                    "suite": suite,
                    "task_id": s.get("task_id", -1),
                    "demo_id": s.get("demo_id", -1),
                    "hdf5_frame_idx": s.get("hdf5_frame_idx", -1),
                    "hdf5_tau_future_idx": s.get("hdf5_tau_future_idx", -1),
                    "tau_future_valid": bool(s.get("tau_future_valid", True)),
                }
                suite_samples[suite].append(sample_id)

        # Check if all suites have enough
        if all(len(suite_samples[s]) >= target_per_suite for s in suites_needed):
            break

    manifest = []
    for suite in sorted(suites_needed):
        manifest.extend(suite_samples[suite][:target_per_suite])
    print(f"  Collected {len(manifest)} samples: { {s: len(v) for s, v in suite_samples.items()} }")

    # ── Save manifest only (metrics deferred to P2 startup) ──
    manifest_data = {
        "manifest": manifest,
        "num_samples": len(manifest),
        "manifest_hash": hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()[:16],
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(manifest_data, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Manifest saved: {len(manifest)} samples")
    print(f"  manifest_hash: {manifest_data['manifest_hash']}")
    print(f"  Saved to: {output_path}")

if __name__ == "__main__":
    main()
