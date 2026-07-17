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

    # ── Compute reference metrics on manifest ────────────────
    dino_cos_vals = []
    cur_dice_vals = []
    fut_dice_vals = []
    goal_dice_vals = []
    rel_acc_vals = []
    valid_count = 0

    # Re-build loader and iterate to find manifest samples
    loader2 = build_dataloader(OmegaConf.create({"datasets": {"vla_data": {
        "dataset_py": "spatial_cot_libero", "spatial_root": SPATIAL,
        "index_path": IDX, "cot_root": COT,
        "alignment_path": SPATIAL + "/cot_spatial_alignment.json",
        "enable_dynamic_mask": True, "per_device_batch_size": 4,
        "num_workers": 0, "state_dim": 7,
    }}}), dataset_py="spatial_cot_libero")

    manifest_set = {(m["suite"], m["task_id"], m["demo_id"], m["hdf5_frame_idx"]) for m in manifest}
    evaluated = 0
    for batch in loader2:
        if evaluated >= len(manifest):
            break
        # Find manifest samples in this batch
        batch_indices = []
        for i, s in enumerate(batch):
            key = (s["suite"], s["task_id"], s["demo_id"], s["hdf5_frame_idx"])
            if key in manifest_set:
                batch_indices.append(i)

        if not batch_indices:
            continue

        # Forward P1 backbone for this batch
        images = [s["image"] for s in batch]
        instructions = [s["lang"] for s in batch]
        with torch.no_grad():
            qo = vla.qwen_vl_interface.encode_observation(
                images=images, instructions=instructions, output_hidden_states=True)
            vh = qo.hidden_states[-1]
            out = vla.spatial_backbone(vh)
            fut_rgb = torch.stack([torch.from_numpy(s["image_tau_future_raw"]).permute(2,0,1) for s in batch]).to("cuda")
            dt = vla.dino_encoder(fut_rgb)
            r = dino_future_cosine(out.pred_future_dino, dt)
            if r["cosine"] is not None:
                dino_cos_vals.append(r["cosine"].item())

        # Mask Dice (simplified)
        cm = torch.from_numpy(np.stack([s["current_affordance_mask_agentview"] for s in batch])).unsqueeze(1).to("cuda").float()
        fm = torch.from_numpy(np.stack([s.get("future_tau_mask_agentview", s.get("future_affordance_mask_agentview", np.zeros((224,224),dtype=np.float32))) for s in batch])).to("cuda").float()
        gm = torch.from_numpy(np.stack([s["goal_affordance_mask_agentview"] for s in batch])).to("cuda").float()
        ri = torch.tensor([s["relation_label_id"] for s in batch], dtype=torch.long, device="cuda")

        R = out.future_mask_logits.shape[-1]
        cur_gt = F.interpolate(cm.float(), size=(R,R), mode='nearest').squeeze(1)
        fut_gt = F.interpolate(fm.unsqueeze(1).float(), size=(R,R), mode='nearest').squeeze(1)
        goal_gt = F.interpolate(gm.unsqueeze(1).float(), size=(R,R), mode='nearest').squeeze(1)
        cur_dice = _dice(out.current_mask_logits, cur_gt)
        fut_dice = _dice(out.future_mask_logits, fut_gt)
        goal_dice = _dice(out.goal_mask_logits, goal_gt)
        rel_pred = out.relation_logits.argmax(dim=1)
        rel_valid = (ri >= 0) & (ri < out.relation_logits.shape[1])
        rel_acc = (rel_pred[rel_valid] == ri[rel_valid]).float().mean().item() if rel_valid.any() else 0

        cur_dice_vals.append(cur_dice.item())
        fut_dice_vals.append(fut_dice.item())
        goal_dice_vals.append(goal_dice.item())
        rel_acc_vals.append(rel_acc)
        valid_count += 1
        evaluated += len(batch_indices)

    # ── Save manifest + reference metrics ────────────────────
    manifest_data = {
        "manifest": manifest,
        "num_samples": len(manifest),
        "reference_metrics": {
            "dino_cosine_mean": float(np.mean(dino_cos_vals)) if dino_cos_vals else 0,
            "dino_cosine_std": float(np.std(dino_cos_vals)) if dino_cos_vals else 0,
            "cur_dice_mean": float(np.mean(cur_dice_vals)) if cur_dice_vals else 0,
            "fut_dice_mean": float(np.mean(fut_dice_vals)) if fut_dice_vals else 0,
            "goal_dice_mean": float(np.mean(goal_dice_vals)) if goal_dice_vals else 0,
            "rel_acc_mean": float(np.mean(rel_acc_vals)) if rel_acc_vals else 0,
            "num_batches": valid_count,
        },
        "manifest_hash": hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()[:16],
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(manifest_data, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Reference metrics on {valid_count} batches ({len(manifest)} samples):")
    for k, v in manifest_data["reference_metrics"].items():
        print(f"  {k}: {v:.4f}")
    print(f"  manifest_hash: {manifest_data['manifest_hash']}")
    print(f"  Saved to: {output_path}")


def _dice(logits, target, eps=1e-6):
    pred = (torch.sigmoid(logits) > 0.5).float()
    if target.dim() == 2: target = target.unsqueeze(1)
    elif target.dim() == 3: target = target.unsqueeze(1)
    inter = (pred * target).sum()
    union = pred.sum() + target.sum()
    return (2.0 * inter + eps) / (union + eps)


if __name__ == "__main__":
    main()
