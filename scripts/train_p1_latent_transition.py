#!/usr/bin/env python
"""
P1 Formal Training: Mask-Conditioned Latent Transition Reasoning.
=================================================================
Trains spatial_transition bottleneck modules (34M params) with frozen Qwen-VL
and action model. Supervised by future_mask, goal_mask, relation_label.

Usage (server, 8 GPU):
    accelerate launch --num_processes=8 scripts/train_p1_latent_transition.py \
        --max-steps 50000 --batch-size 4 --lr 3e-4 --save-interval 5000

Usage (server, single GPU smoke):
    python scripts/train_p1_latent_transition.py --max-steps 500
"""

import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

_THIS = Path(__file__).resolve()
_REPO = _THIS.parents[1]
sys.path.insert(0, str(_REPO))

import warnings; warnings.filterwarnings("ignore")

CKPT = os.environ.get("LARAVLA_CKPT",
    str(_REPO.parent / "models/LaRA-VLA-libero/checkpoints/steps_25000_pytorch_model.pt"))
SPATIAL = str(_REPO / "output" / "spatial_lara_libero")
IDX = str(_REPO / "output" / "spatial_lara_libero_no_noops" /
          "spatial_lara_libero_index_cot_transition_all_fixed_v3.jsonl")
COT = os.environ.get("LEROBOT_ROOT",
    str(_REPO.parent / "datasets" / "lovejuly" / "libero_lerobot_all"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-steps", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--save-interval", type=int, default=5000)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--eval-interval", type=int, default=2000)
    parser.add_argument("--eval-batches", type=int, default=50)
    parser.add_argument("--output-dir", type=str, default=str(_REPO / "results/P1_latent_transition"))
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("P1: Mask-Conditioned Latent Transition Training")
    print(f"  Max steps: {args.max_steps}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  LR: {args.lr}")
    print(f"  Output: {output_dir}")
    print("=" * 60)

    # ── Build dataloader ────────────────────────────────────
    from laravla.dataloader import build_dataloader
    cfg = OmegaConf.create({"datasets": {"vla_data": {
        "dataset_py": "spatial_cot_libero", "spatial_root": SPATIAL,
        "index_path": IDX, "cot_root": COT,
        "alignment_path": SPATIAL + "/cot_spatial_alignment.json",
        "enable_dynamic_mask": True, "per_device_batch_size": args.batch_size,
        "num_workers": 2, "state_dim": 7,
    }}})
    loader = build_dataloader(cfg, dataset_py="spatial_cot_libero")
    print(f"  DataLoader: {len(loader)} batches/epoch")

    # ── Build model ─────────────────────────────────────────
    from laravla.model.tools import read_mode_config
    model_cfg, norm_stats = read_mode_config(Path(CKPT))
    model_cfg["framework"]["mask_conditioned_transition"] = {
        "enable": True, "num_mask_tokens": 8, "num_transition_tokens": 6,
        "mask_res": 56, "num_relation_labels": 6, "transition_dim": 512,
        "loss_weights": {"future_mask": 0.05, "goal_mask": 0.10, "relation": 0.05},
    }
    cfg = OmegaConf.create(model_cfg)
    from laravla.model.framework import build_framework
    vla = build_framework(cfg)
    vla.load_state_dict(torch.load(CKPT, map_location="cpu"), strict=False)
    vla = vla.to("cuda")
    vla.training_stage = "latent_transition"

    # Freeze VLM + action_model + P2 adapter (P1 only trains transition)
    for p in vla.qwen_vl_interface.parameters():
        p.requires_grad = False
    for p in vla.action_model.parameters():
        p.requires_grad = False
    if vla.transition_to_action is not None:
        for p in vla.transition_to_action.parameters():
            p.requires_grad = False
    if vla.transition_action_adapter is not None:
        for p in vla.transition_action_adapter.parameters():
            p.requires_grad = False

    # Full parameter breakdown
    total = 0
    trainable_names = []
    for name, p in vla.named_parameters():
        if p.requires_grad:
            n = p.numel()
            total += n
            trainable_names.append((name, n))
    print(f"  Trainable total: {total/1e6:.1f}M across {len(trainable_names)} params")
    # Summarize by module
    from collections import defaultdict
    by_module = defaultdict(float)
    for name, n in trainable_names:
        module = name.split(".")[0]
        by_module[module] += n
    for module, n in sorted(by_module.items()):
        print(f"    {module}: {n/1e6:.2f}M")
    # Verify frozen
    vlm_t = sum(p.numel() for p in vla.qwen_vl_interface.parameters() if p.requires_grad)
    act_t = sum(p.numel() for p in vla.action_model.parameters() if p.requires_grad)
    print(f"  VLM: {'✅' if vlm_t==0 else '❌ '+str(vlm_t/1e6)+'M'}")
    print(f"  Action: {'✅' if act_t==0 else '❌ '+str(act_t/1e6)+'M'}")

    # ── Optimizer ───────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        [p for p in vla.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=1e-5
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.max_steps, eta_min=args.lr * 0.01
    )

    # ── Training loop ────────────────────────────────────────
    data_iter = iter(loader)
    losses_history = {"total": [], "future_mask": [], "goal_mask": [], "relation": []}
    best_loss = float("inf")
    t0 = time.time()

    for step in range(args.max_steps):
        # Get batch
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        vla.train()
        out = vla.forward(batch)
        loss = out["total_loss"]

        if torch.isnan(loss):
            print(f"  Step {step}: NaN loss, skipping")
            optimizer.zero_grad()
            continue

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in vla.parameters() if p.requires_grad], max_norm=1.0
        )
        optimizer.step()
        scheduler.step()

        # Track
        losses_history["total"].append(loss.item())
        for k in ["future_mask_loss", "goal_mask_loss", "relation_loss"]:
            if k in out:
                losses_history[k.replace("_loss", "")].append(out[k].item())

        # Logging
        if step < 10 or step % args.log_interval == 0:
            lr = scheduler.get_last_lr()[0]
            fm = out.get("future_mask_loss", torch.tensor(0)).item()
            gm = out.get("goal_mask_loss", torch.tensor(0)).item()
            rl = out.get("relation_loss", torch.tensor(0)).item()
            print(f"  Step {step:5d}: total={loss.item():.4f}  "
                  f"future={fm:.4f}  goal={gm:.4f}  rel={rl:.4f}  lr={lr:.2e}")

        # Save checkpoint
        if (step + 1) % args.save_interval == 0:
            ckpt_path = output_dir / f"checkpoint_step{step+1}.pt"
            torch.save({
                "step": step + 1,
                "model_state_dict": {k: v for k, v in vla.state_dict().items()
                                     if any(p in k for p in ["vlm_projector", "mask_token_encoder",
                                         "transition_module", "future_mask_decoder",
                                         "goal_mask_decoder", "relation_head",
                                         "transition_to_action", "transition_action_adapter"])},
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss.item(),
            }, str(ckpt_path))
            print(f"  ✅ Checkpoint saved: {ckpt_path}")

        # Eval
        if (step + 1) % args.eval_interval == 0:
            vla.eval()
            eval_losses = {"total": [], "future": [], "goal": [], "relation": []}
            eval_iter = iter(loader)
            with torch.no_grad():
                for _ in range(args.eval_batches):
                    try:
                        eb = next(eval_iter)
                    except StopIteration:
                        break
                    eout = vla.forward(eb)
                    eval_losses["total"].append(eout["total_loss"].item())
                    if "future_mask_loss" in eout:
                        eval_losses["future"].append(eout["future_mask_loss"].item())
                    if "goal_mask_loss" in eout:
                        eval_losses["goal"].append(eout["goal_mask_loss"].item())
                    if "relation_loss" in eout:
                        eval_losses["relation"].append(eout["relation_loss"].item())

            avg_eval = np.mean(eval_losses["total"]) if eval_losses["total"] else float("nan")
            print(f"  📊 Eval {step+1}: total={avg_eval:.4f}  "
                  f"future={np.mean(eval_losses['future']):.4f}  "
                  f"goal={np.mean(eval_losses['goal']):.4f}  "
                  f"rel={np.mean(eval_losses['relation']):.4f}")

            if avg_eval < best_loss:
                best_loss = avg_eval
                best_path = output_dir / "best_model.pt"
                torch.save({k: v for k, v in vla.state_dict().items()
                            if any(p in k for p in ["vlm_projector", "mask_token_encoder",
                                "transition_module", "future_mask_decoder",
                                "goal_mask_decoder", "relation_head",
                                "transition_to_action", "transition_action_adapter"])},
                           str(best_path))
                print(f"  🏆 Best model saved: {best_path}")

    # ── Final save ──────────────────────────────────────────
    elapsed = time.time() - t0
    final_path = output_dir / "final_model.pt"
    torch.save({k: v for k, v in vla.state_dict().items()
                if any(p in k for p in ["vlm_projector", "mask_token_encoder",
                    "transition_module", "future_mask_decoder",
                    "goal_mask_decoder", "relation_head",
                    "transition_to_action", "transition_action_adapter"])},
               str(final_path))

    print(f"\n{'='*60}")
    print(f"P1 Training Complete")
    print(f"  Steps: {args.max_steps}")
    print(f"  Time: {elapsed/60:.1f} min")
    print(f"  Final loss: {losses_history['total'][-1]:.4f}")
    print(f"  Best val loss: {best_loss:.4f}")
    print(f"  Model: {final_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
