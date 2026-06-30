#!/usr/bin/env python
"""
P1 Formal Training: Mask-Conditioned Latent Transition Reasoning.
=================================================================
Multi-GPU via HuggingFace Accelerate.

Trains spatial_transition bottleneck modules (26.5M params) with frozen Qwen-VL
and action model. Supervised by future_mask, goal_mask, relation_label.

Usage:
    # Single GPU smoke
    python scripts/train_p1_latent_transition.py --max-steps 50 --batch-size 2

    # 8 GPU formal training
    accelerate launch --num_processes=8 scripts/train_p1_latent_transition.py \
        --max-steps 50000 --batch-size 4 --lr 3e-4 --output-dir results/P1_run1
"""

import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from accelerate import Accelerator
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


def dice_coef(pred_logits, target, eps=1e-6):
    pred = (torch.sigmoid(pred_logits) > 0.5).float()
    if target.dim() == 2: target = target.unsqueeze(1)
    elif target.dim() == 3: target = target.unsqueeze(1)
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    return ((2.0 * intersection + eps) / (union + eps))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-steps", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--save-interval", type=int, default=5000)
    parser.add_argument("--eval-interval", type=int, default=2000)
    parser.add_argument("--eval-batches", type=int, default=200)
    parser.add_argument("--output-dir", type=str, default=str(_REPO / "results/P1_latent_transition"))
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

    # ── Accelerator ──────────────────────────────────────────
    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        mixed_precision="bf16",
    )
    rank = accelerator.process_index
    world_size = accelerator.num_processes

    if accelerator.is_main_process:
        print("=" * 60)
        print("P1: Mask-Conditioned Latent Transition Training")
        print(f"  Processes: {world_size}")
        print(f"  Global batch size: {args.batch_size * world_size}")
        print(f"  Max steps: {args.max_steps}")
        print(f"  LR: {args.lr}")
        print(f"  Output: {args.output_dir}")
        print("=" * 60)

    output_dir = Path(args.output_dir)
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)

    # ── Build dataloader ────────────────────────────────────
    from laravla.dataloader import build_dataloader
    cfg = OmegaConf.create({"datasets": {"vla_data": {
        "dataset_py": "spatial_cot_libero", "spatial_root": SPATIAL,
        "index_path": IDX, "cot_root": COT,
        "alignment_path": SPATIAL + "/cot_spatial_alignment.json",
        "enable_dynamic_mask": True, "per_device_batch_size": args.batch_size,
        "num_workers": args.num_workers, "state_dim": 7,
    }}})
    loader = build_dataloader(cfg, dataset_py="spatial_cot_libero")
    if accelerator.is_main_process:
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
    vla.training_stage = "latent_transition"

    if accelerator.is_main_process:
        print("  [Training Stage] latent_transition — VLM frozen, Action frozen, Trainable spatial_transition only")

    # Freeze VLM + action_model + P2 adapter
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

    if accelerator.is_main_process:
        total = sum(p.numel() for p in vla.parameters() if p.requires_grad)
        print(f"  Trainable total: {total/1e6:.1f}M")
        for name in ["vlm_projector", "mask_token_encoder", "transition_module",
                      "future_mask_decoder", "goal_mask_decoder", "relation_head"]:
            m = getattr(vla, name, None)
            if m is not None:
                print(f"    {name}: {sum(p.numel() for p in m.parameters())/1e6:.2f}M")
        vlm_t = sum(p.numel() for p in vla.qwen_vl_interface.parameters() if p.requires_grad)
        act_t = sum(p.numel() for p in vla.action_model.parameters() if p.requires_grad)
        print(f"  VLM: {'✅' if vlm_t==0 else '❌'}")
        print(f"  Action: {'✅' if act_t==0 else '❌'}")

    # ── Optimizer ───────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        [p for p in vla.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=1e-5,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.max_steps, eta_min=args.lr * 0.01,
    )

    # ── Accelerator prepare ─────────────────────────────────
    vla, optimizer, loader, scheduler = accelerator.prepare(
        vla, optimizer, loader, scheduler,
    )

    # ── Resume ──────────────────────────────────────────────
    start_step = 0
    if args.resume and os.path.exists(args.resume):
        accelerator.load_state(args.resume)
        start_step = int(Path(args.resume).stem.split("_")[-1])
        if accelerator.is_main_process:
            print(f"  Resumed from step {start_step}")

    # ── Training loop ────────────────────────────────────────
    data_iter = iter(loader)
    best_eval = float("inf")
    t0 = time.time()

    for step in range(start_step, args.max_steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        vla.train()
        out = vla.forward(batch)
        loss = out["total_loss"]

        if torch.isnan(loss):
            if accelerator.is_main_process:
                print(f"  Step {step}: NaN loss, skipping")
            optimizer.zero_grad()
            continue

        optimizer.zero_grad()
        accelerator.backward(loss)
        if accelerator.sync_gradients:
            accelerator.clip_grad_norm_(
                [p for p in vla.parameters() if p.requires_grad], max_norm=1.0,
            )
        optimizer.step()
        scheduler.step()

        # Logging
        if accelerator.is_main_process and (step < 10 or step % 100 == 0):
            fm = out.get("future_mask_loss", torch.tensor(0)).item()
            gm = out.get("goal_mask_loss", torch.tensor(0)).item()
            rl = out.get("relation_loss", torch.tensor(0)).item()
            lr = scheduler.get_last_lr()[0]
            print(f"  Step {step:5d}: total={loss.item():.4f}  "
                  f"future={fm:.4f}  goal={gm:.4f}  rel={rl:.4f}  lr={lr:.2e}")

        # Save
        if accelerator.is_main_process and (step + 1) % args.save_interval == 0:
            ckpt_path = output_dir / f"checkpoint_step{step+1}.pt"
            unwrapped = accelerator.unwrap_model(vla)
            torch.save({
                "step": step + 1,
                "model_state_dict": {k: v for k, v in unwrapped.state_dict().items()
                                     if any(p in k for p in ["vlm_projector", "mask_token_encoder",
                                         "transition_module", "future_mask_decoder",
                                         "goal_mask_decoder", "relation_head"])},
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss.item(),
            }, str(ckpt_path))
            print(f"  ✅ Checkpoint saved: {ckpt_path}")

        # Eval
        if (step + 1) % args.eval_interval == 0:
            vla.eval()
            eval_losses = {"total": [], "future": [], "goal": [], "relation": []}
            eval_dice_f, eval_dice_g = [], []
            eval_rel_correct, eval_rel_total = 0, 0
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

                    if "transition_tokens" in eout:
                        tt = eout["transition_tokens"]
                        unwrapped = accelerator.unwrap_model(vla)
                        fl = unwrapped.future_mask_decoder(tt)
                        gl = unwrapped.goal_mask_decoder(tt)
                        rl = unwrapped.relation_head(tt)

                        fm = torch.from_numpy(np.stack(
                            [ex.get("future_affordance_mask_agentview", np.zeros((224,224), dtype=np.float32)) for ex in eb]
                        )).float().to(fl.device)
                        gm = torch.from_numpy(np.stack(
                            [ex.get("goal_affordance_mask_agentview", np.zeros((224,224), dtype=np.float32)) for ex in eb]
                        )).float().to(gl.device)
                        fm56 = F.interpolate(fm.unsqueeze(1), size=(56,56), mode='nearest').squeeze(1)
                        gm56 = F.interpolate(gm.unsqueeze(1), size=(56,56), mode='nearest').squeeze(1)

                        eval_dice_f.append(dice_coef(fl, fm56).item())
                        eval_dice_g.append(dice_coef(gl, gm56).item())

                        rel_gt = torch.tensor([ex.get("relation_label_id", -1) for ex in eb],
                                              dtype=torch.long, device=rl.device)
                        rel_pred = rl.argmax(dim=1)
                        valid = (rel_gt >= 0) & (rel_gt < rl.shape[1])
                        eval_rel_correct += (rel_pred[valid] == rel_gt[valid]).sum().item()
                        eval_rel_total += valid.sum().item()

            avg_eval = np.mean(eval_losses["total"]) if eval_losses["total"] else float("nan")
            avg_fd = np.mean(eval_dice_f) if eval_dice_f else 0
            avg_gd = np.mean(eval_dice_g) if eval_dice_g else 0
            rel_acc = eval_rel_correct / max(eval_rel_total, 1)

            if accelerator.is_main_process:
                print(f"  📊 Eval {step+1}: total={avg_eval:.4f}  "
                      f"F-Dice={avg_fd:.3f}  G-Dice={avg_gd:.3f}  RelAcc={rel_acc:.3f}")

                # Log metrics
                with open(output_dir / "metrics.jsonl", "a") as f:
                    f.write(json.dumps({
                        "step": step + 1,
                        "val_total": float(avg_eval),
                        "val_future_dice": float(avg_fd),
                        "val_goal_dice": float(avg_gd),
                        "val_relation_acc": float(rel_acc),
                    }) + "\n")

                if avg_eval < best_eval:
                    best_eval = avg_eval
                    unwrapped = accelerator.unwrap_model(vla)
                    best_path = output_dir / "best_model.pt"
                    torch.save({k: v for k, v in unwrapped.state_dict().items()
                                if any(p in k for p in ["vlm_projector", "mask_token_encoder",
                                    "transition_module", "future_mask_decoder",
                                    "goal_mask_decoder", "relation_head"])},
                               str(best_path))
                    print(f"  🏆 Best model saved (val={best_eval:.4f})")

    # ── Final save ──────────────────────────────────────────
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        elapsed = time.time() - t0
        unwrapped = accelerator.unwrap_model(vla)
        final_path = output_dir / "final_model.pt"
        torch.save({k: v for k, v in unwrapped.state_dict().items()
                    if any(p in k for p in ["vlm_projector", "mask_token_encoder",
                        "transition_module", "future_mask_decoder",
                        "goal_mask_decoder", "relation_head"])},
                   str(final_path))
        print(f"\n{'='*60}")
        print(f"P1 Training Complete")
        print(f"  Steps: {args.max_steps}")
        print(f"  Time: {elapsed/60:.1f} min")
        print(f"  Best val loss: {best_eval:.4f}")
        print(f"  Model: {final_path}")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
