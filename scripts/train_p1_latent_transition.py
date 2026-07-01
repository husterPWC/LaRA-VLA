#!/usr/bin/env python
"""
P1 Formal Training: DDP-safe via P1TransitionWrapper.
======================================================
Only wraps 26.5M P1 modules in DDP. Qwen-VL stays outside as frozen local encoder.

Usage:
    python scripts/train_p1_latent_transition.py --max-steps 50
    accelerate launch --num_processes=8 scripts/train_p1_latent_transition.py ...
"""

import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
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
    args = parser.parse_args()

    # ── Accelerator ──────────────────────────────────────────
    accelerator = Accelerator(gradient_accumulation_steps=1, mixed_precision="bf16")
    if torch.cuda.is_available():
        torch.cuda.set_device(accelerator.local_process_index)

    if accelerator.is_main_process:
        print("=" * 60)
        print("P1: Mask-Conditioned Latent Transition (DDP-safe)")
        print(f"  Processes: {accelerator.num_processes}")
        print(f"  Global batch: {args.batch_size * accelerator.num_processes}")
        print(f"  Max steps: {args.max_steps}  LR: {args.lr}")
        print(f"  Output: {args.output_dir}")
        print("=" * 60)

    output_dir = Path(args.output_dir)
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)

    # ── Dataloader ──────────────────────────────────────────
    from laravla.dataloader import build_dataloader
    loader = build_dataloader(OmegaConf.create({"datasets": {"vla_data": {
        "dataset_py": "spatial_cot_libero", "spatial_root": SPATIAL,
        "index_path": IDX, "cot_root": COT,
        "alignment_path": SPATIAL + "/cot_spatial_alignment.json",
        "enable_dynamic_mask": True, "per_device_batch_size": args.batch_size,
        "num_workers": 2, "state_dim": 7,
    }}}), dataset_py="spatial_cot_libero")
    if accelerator.is_main_process:
        print(f"  DataLoader: {len(loader)} batches/epoch")

    # ── Build FULL VLA (frozen local encoder) ────────────────
    from laravla.model.tools import read_mode_config
    model_cfg, _ = read_mode_config(Path(CKPT))
    model_cfg["framework"]["mask_conditioned_transition"] = {
        "enable": True, "num_mask_tokens": 8, "num_transition_tokens": 6,
        "mask_res": 56, "num_relation_labels": 6, "transition_dim": 512,
        "loss_weights": {"future_mask": 0.05, "goal_mask": 0.10, "relation": 0.05},
    }
    cfg = OmegaConf.create(model_cfg)
    from laravla.model.framework import build_framework
    vla = build_framework(cfg)
    vla.load_state_dict(torch.load(CKPT, map_location="cpu"), strict=False)
    vla = vla.to(accelerator.device)
    vla.eval()

    # ── Freeze all → unfreeze only P1 in wrapper ─────────────
    for p in vla.parameters():
        p.requires_grad_(False)

    from laravla.model.modules.spatial_transition import P1TransitionWrapper
    p1_model = P1TransitionWrapper(vla).to(accelerator.device)
    for p in p1_model.parameters():
        p.requires_grad_(True)

    if accelerator.is_main_process:
        n = sum(p.numel() for p in p1_model.parameters())
        print(f"  P1Wrapper trainable: {n/1e6:.1f}M")
        print(f"  VLM: frozen (no_grad encode only), Action: frozen")
        print(f"  DDP wraps: P1TransitionWrapper only")

    # ── Optimizer ───────────────────────────────────────────
    optimizer = torch.optim.AdamW(p1_model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.max_steps, eta_min=args.lr * 0.01)

    # ── DDP only wraps p1_model ─────────────────────────────
    p1_model, optimizer, loader, scheduler = accelerator.prepare(
        p1_model, optimizer, loader, scheduler)

    # ── Training ────────────────────────────────────────────
    data_iter = iter(loader)
    best_eval = float("inf")
    t0 = time.time()

    for step in range(args.max_steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        # Extract tensors
        cur_masks = torch.from_numpy(
            np.stack([s["current_affordance_mask_agentview"] for s in batch])
        ).unsqueeze(1).to(accelerator.device).float()
        future_masks = torch.from_numpy(
            np.stack([s.get("future_affordance_mask_agentview", np.zeros((224,224), dtype=np.float32)) for s in batch])
        ).to(accelerator.device).float()
        goal_masks = torch.from_numpy(
            np.stack([s.get("goal_affordance_mask_agentview", np.zeros((224,224), dtype=np.float32)) for s in batch])
        ).to(accelerator.device).float()
        rel_ids = torch.tensor([s.get("relation_label_id", -1) for s in batch],
                               dtype=torch.long, device=accelerator.device)

        # Frozen VLM encode
        with torch.no_grad():
            qwen_out = vla.qwen_vl_interface.encode_observation(
                images=[s["image"] for s in batch],
                instructions=[s["lang"] for s in batch],
                output_hidden_states=True,
            )
            vlm_hidden = qwen_out.hidden_states[-1]  # bfloat16 → p1 wrapper handles float conversion

        # P1 forward
        out = p1_model(vlm_hidden, cur_masks, future_masks, goal_masks, rel_ids)
        loss = out["total_loss"]

        if torch.isnan(loss):
            if accelerator.is_main_process:
                print(f"  Step {step}: NaN, skip")
            optimizer.zero_grad()
            continue

        optimizer.zero_grad()
        accelerator.backward(loss)
        if accelerator.sync_gradients:
            accelerator.clip_grad_norm_(p1_model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        # Log
        if accelerator.is_main_process and (step < 10 or step % 100 == 0):
            fm = out["future_mask_loss"].item()
            gm = out["goal_mask_loss"].item()
            rl = out["relation_loss"].item()
            print(f"  Step {step:5d}: total={loss.item():.4f}  "
                  f"F={fm:.4f}  G={gm:.4f}  R={rl:.4f}  lr={scheduler.get_last_lr()[0]:.2e}")

        # Save
        if accelerator.is_main_process and (step + 1) % args.save_interval == 0:
            unwrapped = accelerator.unwrap_model(p1_model)
            ckpt_path = output_dir / f"checkpoint_step{step+1}.pt"
            torch.save({"step": step + 1, "p1_state_dict": unwrapped.state_dict()}, str(ckpt_path))
            print(f"  ✅ Checkpoint: {ckpt_path}")

        # Eval
        if (step + 1) % args.eval_interval == 0:
            p1_model.eval()
            eval_tot, eval_fd, eval_gd, eval_ra = [], [], [], []
            eval_iter = iter(loader)
            with torch.no_grad():
                for _ in range(args.eval_batches):
                    try:
                        eb = next(eval_iter)
                    except StopIteration:
                        break
                    cm = torch.from_numpy(np.stack(
                        [s["current_affordance_mask_agentview"] for s in eb]
                    )).unsqueeze(1).to(accelerator.device).float()
                    fm = torch.from_numpy(np.stack(
                        [s.get("future_affordance_mask_agentview", np.zeros((224,224), dtype=np.float32)) for s in eb]
                    )).to(accelerator.device).float()
                    gm = torch.from_numpy(np.stack(
                        [s.get("goal_affordance_mask_agentview", np.zeros((224,224), dtype=np.float32)) for s in eb]
                    )).to(accelerator.device).float()
                    ri = torch.tensor([s.get("relation_label_id", -1) for s in eb],
                                      dtype=torch.long, device=accelerator.device)
                    qo = vla.qwen_vl_interface.encode_observation(
                        images=[s["image"] for s in eb],
                        instructions=[s["lang"] for s in eb],
                        output_hidden_states=True)
                    vh = qo.hidden_states[-1]
                    eo = p1_model(vh, cm, fm, gm, ri)
                    eval_tot.append(eo["total_loss"].item())
                    eval_fd.append(eo["future_dice"].item())
                    eval_gd.append(eo["goal_dice"].item())
                    eval_ra.append(eo["relation_acc"].item())

            if accelerator.is_main_process:
                avg_t = np.mean(eval_tot) if eval_tot else float("nan")
                avg_fd = np.mean(eval_fd) if eval_fd else 0
                avg_gd = np.mean(eval_gd) if eval_gd else 0
                avg_ra = np.mean(eval_ra) if eval_ra else 0
                print(f"  📊 Eval {step+1}: loss={avg_t:.4f}  "
                      f"F-Dice={avg_fd:.3f}  G-Dice={avg_gd:.3f}  RelAcc={avg_ra:.3f}")
                with open(output_dir / "metrics.jsonl", "a") as f:
                    f.write(json.dumps({"step": step + 1, "val_loss": float(avg_t),
                                        "F_Dice": float(avg_fd), "G_Dice": float(avg_gd),
                                        "RelAcc": float(avg_ra)}) + "\n")
                if avg_t < best_eval:
                    best_eval = avg_t
                    unwrapped = accelerator.unwrap_model(p1_model)
                    torch.save({"p1_state_dict": unwrapped.state_dict()}, str(output_dir / "best_model.pt"))
                    print(f"  🏆 Best saved (val={best_eval:.4f})")

    # ── Final ───────────────────────────────────────────────
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(p1_model)
        torch.save({"p1_state_dict": unwrapped.state_dict()}, str(output_dir / "final_model.pt"))
        print(f"\n{'='*60}\nP1 Complete\n  Best val: {best_eval:.4f}\n  Time: {(time.time()-t0)/60:.0f}min\n{'='*60}")


if __name__ == "__main__":
    main()
