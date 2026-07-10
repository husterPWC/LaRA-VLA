#!/usr/bin/env python
"""
P2 Formal Training: Transition-conditioned Action Generation.
==============================================================
Loads P1 best checkpoint, trains GatedTransitionActionAdapter + action model.
Qwen-VL stays frozen. Transition modules optionally fine-tuned.

Usage:
    python scripts/train_p2_transition_action.py --max-steps 100 --p1-ckpt results/P1_formal/best_model.pt

    accelerate launch --num_processes=5 scripts/train_p2_transition_action.py \
        --max-steps 80000 --p1-ckpt results/P1_formal/best_model.pt ...
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

RELATION_NAMES = {0: "approach", 1: "grasp", 2: "release",
                  3: "place_inside", 4: "place_on_top",
                  5: "open_articulated", 6: "move_toward"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-steps", type=int, default=80000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--p1-ckpt", type=str, required=True)
    parser.add_argument("--save-interval", type=int, default=5000)
    parser.add_argument("--eval-interval", type=int, default=2000)
    parser.add_argument("--eval-batches", type=int, default=200)
    parser.add_argument("--output-dir", type=str, default=str(_REPO / "results/P2_transition_action"))
    parser.add_argument("--freeze-transition", action="store_true",
                        help="Freeze P1 transition modules, train only adapter+action")
    args = parser.parse_args()

    accelerator = Accelerator(gradient_accumulation_steps=1, mixed_precision="bf16")
    if torch.cuda.is_available():
        torch.cuda.set_device(accelerator.local_process_index)

    if accelerator.is_main_process:
        print("=" * 60)
        print("P2: Transition-Conditioned Action Generation")
        print(f"  Processes: {accelerator.num_processes}")
        print(f"  P1 checkpoint: {args.p1_ckpt}")
        print(f"  Max steps: {args.max_steps}  LR: {args.lr}")
        print(f"  Freeze transition: {args.freeze_transition}")
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

    # ── Build VLA + load P1 weights ─────────────────────────
    from laravla.model.tools import read_mode_config
    model_cfg, _ = read_mode_config(Path(CKPT))
    model_cfg["framework"]["mask_conditioned_transition"] = {
        "enable": True, "num_mask_tokens": 8, "num_transition_tokens": 6,
        "mask_res": 56, "num_relation_labels": 6, "transition_dim": 512,
        "loss_weights": {"future_mask": 0.05, "goal_mask": 0.10, "relation": 0.05},
    }
    from laravla.model.framework import build_framework
    vla = build_framework(OmegaConf.create(model_cfg))
    vla.load_state_dict(torch.load(CKPT, map_location="cpu"), strict=False)

    # Inject P1 trained weights
    p1_state = torch.load(args.p1_ckpt, map_location="cpu")
    if "p1_state_dict" in p1_state:
        p1_state = p1_state["p1_state_dict"]
    vla.load_state_dict(p1_state, strict=False)
    vla = vla.to(accelerator.device)
    vla.training_stage = "transition_action"

    if accelerator.is_main_process:
        print("  P1 weights loaded.")

    # ── Freeze / Unfreeze ────────────────────────────────────
    # Qwen-VL always frozen
    for p in vla.qwen_vl_interface.parameters():
        p.requires_grad_(False)

    # P1 transition modules
    p1_modules = [vla.vlm_projector, vla.mask_token_encoder, vla.transition_module,
                   vla.future_mask_decoder, vla.goal_mask_decoder, vla.relation_head]
    if args.freeze_transition:
        for m in p1_modules:
            for p in m.parameters():
                p.requires_grad_(False)
    else:
        for m in p1_modules:
            for p in m.parameters():
                p.requires_grad_(True)

    # Adapter + action model always trainable
    for p in vla.transition_action_adapter.parameters():
        p.requires_grad_(True)
    for p in vla.action_model.parameters():
        p.requires_grad_(True)

    trainable = sum(p.numel() for p in vla.parameters() if p.requires_grad)
    if accelerator.is_main_process:
        print(f"  Trainable: {trainable/1e6:.1f}M")
        print(f"  VLM: frozen | P1 modules: {'frozen' if args.freeze_transition else 'trainable'}")
        print(f"  Adapter: trainable | Action: trainable")

    # ── Optimizer ───────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        [p for p in vla.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.max_steps, eta_min=args.lr * 0.01)

    # ── Prepare ─────────────────────────────────────────────
    vla, optimizer, loader, scheduler = accelerator.prepare(
        vla, optimizer, loader, scheduler)

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

        vla.train()
        out = vla.forward(batch)
        loss = out["total_loss"]

        if torch.isnan(loss):
            if accelerator.is_main_process:
                print(f"  Step {step}: NaN, skip")
            optimizer.zero_grad()
            continue

        optimizer.zero_grad()
        accelerator.backward(loss)
        if accelerator.sync_gradients:
            accelerator.clip_grad_norm_(
                [p for p in vla.parameters() if p.requires_grad], max_norm=1.0)
        optimizer.step()
        scheduler.step()

        # Log
        if accelerator.is_main_process and (step < 10 or step % 100 == 0):
            al = out.get("action_loss", torch.tensor(0)).item()
            fm = out.get("future_mask_loss", torch.tensor(0)).item()
            gm = out.get("goal_mask_loss", torch.tensor(0)).item()
            rl = out.get("relation_loss", torch.tensor(0)).item()
            gate = torch.tanh(vla.module.transition_action_adapter.gate).item() if hasattr(vla, 'module') else torch.tanh(vla.transition_action_adapter.gate).item()
            print(f"  Step {step:5d}: total={loss.item():.4f} action={al:.4f} "
                  f"F={fm:.4f} G={gm:.4f} R={rl:.4f} gate={gate:.4f} lr={scheduler.get_last_lr()[0]:.2e}")

        # Save
        if accelerator.is_main_process and (step + 1) % args.save_interval == 0:
            unwrapped = accelerator.unwrap_model(vla)
            ckpt_path = output_dir / f"checkpoint_step{step+1}.pt"
            torch.save({"step": step + 1, "model_state_dict": unwrapped.state_dict()}, str(ckpt_path))
            print(f"  ✅ Checkpoint: {ckpt_path}")

        # Eval
        if (step + 1) % args.eval_interval == 0:
            vla.eval()
            eval_al, eval_tot = [], []
            eval_iter = iter(loader)
            with torch.no_grad():
                for _ in range(args.eval_batches):
                    try:
                        eb = next(eval_iter)
                    except StopIteration:
                        break
                    eo = vla.forward(eb)
                    eval_tot.append(eo["total_loss"].item())
                    if "action_loss" in eo:
                        eval_al.append(eo["action_loss"].item())

            if accelerator.is_main_process:
                avg_tot = np.mean(eval_tot) if eval_tot else float("nan")
                avg_al = np.mean(eval_al) if eval_al else float("nan")
                unwrapped = accelerator.unwrap_model(vla)
                gate_val = torch.tanh(unwrapped.transition_action_adapter.gate).item()
                print(f"  📊 Eval {step+1}: total={avg_tot:.4f} action={avg_al:.4f} gate={gate_val:.4f}")
                with open(output_dir / "metrics.jsonl", "a") as f:
                    f.write(json.dumps({"step": step + 1, "val_total": float(avg_tot),
                                        "val_action": float(avg_al), "gate": float(gate_val)}) + "\n")
                if avg_tot < best_eval:
                    best_eval = avg_tot
                    torch.save({"model_state_dict": unwrapped.state_dict()},
                               str(output_dir / "best_model.pt"))
                    print(f"  🏆 Best saved (val={best_eval:.4f})")

    # ── Final ───────────────────────────────────────────────
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(vla)
        torch.save({"model_state_dict": unwrapped.state_dict()},
                   str(output_dir / "final_model.pt"))
        print(f"\n{'='*60}\nP2 Complete\n  Best val: {best_eval:.4f}\n  Time: {(time.time()-t0)/60:.0f}min\n{'='*60}")


if __name__ == "__main__":
    main()
