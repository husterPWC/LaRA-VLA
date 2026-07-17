#!/usr/bin/env python
"""
Unified Spatial Action Training — single process, two phases.
===============================================================
Phase 1: train SpatialTransitionBackbone (mask/relation)
Phase 2: freeze backbone, train SpatialActionAdapter + ActionHead

No cross-process checkpoint loading — phase switch is in-place.

Usage:
    python scripts/train_spatial_action.py --p1-steps 2000 --p2-steps 500
"""

import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

_THIS = Path(__file__).resolve()
_REPO = _THIS.parents[1]
sys.path.insert(0, str(_REPO))

import warnings; warnings.filterwarnings("ignore")

CKPT = os.environ.get("LARAVLA_CKPT",
    str(_REPO.parent / "models/LaRA-VLA-libero/checkpoints/steps_25000_pytorch_model.pt"))
SPATIAL = str(_REPO / "output" / "spatial_lara_libero")
IDX = str(_REPO / "output" / "spatial_lara_libero_no_noops" /
          "spatial_lara_libero_index_cot_transition_all_fixed_v4_tau.jsonl")
COT = os.environ.get("LEROBOT_ROOT",
    str(_REPO.parent / "datasets" / "lovejuly" / "libero_lerobot_all"))


def build_model(args):
    """Build the single FormalSpatialActionModel."""
    from laravla.model.tools import read_mode_config
    from omegaconf import OmegaConf
    model_cfg, _ = read_mode_config(Path(CKPT))
    model_cfg["framework"]["mask_conditioned_transition"] = {
        "enable": True, "num_transition_tokens": 6,
        "mask_res": 56, "num_relation_labels": 7, "transition_dim": 512,
        "loss_weights": {"future_mask": 0.05, "goal_mask": 0.10, "relation": 0.05,
                         "current_mask": 0.05,
                         "slot_residual_gamma": args.gamma},
    }
    from laravla.model.framework import build_framework
    vla = build_framework(OmegaConf.create(model_cfg))
    vla.load_state_dict(torch.load(CKPT, map_location="cpu"), strict=False)
    return vla


def phase1_train(args, vla, loader, output_dir):
    """Train SpatialTransitionBackbone."""
    from laravla.model.modules.spatial_transition import (
        SpatialTransitionBackbone, P1NoMaskWrapper
    )
    backbone = SpatialTransitionBackbone(
        vlm_dim=vla.qwen_vl_interface.model.config.hidden_size,
        hidden_dim=512, num_slots=6, gamma=args.gamma,
    )
    posterior = vla.posterior_encoder
    p1_model = P1NoMaskWrapper(
        backbone=backbone, posterior_encoder=posterior,
        loss_weights=vla.transition_loss_weights,
    ).to("cuda")
    for p in p1_model.parameters():
        p.requires_grad_(True)
    # Freeze VLM
    vla.qwen_vl_interface.eval()
    for p in vla.qwen_vl_interface.parameters():
        p.requires_grad_(False)

    opt = torch.optim.AdamW(p1_model.parameters(), lr=args.p1_lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.p1_steps, eta_min=args.p1_lr * 0.01)

    best_score = -float("inf")
    best_state = None
    t0 = time.time()
    data_iter = iter(loader)
    for step in range(args.p1_steps):
        try: batch = next(data_iter)
        except StopIteration: data_iter = iter(loader); batch = next(data_iter)

        # Prepare batch
        images = [s["image"] for s in batch]
        instructions = [s["lang"] for s in batch]
        cur_masks = torch.from_numpy(np.stack([s["current_affordance_mask_agentview"] for s in batch])).unsqueeze(1).to("cuda").float()
        fut_masks = torch.from_numpy(np.stack([s.get("future_tau_mask_agentview", s.get("future_affordance_mask_agentview", np.zeros((224,224),dtype=np.float32))) for s in batch])).to("cuda").float()
        gl_masks = torch.from_numpy(np.stack([s["goal_affordance_mask_agentview"] for s in batch])).to("cuda").float()
        rel_ids = torch.tensor([s["relation_label_id"] for s in batch], dtype=torch.long, device="cuda")

        with torch.no_grad():
            qo = vla.qwen_vl_interface.encode_observation(images=images, instructions=instructions, output_hidden_states=True)
            vlm_hidden = qo.hidden_states[-1]

        p1_model.train()
        out = p1_model(vlm_hidden, cur_masks, fut_masks, gl_masks, rel_ids)
        loss = out["total_loss"]
        if torch.isnan(loss): opt.zero_grad(); continue

        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(p1_model.parameters(), 1.0)
        opt.step(); sched.step()

        if step % 100 == 0 or step == args.p1_steps - 1:
            cd = out.get("current_dice", torch.tensor(0)).item()
            fd = out.get("future_dice", torch.tensor(0)).item()
            print(f"  P1 {step:4d}: loss={loss.item():.4f}  C={cd:.2f} F={fd:.2f}  lr={sched.get_last_lr()[0]:.2e}")

        # Eval + save best
        if (step + 1) % args.eval_interval == 0:
            p1_model.eval()
            eval_cd, eval_fd, eval_gd, eval_ra = [], [], [], []
            eval_iter = iter(loader)
            with torch.no_grad():
                for _ in range(min(args.eval_batches, 50)):
                    try: eb = next(eval_iter)
                    except StopIteration: break
                    ecm = torch.from_numpy(np.stack([s["current_affordance_mask_agentview"] for s in eb])).unsqueeze(1).to("cuda").float()
                    efm = torch.from_numpy(np.stack([s.get("future_tau_mask_agentview", s.get("future_affordance_mask_agentview", np.zeros((224,224),dtype=np.float32))) for s in eb])).to("cuda").float()
                    egm = torch.from_numpy(np.stack([s["goal_affordance_mask_agentview"] for s in eb])).to("cuda").float()
                    eri = torch.tensor([s["relation_label_id"] for s in eb], dtype=torch.long, device="cuda")
                    qo = vla.qwen_vl_interface.encode_observation(images=[s["image"] for s in eb], instructions=[s["lang"] for s in eb], output_hidden_states=True)
                    evh = qo.hidden_states[-1]
                    eo = p1_model(evh, ecm, efm, egm, eri)
                    eval_cd.append(eo["current_dice"].item())
                    eval_fd.append(eo["future_dice"].item())
                    eval_gd.append(eo["goal_dice"].item())
                    eval_ra.append(eo["relation_acc"].item())
            avg_score = (np.mean(eval_cd)+np.mean(eval_fd)+np.mean(eval_gd))/3 + 0.2*np.mean(eval_ra)
            print(f"  📊 Eval {step+1}: C={np.mean(eval_cd):.3f} F={np.mean(eval_fd):.3f} G={np.mean(eval_gd):.3f} score={avg_score:.3f}")
            if avg_score > best_score:
                best_score = avg_score
                best_state = {k: v.cpu().clone() for k, v in p1_model.state_dict().items()}
                print(f"  🏆 Best P1 (score={best_score:.4f})")
            p1_model.train()

    print(f"\n  P1 Complete. Best score={best_score:.4f}  Time: {(time.time()-t0)/60:.0f}min")
    return best_state, p1_model


def phase_switch_parity(vla, p1_model, loader):
    """Verify phase switch doesn't change any forward output."""
    print(f"\n{'='*60}")
    print("Phase Switch Parity Check")
    print(f"{'='*60}")

    batch = next(iter(loader))
    images = [s["image"] for s in batch]
    instructions = [s["lang"] for s in batch]
    cur_masks = torch.from_numpy(np.stack([s["current_affordance_mask_agentview"] for s in batch])).unsqueeze(1).to("cuda").float()
    fut_masks = torch.from_numpy(np.stack([s.get("future_tau_mask_agentview", s.get("future_affordance_mask_agentview", np.zeros((224,224),dtype=np.float32))) for s in batch])).to("cuda").float()
    gl_masks = torch.from_numpy(np.stack([s["goal_affordance_mask_agentview"] for s in batch])).to("cuda").float()
    rel_ids = torch.tensor([s["relation_label_id"] for s in batch], dtype=torch.long, device="cuda")

    with torch.no_grad():
        qo = vla.qwen_vl_interface.encode_observation(images=images, instructions=instructions, output_hidden_states=True)
        vlm_hidden = qo.hidden_states[-1]

    # Before switch
    p1_model.eval()
    out_before = p1_model(vlm_hidden, cur_masks, fut_masks, gl_masks, rel_ids)

    # Save reference tensors
    ref = {
        "z_student": out_before["transition_tokens"].clone(),
        "current_dice": out_before["current_dice"].clone(),
        "future_dice": out_before["future_dice"].clone(),
        "goal_dice": out_before["goal_dice"].clone(),
    }

    # In-phase reload (simulates what P2 would do)
    from laravla.model.modules.spatial_transition import SpatialTransitionBackbone, P1NoMaskWrapper
    backbone2 = SpatialTransitionBackbone(vlm_dim=vla.qwen_vl_interface.model.config.hidden_size, hidden_dim=512, num_slots=6, gamma=1.5)
    backbone2.load_state_dict(p1_model.backbone.state_dict())
    backbone2.eval()
    for p in backbone2.parameters():
        p.requires_grad_(False)
    p1_reload = P1NoMaskWrapper(backbone=backbone2, loss_weights=vla.transition_loss_weights).to("cuda")
    p1_reload.eval()

    out_after = p1_reload(vlm_hidden, cur_masks, fut_masks, gl_masks, rel_ids)

    z_diff = (ref["z_student"] - out_after["transition_tokens"]).abs().max().item()

    print(f"  z_student max_abs_diff:  {z_diff:.2e} {'✅' if z_diff<1e-5 else '❌'}")

    if z_diff >= 1e-5:
        raise RuntimeError(f"Phase switch parity FAILED: z_diff={z_diff:.2e}")

    print(f"  ✅ Phase switch parity PASSED")
    return p1_reload


def phase2_train(args, vla, p1_model, loader, output_dir):
    """Train SpatialActionAdapter + ActionHead with frozen backbone."""
    print(f"\n{'='*60}")
    print("Phase 2: Action Training")
    print(f"{'='*60}")

    # Freeze backbone
    for p in p1_model.backbone.parameters():
        p.requires_grad_(False)
    p1_model.backbone.eval()

    # Trainable: adapter + action
    for p in vla.transition_action_adapter.parameters():
        p.requires_grad_(True)
    for p in vla.proprio_encoder.parameters():
        p.requires_grad_(True)
    for p in vla.action_model.parameters():
        p.requires_grad_(True)

    trainable_params = [p for n, p in vla.named_parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable_params, lr=args.p2_lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.p2_steps, eta_min=args.p2_lr*0.01)

    vla.train()
    vla.training_stage = "transition_action_nomask"

    best_action = float("inf")
    data_iter = iter(loader)

    # Eval step 0 baseline
    vla.eval()
    eval_al0 = []
    eval_iter = iter(loader)
    with torch.no_grad():
        for _ in range(min(args.eval_batches, 50)):
            try: eb = next(eval_iter)
            except StopIteration: break
            eo = vla.forward(eb)
            eval_al0.append(eo.get("action_loss", torch.tensor(0)).item())
    print(f"  📊 Eval step 0: action={np.mean(eval_al0):.4f}")
    # Re-create data_iter for training (eval consumed batches)
    data_iter = iter(loader)

    for step in range(args.p2_steps):
        try: batch = next(data_iter)
        except StopIteration: data_iter = iter(loader); batch = next(data_iter)

        vla.train()
        p1_model.backbone.eval()

        out = vla.forward(batch)
        loss = out["total_loss"]
        if torch.isnan(loss): opt.zero_grad(); continue

        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
        opt.step(); sched.step()

        al = out.get("action_loss", torch.tensor(0)).item()
        gate = torch.sigmoid(vla.transition_action_adapter.gate_logit).item()
        if step % 50 == 0 or step == args.p2_steps - 1:
            print(f"  P2 {step:4d}: action={al:.4f} gate={gate:.3f} lr={sched.get_last_lr()[0]:.2e}")

        if (step + 1) % args.eval_interval == 0:
            vla.eval()
            eval_al = []
            eval_iter = iter(loader)
            with torch.no_grad():
                for _ in range(min(args.eval_batches, 50)):
                    try: eb = next(eval_iter)
                    except StopIteration: break
                    eo = vla.forward(eb)
                    eval_al.append(eo.get("action_loss", torch.tensor(0)).item())
            avg_al = np.mean(eval_al) if eval_al else 0
            print(f"  📊 Eval {step+1}: action={avg_al:.4f}")
            if avg_al < best_action:
                best_action = avg_al
                torch.save({"model_state_dict": {k:v.cpu() for k,v in vla.state_dict().items() if any(p in k for p in ['spatial_backbone','transition_action_adapter','proprio_encoder','action_model'])}}, str(output_dir / "best_p2.pt"))
                print(f"  🏆 Best P2 (action={best_action:.4f})")
            vla.train()

    print(f"\n  P2 Complete. Best action={best_action:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--p1-steps", type=int, default=2000)
    parser.add_argument("--p1-lr", type=float, default=3e-4)
    parser.add_argument("--p2-steps", type=int, default=500)
    parser.add_argument("--p2-lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gamma", type=float, default=1.5)
    parser.add_argument("--w-distill", type=float, default=0.0)
    parser.add_argument("--gate-init", type=float, default=-2.2)
    parser.add_argument("--eval-interval", type=int, default=200)
    parser.add_argument("--eval-batches", type=int, default=50)
    parser.add_argument("--output-dir", type=str, default=str(_REPO / "results/SpatialAction_formal"))
    args = parser.parse_args()

    output_dir = Path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Unified Spatial Action Training")
    print(f"  P1: {args.p1_steps} steps, lr={args.p1_lr}")
    print(f"  P2: {args.p2_steps} steps, lr={args.p2_lr}")
    print(f"  gamma={args.gamma}  w_distill={args.w_distill}  gate_init={args.gate_init}")
    print("=" * 60)

    # Build
    vla = build_model(args)
    vla = vla.to("cuda")
    vla.transition_action_adapter.gate_logit.data.fill_(args.gate_init)

    from laravla.dataloader import build_dataloader
    from omegaconf import OmegaConf
    loader = build_dataloader(OmegaConf.create({"datasets": {"vla_data": {
        "dataset_py": "spatial_cot_libero", "spatial_root": SPATIAL,
        "index_path": IDX, "cot_root": COT,
        "alignment_path": SPATIAL + "/cot_spatial_alignment.json",
        "enable_dynamic_mask": True, "per_device_batch_size": args.batch_size,
        "num_workers": 2, "state_dim": 7,
    }}}), dataset_py="spatial_cot_libero")

    # Phase 1
    best_p1, p1_model = phase1_train(args, vla, loader, output_dir)
    torch.save({"p1_state_dict": best_p1}, str(output_dir / "best_p1.pt"))

    # Phase switch parity
    p1_model = phase_switch_parity(vla, p1_model, loader)

    # Phase 2
    phase2_train(args, vla, p1_model, loader, output_dir)

    print(f"\n{'='*60}\nTraining Complete\n{'='*60}")


if __name__ == "__main__":
    main()
