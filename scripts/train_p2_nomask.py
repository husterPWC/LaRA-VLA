#!/usr/bin/env python
"""
P2-New Formal Training: Transition-Conditioned Action Generation (no-mask).
============================================================================
Loads P1-New best checkpoint. No mask_token_encoder — all masks are supervision only.
Transition tokens from VLM hidden only → gated adapter → action generation.

Qwen-VL stays frozen. Transition modules + adapter + action model trainable.
Auxiliary spatial losses provide regularization.

Usage:
    python scripts/train_p2_nomask.py --max-steps 100 --p1-ckpt results/P1_nomask/best_model.pt

    accelerate launch --num_processes=6 scripts/train_p2_nomask.py \
        --max-steps 80000 --p1-ckpt results/P1_nomask/best_model.pt
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
    parser.add_argument("--max-steps", type=int, default=80000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--p1-ckpt", type=str, required=True,
                        help="Path to P1-New (no-mask) checkpoint")
    parser.add_argument("--save-interval", type=int, default=5000)
    parser.add_argument("--eval-interval", type=int, default=2000)
    parser.add_argument("--eval-batches", type=int, default=200)
    parser.add_argument("--output-dir", type=str, default=str(_REPO / "results/P2_nomask"))
    parser.add_argument("--freeze-transition", action="store_true", default=True,
                        help="Freeze P1 transition modules (default True for Stage 6A)")
    parser.add_argument("--no-freeze-transition", action="store_true",
                        help="Disable freeze (Stage 6B fine-tuning)")
    parser.add_argument("--aux-weight", type=float, default=0.05,
                        help="Weight for auxiliary spatial losses in P2 (default 0.05)")
    parser.add_argument("--gate-init", type=float, default=-2.2,
                        help="Gate logit init: -2.2→sig≈0.1, -10→sig≈0 (ablation)")
    args = parser.parse_args()
    if args.no_freeze_transition:
        args.freeze_transition = False

    accelerator = Accelerator(gradient_accumulation_steps=1, mixed_precision="bf16")
    if torch.cuda.is_available():
        torch.cuda.set_device(accelerator.local_process_index)

    if accelerator.is_main_process:
        print("=" * 60)
        print("P2-New: Transition-Conditioned Action (no-mask)")
        print(f"  Processes: {accelerator.num_processes}")
        print(f"  P1-New checkpoint: {args.p1_ckpt}")
        print(f"  Max steps: {args.max_steps}  LR: {args.lr}")
        print(f"  Aux weight: {args.aux_weight}")
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

    # ── Build VLA + load P1-New weights ─────────────────────
    from laravla.model.tools import read_mode_config
    model_cfg, _ = read_mode_config(Path(CKPT))
    model_cfg["framework"]["mask_conditioned_transition"] = {
        "enable": True, "num_mask_tokens": 8, "num_transition_tokens": 6,
        "mask_res": 56, "num_relation_labels": 7, "transition_dim": 512,
        "loss_weights": {"future_mask": 0.05, "goal_mask": 0.10, "relation": 0.05,
                         "current_mask": 0.05, "dino_future": 0.05,
                         "slot_residual_gamma": 1.5},
    }
    from laravla.model.framework import build_framework
    vla = build_framework(OmegaConf.create(model_cfg))
    vla.load_state_dict(torch.load(CKPT, map_location="cpu"), strict=False)

    # Inject P1-New trained weights
    p1_state = torch.load(args.p1_ckpt, map_location="cpu")
    if "p1_state_dict" in p1_state:
        p1_state = p1_state["p1_state_dict"]

    # Remap P1NoMaskWrapper shared mask_decoder → all three VLA decoders
    remapped = {}
    p1_aux_keys = []
    for k, v in p1_state.items():
        if k.startswith("mask_decoder."):
            # Map shared → current, future, goal (all three get same weights)
            for prefix in ["current_mask_decoder.", "future_mask_decoder.", "goal_mask_decoder."]:
                remapped[k.replace("mask_decoder.", prefix)] = v
            p1_aux_keys.append(k)
        else:
            remapped[k] = v
    missing, unexpected = vla.load_state_dict(remapped, strict=False)

    vla = vla.to(accelerator.device)
    vla.training_stage = "transition_action_nomask"

    # Apply gate initialization (supports ablation: -2.2≈0.1, -10≈0)
    if hasattr(vla.transition_action_adapter, 'gate_logit'):
        vla.transition_action_adapter.gate_logit.data.fill_(args.gate_init)

    if accelerator.is_main_process:
        print(f"  P1-New weights loaded (remapped {len(p1_aux_keys)} shared decoder keys).")
        p1_relevant_missing = [k for k in missing if not k.startswith("action_model.")
                               and "dino_encoder" not in k and "_distill" not in k
                               and "posterior_encoder" not in k]
        if p1_relevant_missing:
            print(f"    ⚠️  P1-relevant missing: {len(p1_relevant_missing)}")
            for k in sorted(p1_relevant_missing)[:10]:
                print(f"      - {k}")
        gamma = vla.transition_loss_weights.get("slot_residual_gamma", "N/A")
        print(f"  gamma={gamma}  gate_init={args.gate_init}  sigmoid(gate_init)={torch.sigmoid(torch.tensor(args.gate_init)).item():.4f}")

    # ── P1 parity check: run one batch and print aux metrics ──
    if accelerator.is_main_process:
        data_iter = iter(loader)
        pb = next(data_iter)
        vla.eval()
        with torch.no_grad():
            parity = vla.forward(pb)
        print(f"  P1 parity (after load, before P2 training):")
        for k in ["current_mask_loss", "future_mask_loss", "goal_mask_loss",
                  "relation_loss", "dino_future_loss", "dino_future_cos"]:
            v = parity.get(k, torch.tensor(float("nan")))
            print(f"    {k}: {v.item():.4f}")

    # ── Freeze / Unfreeze ────────────────────────────────────
    # Qwen-VL always frozen
    for p in vla.qwen_vl_interface.parameters():
        p.requires_grad_(False)

    # Step 6A: freeze P1 modules + force eval mode (BatchNorm stays in eval)
    p1_modules = [vla.vlm_projector, vla.transition_module,
                   vla.current_mask_decoder, vla.relation_head,
                   vla.dino_future_head, vla.dino_encoder]
    if vla.posterior_encoder is not None:
        p1_modules.append(vla.posterior_encoder)
    for m in p1_modules:
        if m is not None:
            for p in m.parameters():
                p.requires_grad_(False)
            m.eval()  # critical: prevent BatchNorm train-mode drift

    # Freeze unused modules
    if vla.mask_token_encoder is not None:
        for p in vla.mask_token_encoder.parameters():
            p.requires_grad_(False)
        vla.mask_token_encoder.eval()
    if vla.transition_to_action is not None:
        for p in vla.transition_to_action.parameters():
            p.requires_grad_(False)
        vla.transition_to_action.eval()

    # Step 6A trainable: adapter + spatial projectors + action model
    for p in vla.transition_action_adapter.parameters():
        p.requires_grad_(True)
    if vla.proprio_encoder is not None:
        for p in vla.proprio_encoder.parameters():
            p.requires_grad_(True)
    if vla.dino_spatial_projector is not None:
        for p in vla.dino_spatial_projector.parameters():
            p.requires_grad_(True)
    for p in vla.action_model.parameters():
        p.requires_grad_(True)

    # After model.train(), force P1 frozen modules back to eval
    # (BatchNorm in train mode changes output even with frozen weights)

    trainable = sum(p.numel() for p in vla.parameters() if p.requires_grad)
    if accelerator.is_main_process:
        print(f"  Trainable: {trainable/1e6:.1f}M")
        print(f"  VLM: frozen | mask_token_encoder: frozen (unused)")
        print(f"  P1 modules: {'frozen' if args.freeze_transition else 'trainable'}")
        print(f"  Adapter: trainable | Action: trainable")

    # ── P2-side DINO parity check ────────────────────────────
    if accelerator.is_main_process:
        print(f"\n{'='*60}")
        print("P2 DINO Parity Check (after P1 load, before training)")
        print(f"{'='*60}")
        _p2_dino_parity(vla, loader, accelerator.device)

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
        # Force frozen P1 modules to stay eval (BatchNorm train-mode breaks aux)
        for m in p1_modules:
            if m is not None:
                m.eval()
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
            cm = out.get("current_mask_loss", torch.tensor(0)).item()
            fm = out.get("future_mask_loss", torch.tensor(0)).item()
            gm = out.get("goal_mask_loss", torch.tensor(0)).item()
            rl = out.get("relation_loss", torch.tensor(0)).item()
            vla_unwrapped = accelerator.unwrap_model(vla)
            vla_unwrapped = accelerator.unwrap_model(vla)
            gate_logit = vla_unwrapped.transition_action_adapter.gate_logit.item()
            gate_act = torch.sigmoid(vla_unwrapped.transition_action_adapter.gate_logit).item()
            dl = out.get("dino_future_loss", torch.tensor(0)).item()
            ts = out.get("transition_tokens", None)
            z_norm = ts.float().norm(dim=-1).mean().item() if ts is not None else 0
            spat_grad = 0.0
            if vla_unwrapped.transition_action_adapter.gate_logit.grad is not None:
                spat_grad = vla_unwrapped.transition_action_adapter.gate_logit.grad.item()
            print(f"  Step {step:5d}: total={loss.item():.4f} action={al:.4f} "
                  f"C={cm:.4f} F={fm:.4f} G={gm:.4f} R={rl:.4f} "
                  f"DINO={dl:.4f} gate_logit={gate_logit:.3f} gate_sig={gate_act:.3f} g_grad={spat_grad:.2e} "
                  f"|z|={z_norm:.1f} lr={scheduler.get_last_lr()[0]:.2e}")

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
                gate_val = torch.sigmoid(unwrapped.transition_action_adapter.gate_logit).item()
                print(f"  📊 Eval {step+1}: total={avg_tot:.4f} action={avg_al:.4f} gate_sig={gate_val:.4f}")
                with open(output_dir / "metrics.jsonl", "a") as f:
                    f.write(json.dumps({"step": step + 1, "val_total": float(avg_tot),
                                        "val_action": float(avg_al), "gate_sig": float(gate_val)}) + "\n")
                if avg_al < best_eval:
                    best_eval = avg_al
                    trainable_state = {k: v for k, v in unwrapped.state_dict().items()
                                       if any(p in k for p in ['transition_action_adapter', 'proprio_encoder', 'dino_spatial_projector', 'action_model'])}
                    torch.save({"model_state_dict": trainable_state},
                               str(output_dir / "best_model.pt"))
                    print(f"  🏆 Best action (eval_action={best_eval:.4f})")

    # ── Final ───────────────────────────────────────────────
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(vla)
        trainable_state = {k: v for k, v in unwrapped.state_dict().items()
                           if any(p in k for p in ['transition_action_adapter', 'proprio_encoder', 'dino_spatial_projector', 'action_model'])}
        torch.save({"model_state_dict": trainable_state},
                   str(output_dir / "final_model.pt"))
        print(f"\n{'='*60}\nP2-New Complete\n  Best val: {best_eval:.4f}\n"
              f"  Time: {(time.time()-t0)/60:.0f}min\n{'='*60}")


def _p2_dino_parity(vla, loader, device):
    """Verify P2's frozen dino_future_head matches P1 fresh-reload performance."""
    import torch.nn.functional as F
    from laravla.model.modules.spatial_transition import P1NoMaskWrapper

    vla.eval()
    # P1NoMaskWrapper shares VLA's already-loaded params (no disk reload needed)
    p1_ref = P1NoMaskWrapper(vla).to(device)
    p1_ref.eval()
    for p in p1_ref.parameters():
        p.requires_grad_(False)

    dino_key = 'image_tau_future'
    dino_cos_p1, dino_cos_p2 = [], []
    for bi, batch in enumerate(loader):
        if bi >= 10: break
        images = [s['image'] for s in batch]
        instructions = [s['lang'] for s in batch]
        cm = torch.from_numpy(np.stack([s['current_affordance_mask_agentview'] for s in batch])).unsqueeze(1).to(device).float()
        fm = torch.from_numpy(np.stack([s.get('future_tau_mask_agentview', s.get('future_affordance_mask_agentview', np.zeros((224,224),dtype=np.float32))) for s in batch])).to(device).float()
        gm = torch.from_numpy(np.stack([s['goal_affordance_mask_agentview'] for s in batch])).to(device).float()
        ri = torch.tensor([s['relation_label_id'] for s in batch], dtype=torch.long, device=device)
        with torch.no_grad():
            qo = vla.qwen_vl_interface.encode_observation(images=images, instructions=instructions, output_hidden_states=True)
            vh = qo.hidden_states[-1]
        # Tau future DINO target
        fimgs = [s.get(dino_key, s.get('image_next', None)) for s in batch]
        ft = []
        for i, fi in enumerate(fimgs):
            if fi is not None and isinstance(fi, list) and len(fi) > 0: fi = fi[0]
            if fi is not None: ft.append(torch.from_numpy(np.array(fi, dtype=np.uint8)).permute(2,0,1))
            else: ft.append(torch.from_numpy(np.array(images[i][0], dtype=np.uint8)).permute(2,0,1))
        with torch.no_grad():
            dt = vla.dino_encoder(torch.stack(ft).to(device))
        # P1 reference
        p1_out = p1_ref(vh, cm, fm, gm, ri, dino_future_target=dt)
        dino_cos_p1.append(p1_out.get('dino_future_cos', torch.tensor(0)).item())
        # P2 internal
        p2_out = vla.forward(batch)
        # Compute P2 DINO cos manually
        ts = p2_out.get('transition_tokens')
        if ts is not None:
            ftok = ts[:, 2:4, :]
            pred = vla.dino_future_head(ftok)
            cos = (F.normalize(pred.float(), dim=-1) * F.normalize(dt.float(), dim=-1)).sum(dim=-1).mean().item()
            dino_cos_p2.append(cos)
        if bi == 0:
            z_diff = (p1_out['transition_tokens'] - ts).abs().max().item() if ts is not None else -1
            print(f"  Batch 0: P1 DINO cos={dino_cos_p1[-1]:.4f}  P2 DINO cos={dino_cos_p2[-1]:.4f}  z_diff={z_diff:.2e}")

    avg_p1 = np.mean(dino_cos_p1) if dino_cos_p1 else 0
    avg_p2 = np.mean(dino_cos_p2) if dino_cos_p2 else 0
    diff = abs(avg_p1 - avg_p2)
    print(f"  P1 ref DINO cos={avg_p1:.4f}  P2 internal DINO cos={avg_p2:.4f}  |diff|={diff:.4f}")
    if diff < 0.05:
        print(f"  ✅ P2 DINO parity PASSED (P1≈P2, consistent)")
        print(f"  Note: low DINO cos on early batches is normal; P1 best reaches ~0.8 on eval")
    else:
        print(f"  ❌ P2 DINO cos differs from P1 by {diff:.4f} — check P1→P2 interface")
    vla.train()


if __name__ == "__main__":
    main()
