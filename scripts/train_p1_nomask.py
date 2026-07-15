#!/usr/bin/env python
"""
P1-New Formal Training: RGB-only Latent Spatial Transition (DDP-safe).
=======================================================================
Mask-supervised but mask-free-inference. No mask_token_encoder.
Transition tokens learned from VLM hidden only, supervised by:
  current mask + future mask + goal mask + relation.

Uses P1NoMaskWrapper for DDP isolation. Qwen-VL stays outside as frozen
local encoder. Only ~27M trainable params.

Usage:
    python scripts/train_p1_nomask.py --max-steps 50
    accelerate launch --num_processes=8 scripts/train_p1_nomask.py \
        --max-steps 80000 --batch-size 4 --lr 3e-4
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
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--save-interval", type=int, default=5000)
    parser.add_argument("--eval-interval", type=int, default=2000)
    parser.add_argument("--eval-batches", type=int, default=200)
    parser.add_argument("--output-dir", type=str, default=str(_REPO / "results/P1_nomask"))
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--index-path", type=str, default=str(_REPO / "output" /
        "spatial_lara_libero_no_noops" / "spatial_lara_libero_index_cot_transition_all_fixed_v4_tau.jsonl"),
        help="Path to index JSONL (default: V4 tau index)")
    parser.add_argument("--use-tau-future", action="store_true", default=True,
        help="Use fixed-τ future for masks and DINO target (Step 4)")
    parser.add_argument("--no-tau-future", action="store_true",
        help="Disable tau future (use original CoT future)")
    parser.add_argument("--gamma", type=float, default=2.0,
                        help="Slot identity residual gamma (try 1.0, 1.5, 2.0)")
    parser.add_argument("--w-distill", type=float, default=0.05,
                        help="Distill weight (0.0=teacher-only, 0.05=default, 0.10=strong)")
    args = parser.parse_args()

    if args.no_tau_future:
        args.use_tau_future = False

    # ── Accelerator ──────────────────────────────────────────
    accelerator = Accelerator(gradient_accumulation_steps=1, mixed_precision="bf16")
    if torch.cuda.is_available():
        torch.cuda.set_device(accelerator.local_process_index)

    if accelerator.is_main_process:
        print("=" * 60)
        print("P1-New: RGB-only Latent Spatial Transition (no-mask)")
        print(f"  Processes: {accelerator.num_processes}")
        print(f"  Global batch: {args.batch_size * accelerator.num_processes}")
        print(f"  Max steps: {args.max_steps}  LR: {args.lr}")
        print(f"  Index: {args.index_path}")
        print(f"  Tau future: {args.use_tau_future}")
        print(f"  Output: {args.output_dir}")
        print("=" * 60)

    output_dir = Path(args.output_dir)
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)

    # ── Dataloader ──────────────────────────────────────────
    from laravla.dataloader import build_dataloader
    loader = build_dataloader(OmegaConf.create({"datasets": {"vla_data": {
        "dataset_py": "spatial_cot_libero", "spatial_root": SPATIAL,
        "index_path": args.index_path, "cot_root": COT,
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
        "mask_res": 56, "num_relation_labels": 7, "transition_dim": 512,
        "loss_weights": {"future_mask": 0.05, "goal_mask": 0.10, "relation": 0.05,
                         "current_mask": 0.05, "dino_future": 0.05,
                         "slot_residual_gamma": args.gamma,
                         "distill_weight": args.w_distill,
                         "distill_warmup_steps": 100,
                         "teacher_loss_weight": 0.5},
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

    from laravla.model.modules.spatial_transition import P1NoMaskWrapper
    p1_model = P1NoMaskWrapper(vla).to(accelerator.device)
    for p in p1_model.parameters():
        p.requires_grad_(True)

    if accelerator.is_main_process:
        n = sum(p.numel() for p in p1_model.parameters())
        print(f"  P1NoMaskWrapper trainable: {n/1e6:.1f}M")
        print(f"  Modules: vlm_projector, transition_module, "
              f"current/future/goal_mask_decoder, relation_head, dino_future_head")
        print(f"  VLM: frozen (no_grad encode), DINO: frozen, mask_token_encoder: NOT USED")
        print(f"  DDP wraps: P1NoMaskWrapper only")

    # ── Optimizer ───────────────────────────────────────────
    optimizer = torch.optim.AdamW(p1_model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.max_steps, eta_min=args.lr * 0.01)

    # ── DDP only wraps p1_model ─────────────────────────────
    p1_model, optimizer, loader, scheduler = accelerator.prepare(
        p1_model, optimizer, loader, scheduler)

    # ── Training ────────────────────────────────────────────
    data_iter = iter(loader)
    best_eval = float("-inf")  # student_score: higher is better
    t0 = time.time()

    for step in range(args.max_steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        # Extract supervision masks (cur/future/goal are ALL supervision, no input)
        cur_masks = torch.from_numpy(
            np.stack([s["current_affordance_mask_agentview"] for s in batch])
        ).unsqueeze(1).to(accelerator.device).float()

        # Future mask: use tau future if enabled, else original CoT future
        if args.use_tau_future:
            future_key = "future_tau_mask_agentview"
            dino_image_key = "image_tau_future"
        else:
            future_key = "future_affordance_mask_agentview"
            dino_image_key = "image_next"

        future_masks = torch.from_numpy(
            np.stack([s.get(future_key, np.zeros((224,224), dtype=np.float32)) for s in batch])
        ).to(accelerator.device).float()
        goal_masks = torch.from_numpy(
            np.stack([s.get("goal_affordance_mask_agentview", np.zeros((224,224), dtype=np.float32)) for s in batch])
        ).to(accelerator.device).float()
        rel_ids = torch.tensor([s.get("relation_label_id", -1) for s in batch],
                               dtype=torch.long, device=accelerator.device)

        # tau_future_valid: only used when use_tau_future=True
        tau_valid = None
        if args.use_tau_future:
            tau_valid = torch.tensor(
                [s.get("tau_future_valid", True) for s in batch],
                dtype=torch.bool, device=accelerator.device
            )

        # Frozen VLM encode (RGB + instruction only, NO mask input)
        with torch.no_grad():
            qwen_out = vla.qwen_vl_interface.encode_observation(
                images=[s["image"] for s in batch],
                instructions=[s["lang"] for s in batch],
                output_hidden_states=True,
            )
            vlm_hidden = qwen_out.hidden_states[-1]

        # DINO features: current RGB + tau future RGB (teacher uses both)
        dino_future_target = None
        dino_cur = None
        if vla.dino_encoder is not None:
            # Current DINO (for teacher posterior)
            cur_tensors = []
            for s in batch:
                arr = np.array(s["image"][0], dtype=np.uint8)
                cur_tensors.append(torch.from_numpy(arr).permute(2, 0, 1))
            cur_rgb = torch.stack(cur_tensors).to(accelerator.device)
            with torch.no_grad():
                dino_cur = vla.dino_encoder(cur_rgb)

            # Future DINO (for DINO loss target + teacher)
            future_imgs = [s.get(dino_image_key, None) for s in batch]
            if any(fi is not None for fi in future_imgs):
                future_tensors = []
                for i, fi in enumerate(future_imgs):
                    if fi is not None and isinstance(fi, list) and len(fi) > 0:
                        fi = fi[0]
                    if fi is not None:
                        arr = np.array(fi, dtype=np.uint8)
                        future_tensors.append(torch.from_numpy(arr).permute(2, 0, 1))
                    else:
                        arr = np.array(batch[i]["image"][0], dtype=np.uint8)
                        future_tensors.append(torch.from_numpy(arr).permute(2, 0, 1))
                future_rgb = torch.stack(future_tensors).to(accelerator.device)
                with torch.no_grad():
                    dino_future_target = vla.dino_encoder(future_rgb)

        # P1 forward: student + teacher + distill
        out = p1_model(vlm_hidden, cur_masks, future_masks, goal_masks, rel_ids,
                       dino_future_target=dino_future_target,
                       tau_future_valid=tau_valid,
                       dino_cur=dino_cur)
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
            cm = out.get("current_mask_loss", torch.tensor(0)).item()
            fm = out.get("future_mask_loss", torch.tensor(0)).item()
            gm = out.get("goal_mask_loss", torch.tensor(0)).item()
            rl = out.get("relation_loss", torch.tensor(0)).item()
            dl = out.get("dino_future_loss", torch.tensor(0)).item() if dino_future_target is not None else 0
            dc = out.get("dino_future_cos", torch.tensor(0)).item() if dino_future_target is not None else 0
            cd = out.get("current_dice", torch.tensor(0)).item()
            fd = out.get("future_dice", torch.tensor(0)).item()
            gd = out.get("goal_dice", torch.tensor(0)).item()
            ra = out.get("relation_acc", torch.tensor(0)).item()
            lv = out.get("latent_var", torch.tensor(0)).item()
            lpc = out.get("latent_pair_cos", torch.tensor(0)).item()  # inter-type
            ipc = out.get("intra_pair_cos", torch.tensor(0)).item()   # intra-type
            lnm = out.get("latent_norm_mean", torch.tensor(0)).item()
            lns = out.get("latent_norm_std", torch.tensor(0)).item()
            tv = out.get("tau_valid_ratio", torch.tensor(1.0)).item()
            tdl = out.get("distill_loss", torch.tensor(0)).item()
            tcos = out.get("teacher_student_cos", torch.tensor(0)).item()
            tC = out.get("teacher_C_dice", torch.tensor(0)).item()
            tF = out.get("teacher_F_dice", torch.tensor(0)).item()
            tG = out.get("teacher_G_dice", torch.tensor(0)).item()
            tR = out.get("teacher_RelAcc", torch.tensor(0)).item()
            tDC = out.get("teacher_dino_cos", torch.tensor(0)).item()
            dw = out.get("distill_weight", torch.tensor(0)).item()
            print(f"  Step {step:5d}: total={loss.item():.4f} (S) "
                  f"C={cm:.4f}(D{cd:.2f}) F={fm:.4f}(D{fd:.2f}) G={gm:.4f}(D{gd:.2f}) "
                  f"R={rl:.4f}(A{ra:.2f}) DINO={dl:.4f}(cos{dc:.2f})  τv={tv:.2f}  "
                  f"inter={lpc:.3f}  |  (T) C={tC:.2f} F={tF:.2f} G={tG:.2f} R={tR:.2f} Dc={tDC:.2f}  "
                  f"distill={tdl:.4f}(cos{tcos:.3f} dw{dw:.3f})  "
                  f"lr={scheduler.get_last_lr()[0]:.2e}")

        # Save
        if accelerator.is_main_process and (step + 1) % args.save_interval == 0:
            unwrapped = accelerator.unwrap_model(p1_model)
            ckpt_path = output_dir / f"checkpoint_step{step+1}.pt"
            torch.save({"step": step + 1, "p1_state_dict": unwrapped.state_dict()}, str(ckpt_path))
            print(f"  ✅ Checkpoint: {ckpt_path}")

        # Eval
        if (step + 1) % args.eval_interval == 0:
            p1_model.eval()
            eval_tot, eval_cd, eval_fd, eval_gd, eval_ra = [], [], [], [], []
            eval_dl, eval_dc = [], []
            eval_lpc = []
            eval_tcd, eval_tfd, eval_tgd, eval_tra, eval_tdc = [], [], [], [], []  # teacher
            eval_iter = iter(loader)
            with torch.no_grad():
                for _ in range(args.eval_batches):
                    try:
                        eb = next(eval_iter)
                    except StopIteration:
                        break
                    # All masks are supervision only
                    cm = torch.from_numpy(np.stack(
                        [s["current_affordance_mask_agentview"] for s in eb]
                    )).unsqueeze(1).to(accelerator.device).float()
                    # Future mask: tau or original
                    eval_future_key = future_key  # from outer scope
                    fm = torch.from_numpy(np.stack(
                        [s.get(eval_future_key, np.zeros((224,224), dtype=np.float32)) for s in eb]
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

                    # tau_future_valid for eval
                    eval_tau_valid = None
                    if args.use_tau_future:
                        eval_tau_valid = torch.tensor(
                            [s.get("tau_future_valid", True) for s in eb],
                            dtype=torch.bool, device=accelerator.device)

                    # DINO features for eval
                    dino_eval_target = None
                    dino_eval_cur = None
                    if vla.dino_encoder is not None:
                        # Current DINO
                        ceval_tensors = []
                        for s in eb:
                            arr = np.array(s["image"][0], dtype=np.uint8)
                            ceval_tensors.append(torch.from_numpy(arr).permute(2, 0, 1))
                        with torch.no_grad():
                            dino_eval_cur = vla.dino_encoder(
                                torch.stack(ceval_tensors).to(accelerator.device))

                        # Future DINO
                        feval_imgs = [s.get(dino_image_key, None) for s in eb]
                        if any(fi is not None for fi in feval_imgs):
                            feval_tensors = []
                            for fi in feval_imgs:
                                if fi is not None and isinstance(fi, list) and len(fi) > 0:
                                    fi = fi[0]
                                if fi is not None:
                                    arr = np.array(fi, dtype=np.uint8)
                                    feval_tensors.append(torch.from_numpy(arr).permute(2, 0, 1))
                                else:
                                    s2 = eb[len(feval_tensors)]
                                    arr = np.array(s2["image"][0], dtype=np.uint8)
                                    feval_tensors.append(torch.from_numpy(arr).permute(2, 0, 1))
                            future_rgb_eval = torch.stack(feval_tensors).to(accelerator.device)
                            with torch.no_grad():
                                dino_eval_target = vla.dino_encoder(future_rgb_eval)

                    eo = p1_model(vh, cm, fm, gm, ri,
                                  dino_future_target=dino_eval_target,
                                  tau_future_valid=eval_tau_valid,
                                  dino_cur=dino_eval_cur)
                    eval_tot.append(eo["total_loss"].item())
                    eval_cd.append(eo["current_dice"].item())
                    eval_fd.append(eo["future_dice"].item())
                    eval_gd.append(eo["goal_dice"].item())
                    eval_ra.append(eo["relation_acc"].item())
                    if dino_eval_target is not None:
                        eval_dl.append(eo.get("dino_future_loss", torch.tensor(0)).item())
                        eval_dc.append(eo.get("dino_future_cos", torch.tensor(0)).item())
                    eval_lpc.append(eo.get("latent_pair_cos", torch.tensor(0)).item())
                    # Teacher eval
                    eval_tcd.append(eo.get("teacher_C_dice", torch.tensor(0)).item())
                    eval_tfd.append(eo.get("teacher_F_dice", torch.tensor(0)).item())
                    eval_tgd.append(eo.get("teacher_G_dice", torch.tensor(0)).item())
                    eval_tra.append(eo.get("teacher_RelAcc", torch.tensor(0)).item())
                    eval_tdc.append(eo.get("teacher_dino_cos", torch.tensor(0)).item())

            if accelerator.is_main_process:
                avg_t = np.mean(eval_tot) if eval_tot else float("nan")
                avg_cd = np.mean(eval_cd) if eval_cd else 0
                avg_fd = np.mean(eval_fd) if eval_fd else 0
                avg_gd = np.mean(eval_gd) if eval_gd else 0
                avg_ra = np.mean(eval_ra) if eval_ra else 0
                avg_dl = np.mean(eval_dl) if eval_dl else 0
                avg_dc = np.mean(eval_dc) if eval_dc else 0
                avg_lpc = np.mean(eval_lpc) if eval_lpc else 0
                avg_tcd = np.mean(eval_tcd) if eval_tcd else 0
                avg_tfd = np.mean(eval_tfd) if eval_tfd else 0
                avg_tgd = np.mean(eval_tgd) if eval_tgd else 0
                avg_tra = np.mean(eval_tra) if eval_tra else 0
                avg_tdc = np.mean(eval_tdc) if eval_tdc else 0

                student_score = (avg_cd + avg_fd + avg_gd) / 3 + 0.2 * avg_ra + 0.2 * avg_dc

                print(f"  📊 Eval {step+1}: Student "
                      f"C={avg_cd:.3f} F={avg_fd:.3f} G={avg_gd:.3f} "
                      f"Rel={avg_ra:.3f} Dc={avg_dc:.2f} inter={avg_lpc:.3f} "
                      f"score={student_score:.3f}  |  Teacher "
                      f"C={avg_tcd:.3f} F={avg_tfd:.3f} G={avg_tgd:.3f} "
                      f"Rel={avg_tra:.3f} Dc={avg_tdc:.2f}")
                metrics = {"step": step + 1, "val_loss": float(avg_t),
                          "student_score": float(student_score),
                          "C_Dice": float(avg_cd), "F_Dice": float(avg_fd),
                          "G_Dice": float(avg_gd), "RelAcc": float(avg_ra),
                          "DINO_loss": float(avg_dl), "DINO_cos": float(avg_dc),
                          "pair_cos": float(avg_lpc),
                          "teacher_C": float(avg_tcd), "teacher_F": float(avg_tfd),
                          "teacher_G": float(avg_tgd), "teacher_Rel": float(avg_tra),
                          "teacher_Dc": float(avg_tdc)}
                with open(output_dir / "metrics.jsonl", "a") as f:
                    f.write(json.dumps(metrics) + "\n")
                # Best by student_score (not total loss which includes teacher+distill)
                if student_score > best_eval:
                    best_eval = student_score
                    unwrapped = accelerator.unwrap_model(p1_model)
                    torch.save({"p1_state_dict": unwrapped.state_dict()}, str(output_dir / "best_model.pt"))
                    torch.save({"p1_state_dict": unwrapped.state_dict()},
                               str(output_dir / "best_student.pt"))
                    print(f"  🏆 Best student (score={student_score:.4f})")

    # ── Final ───────────────────────────────────────────────
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(p1_model)
        torch.save({"p1_state_dict": unwrapped.state_dict()}, str(output_dir / "final_model.pt"))
        print(f"\n{'='*60}\nP1-New Complete\n  Best val: {best_eval:.4f}\n"
              f"  Time: {(time.time()-t0)/60:.0f}min\n{'='*60}")


if __name__ == "__main__":
    main()
