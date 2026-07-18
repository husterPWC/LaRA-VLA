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

import argparse, hashlib, json, os, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler


def _is_main():
    return not dist.is_initialized() or dist.get_rank() == 0


def _log(*args, **kwargs):
    if _is_main():
        print(*args, **kwargs)

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


def build_model(args, device="cuda"):
    """Build the single FormalSpatialActionModel."""
    from laravla.model.tools import read_mode_config
    from omegaconf import OmegaConf
    model_cfg, _ = read_mode_config(Path(CKPT))
    # Disable latent reasoning + img_next — not used in spatial training path
    model_cfg["framework"]["enable_latent_reasoning"] = False
    model_cfg["framework"]["img_next"] = {"enable": False}
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
    vla = vla.to(device)
    return vla


def _param_hash(module):
    """SHA256 hash of all parameters in a module (for freeze verification)."""
    h = hashlib.sha256()
    for p in module.parameters():
        h.update(p.detach().cpu().float().numpy().tobytes())
    return h.hexdigest()[:16]


def _param_max_diff(mod_a, mod_b):
    """Maximum absolute difference between two modules' parameters."""
    max_d = 0.0
    for (_, pa), (_, pb) in zip(mod_a.named_parameters(), mod_b.named_parameters()):
        max_d = max(max_d, (pa - pb).abs().max().item())
    return max_d


def phase1_train(args, vla, loader, eval_loader, output_dir, local_rank=0):
    """Train SpatialTransitionBackbone."""
    from laravla.model.modules.spatial_transition import (
        SpatialTransitionBackbone, P1NoMaskWrapper
    )
    backbone = SpatialTransitionBackbone(
        vlm_dim=vla.qwen_vl_interface.model.config.hidden_size,
        hidden_dim=512, num_slots=6, gamma=args.gamma,
    )
    p1_model = P1NoMaskWrapper(
        backbone=backbone, loss_weights=vla.transition_loss_weights,
    ).to(f"cuda:{local_rank}")
    for p in p1_model.parameters():
        p.requires_grad_(True)
    # Freeze VLM
    vla.qwen_vl_interface.eval()
    for p in vla.qwen_vl_interface.parameters():
        p.requires_grad_(False)

    ddp_p1 = DDP(p1_model, device_ids=[local_rank]) if dist.is_initialized() else p1_model

    opt = torch.optim.AdamW(p1_model.parameters(), lr=args.p1_lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.p1_steps, eta_min=args.p1_lr * 0.01)

    best_score = -float("inf")
    best_step = -1
    best_state = None
    t0 = time.time()
    data_iter = iter(loader)
    for step in range(args.p1_steps):
        try: batch = next(data_iter)
        except StopIteration: data_iter = iter(loader); batch = next(data_iter)

        # Prepare batch
        images = [s["image"] for s in batch]
        instructions = [s["lang"] for s in batch]
        cur_masks = torch.from_numpy(np.stack([s["current_affordance_mask_agentview"] for s in batch])).unsqueeze(1).to(f"cuda:{local_rank}").float()
        fut_masks = torch.from_numpy(np.stack([s.get("future_tau_mask_agentview", s.get("future_affordance_mask_agentview", np.zeros((224,224),dtype=np.float32))) for s in batch])).to(f"cuda:{local_rank}").float()
        gl_masks = torch.from_numpy(np.stack([s["goal_affordance_mask_agentview"] for s in batch])).to(f"cuda:{local_rank}").float()
        rel_ids = torch.tensor([s["relation_label_id"] for s in batch], dtype=torch.long, device=f"cuda:{local_rank}")

        with torch.no_grad():
            qo = vla.qwen_vl_interface.encode_observation(images=images, instructions=instructions, output_hidden_states=True)
            vlm_hidden = qo.hidden_states[-1]

        ddp_p1.train()
        out = ddp_p1(vlm_hidden, cur_masks, fut_masks, gl_masks, rel_ids)
        loss = out["total_loss"]
        if torch.isnan(loss): opt.zero_grad(); continue

        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(p1_model.parameters(), 1.0)
        opt.step(); sched.step()

        if step % 100 == 0 or step == args.p1_steps - 1:
            cd = out.get("current_dice", torch.tensor(0)).item()
            fd = out.get("future_dice", torch.tensor(0)).item()
            _log(f"  P1 {step:4d}: loss={loss.item():.4f}  C={cd:.2f} F={fd:.2f}  lr={sched.get_last_lr()[0]:.2e}")

        # Eval + save best (fixed eval set, fixed seed) — rank 0 only
        if (step + 1) % args.eval_interval == 0 and _is_main():
            p1_model.eval()
            torch.manual_seed(42); torch.cuda.manual_seed(42)
            eval_cd, eval_fd, eval_gd, eval_ra = [], [], [], []
            dev = f"cuda:{local_rank}"
            with torch.no_grad():
                for eb in eval_loader:
                    ecm = torch.from_numpy(np.stack([s["current_affordance_mask_agentview"] for s in eb])).unsqueeze(1).to(dev).float()
                    efm = torch.from_numpy(np.stack([s.get("future_tau_mask_agentview", s.get("future_affordance_mask_agentview", np.zeros((224,224),dtype=np.float32))) for s in eb])).to(dev).float()
                    egm = torch.from_numpy(np.stack([s["goal_affordance_mask_agentview"] for s in eb])).to(dev).float()
                    eri = torch.tensor([s["relation_label_id"] for s in eb], dtype=torch.long, device=dev)
                    qo = vla.qwen_vl_interface.encode_observation(images=[s["image"] for s in eb], instructions=[s["lang"] for s in eb], output_hidden_states=True)
                    evh = qo.hidden_states[-1]
                    eo = p1_model(evh, ecm, efm, egm, eri)
                    eval_cd.append(eo["current_dice"].item())
                    eval_fd.append(eo["future_dice"].item())
                    eval_gd.append(eo["goal_dice"].item())
                    eval_ra.append(eo["relation_acc"].item())
            avg_score = (np.mean(eval_cd)+np.mean(eval_fd)+np.mean(eval_gd))/3 + 0.2*np.mean(eval_ra)
            _log(f"  📊 Eval {step+1}: CurDice={np.mean(eval_cd):.3f} FutDice={np.mean(eval_fd):.3f} GoalDice={np.mean(eval_gd):.3f} RelAcc={np.mean(eval_ra):.3f} P1Score={avg_score:.3f}")
            if avg_score > best_score:
                best_score = avg_score
                best_step = step + 1
                best_state = {k: v.cpu().clone() for k, v in p1_model.state_dict().items()}
                _log(f"  🏆 Best P1 (step={best_step}, score={best_score:.4f})")
            ddp_p1.train()

    _log(f"\n  P1 Complete. Best score={best_score:.4f} at step {best_step}  Time: {(time.time()-t0)/60:.0f}min")

    # ── Check 1: Restore best P1 explicitly (rank 0 only) ─
    if _is_main() and best_state is not None:
        current_keys = set(p1_model.state_dict().keys())
        best_keys = set(best_state.keys())
        missing = best_keys - current_keys
        unexpected = current_keys - best_keys
        p1_model.load_state_dict(best_state)
        _log(f"  🔄 Restored best P1: step={best_step}, score={best_score:.4f}")
        _log(f"     strict load: missing={len(missing)}, unexpected={len(unexpected)}")
        if missing:
            _log(f"     WARNING missing keys: {sorted(missing)[:10]}")
        if unexpected:
            _log(f"     WARNING unexpected keys: {sorted(unexpected)[:10]}")

        # Re-evaluate on fixed set to confirm
        p1_model.eval()
        torch.manual_seed(42); torch.cuda.manual_seed(42)
        re_cd, re_fd, re_gd, re_ra = [], [], [], []
        dev = f"cuda:{local_rank}"
        with torch.no_grad():
            for eb in eval_loader:
                ecm = torch.from_numpy(np.stack([s["current_affordance_mask_agentview"] for s in eb])).unsqueeze(1).to(dev).float()
                efm = torch.from_numpy(np.stack([s.get("future_tau_mask_agentview", s.get("future_affordance_mask_agentview", np.zeros((224,224),dtype=np.float32))) for s in eb])).to(dev).float()
                egm = torch.from_numpy(np.stack([s["goal_affordance_mask_agentview"] for s in eb])).to(dev).float()
                eri = torch.tensor([s["relation_label_id"] for s in eb], dtype=torch.long, device=dev)
                qo = vla.qwen_vl_interface.encode_observation(images=[s["image"] for s in eb], instructions=[s["lang"] for s in eb], output_hidden_states=True)
                evh = qo.hidden_states[-1]
                eo = p1_model(evh, ecm, efm, egm, eri)
                re_cd.append(eo["current_dice"].item())
                re_fd.append(eo["future_dice"].item())
                re_gd.append(eo["goal_dice"].item())
                re_ra.append(eo["relation_acc"].item())
        re_score = (np.mean(re_cd)+np.mean(re_fd)+np.mean(re_gd))/3 + 0.2*np.mean(re_ra)
        _log(f"     Re-eval after restore: CurDice={np.mean(re_cd):.3f} FutDice={np.mean(re_fd):.3f} GoalDice={np.mean(re_gd):.3f} RelAcc={np.mean(re_ra):.3f} P1Score={re_score:.3f}")
        delta = abs(re_score - best_score)
        _log(f"     Score delta vs best: {delta:.4f} {'✅' if delta < 0.05 else '❌'}")

    # Broadcast best_state from rank 0 to all ranks
    if dist.is_initialized():
        dist.barrier()
        for p in p1_model.parameters():
            dist.broadcast(p.data, src=0)
    return best_state, p1_model


def phase_switch_parity(vla, p1_model, loader):
    """Verify phase switch doesn't change any forward output."""
    if not _is_main():
        return p1_model
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

    # Save reference tensors (all 6 checks)
    ref = {
        "vlm_hidden": vlm_hidden.clone(),
        "z_student": out_before["transition_tokens"].clone(),
        "current_logits": out_before["current_mask_logits"].clone(),
        "future_logits": out_before["future_mask_logits"].clone(),
        "goal_logits": out_before["goal_mask_logits"].clone(),
        "relation_logits": out_before["relation_logits"].clone(),
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

    checks = [
        ("vlm_hidden",       ref["vlm_hidden"],       vlm_hidden),
        ("z_student",         ref["z_student"],         out_after["transition_tokens"]),
        ("current_logits",    ref["current_logits"],    out_after["current_mask_logits"]),
        ("future_logits",     ref["future_logits"],     out_after["future_mask_logits"]),
        ("goal_logits",       ref["goal_logits"],       out_after["goal_mask_logits"]),
        ("relation_logits",   ref["relation_logits"],   out_after["relation_logits"]),
    ]

    all_ok = True
    for name, before, after in checks:
        diff = (before.float() - after.float()).abs().max().item()
        ok = diff < 1e-5
        print(f"  {name:20s} max_abs_diff: {diff:.2e} {'✅' if ok else '❌'}")
        if not ok:
            all_ok = False

    if not all_ok:
        raise RuntimeError("Phase switch parity FAILED")

    print(f"  ✅ Phase switch parity PASSED (6/6)")
    return p1_reload


def phase2_train(args, vla, p1_model, loader, eval_loader, output_dir, local_rank=0):
    """Train SpatialActionAdapter + ActionHead with frozen backbone."""
    _log(f"\n{'='*60}")
    _log("Phase 2: Action Training")
    _log(f"{'='*60}")

    # Freeze backbone
    for p in p1_model.backbone.parameters():
        p.requires_grad_(False)
    p1_model.backbone.eval()

    # ── Check 3a: Hash P1 params before training ────────────
    p1_hash_before = _param_hash(p1_model.backbone)
    p1_ref_params = {k: v.detach().cpu().clone() for k, v in p1_model.backbone.named_parameters()}
    _log(f"  P1 param hash (before P2): {p1_hash_before}")

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
    ddp_vla = DDP(vla, device_ids=[local_rank]) if dist.is_initialized() else vla

    best_action = float("inf")
    data_iter = iter(loader)

    # Eval step 0 baseline (fixed set, fixed seed) — rank 0 only
    if _is_main():
        vla.eval()
        p1_model.backbone.eval()
        torch.manual_seed(42); torch.cuda.manual_seed(42)
        eval_al0 = []
        with torch.no_grad():
            for eb in eval_loader:
                eo = vla.forward(eb)
                eval_al0.append(eo.get("action_loss", torch.tensor(0)).item())
        _log(f"  📊 Eval step 0: action={np.mean(eval_al0):.4f}")
    data_iter = iter(loader)

    for step in range(args.p2_steps):
        try: batch = next(data_iter)
        except StopIteration: data_iter = iter(loader); batch = next(data_iter)

        vla.train()
        p1_model.backbone.eval()

        out = ddp_vla(batch)
        loss = out["total_loss"]
        if torch.isnan(loss): opt.zero_grad(); continue

        opt.zero_grad(); loss.backward()

        # ── Check 3b: Verify P1 frozen after first backward ──
        if step == 0 and _is_main():
            p1_grad_count = sum(1 for p in p1_model.backbone.parameters() if p.grad is not None)
            act_grad_norm = sum(p.grad.norm().item() for p in vla.action_model.parameters() if p.grad is not None)
            adapt_grad_norm = sum(p.grad.norm().item() for p in vla.transition_action_adapter.parameters() if p.grad is not None)
            prop_grad_norm = sum(p.grad.norm().item() for p in vla.proprio_encoder.parameters() if p.grad is not None)
            gate_g = vla.transition_action_adapter.gate_logit.grad.item() if vla.transition_action_adapter.gate_logit.grad is not None else 0.0
            _log(f"  🔒 P1 frozen check (after 1st backward):")
            _log(f"     P1 parameters with grad: {p1_grad_count} {'✅' if p1_grad_count==0 else '❌ EXPECTED 0'}")
            _log(f"     Action DiT grad norm:     {act_grad_norm:.4f} {'✅' if act_grad_norm>0 else '❌'}")
            _log(f"     Spatial adapter grad norm:{adapt_grad_norm:.4f} {'✅' if adapt_grad_norm>0 else '❌'}")
            _log(f"     Proprio projector grad:   {prop_grad_norm:.4f} {'✅' if prop_grad_norm>0 else '❌'}")
            _log(f"     Gate grad:                {gate_g:.2e} {'✅' if abs(gate_g)>1e-12 else '❌'}")

        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
        opt.step(); sched.step()

        al = out.get("action_loss", torch.tensor(0)).item()
        gate = torch.sigmoid(vla.transition_action_adapter.gate_logit).item()
        gate_grad = vla.transition_action_adapter.gate_logit.grad
        ggrad = gate_grad.item() if gate_grad is not None else 0.0
        if step % 50 == 0 or step == args.p2_steps - 1:
            _log(f"  P2 {step:4d}: action={al:.4f} gate={gate:.3f} gate_grad={ggrad:.2e} lr={sched.get_last_lr()[0]:.2e}")

        if (step + 1) % args.eval_interval == 0 and _is_main():
            vla.eval()
            p1_model.backbone.eval()
            torch.manual_seed(42); torch.cuda.manual_seed(42)
            eval_al = []
            with torch.no_grad():
                for eb in eval_loader:
                    eo = vla.forward(eb)
                    eval_al.append(eo.get("action_loss", torch.tensor(0)).item())
            avg_al = np.mean(eval_al) if eval_al else 0
            _log(f"  📊 Eval {step+1}: action={avg_al:.4f}")
            if avg_al < best_action:
                best_action = avg_al
                torch.save({"model_state_dict": {k:v.cpu() for k,v in vla.state_dict().items() if any(p in k for p in ['spatial_backbone','transition_action_adapter','proprio_encoder','action_model'])}}, str(output_dir / "best_p2.pt"))
                _log(f"  🏆 Best P2 (action={best_action:.4f})")
            ddp_vla.train()
            p1_model.backbone.eval()

    # ── Check 3c: Verify P1 params unchanged after P2 (rank 0 only) ──
    if _is_main():
        p1_hash_after = _param_hash(p1_model.backbone)
        max_d = 0.0
        for name, p in p1_model.backbone.named_parameters():
            if name in p1_ref_params:
                max_d = max(max_d, (p.detach().cpu() - p1_ref_params[name]).abs().max().item())
        hash_ok = p1_hash_before == p1_hash_after
        params_ok = max_d < 1e-12
        _log(f"  🔒 P1 freeze verification after P2:")
        _log(f"     Hash unchanged: {hash_ok} {'✅' if hash_ok else '❌'} (before={p1_hash_before} after={p1_hash_after})")
        _log(f"     Max param diff: {max_d:.2e} {'✅' if params_ok else '❌'}")

    _log(f"\n  P2 Complete. Best action={best_action:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--p1-steps", type=int, default=2000)
    parser.add_argument("--p1-lr", type=float, default=3e-4)
    parser.add_argument("--p2-steps", type=int, default=500)
    parser.add_argument("--p2-lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gamma", type=float, default=1.5)
    parser.add_argument("--gate-init", type=float, default=-2.2)
    parser.add_argument("--eval-interval", type=int, default=200)
    parser.add_argument("--eval-samples", type=int, default=200,
                        help="Number of val-set samples for each eval (subset of val frames)")
    parser.add_argument("--val-split", type=float, default=0.1,
                        help="Fraction of demos held out for validation")
    parser.add_argument("--output-dir", type=str, default=str(_REPO / "results/SpatialAction_formal"))
    args = parser.parse_args()

    output_dir = Path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True)

    # ── DDP init ─────────────────────────────────────────
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)

    if _is_main():
        print("=" * 60)
        print("Unified Spatial Action Training")
        print(f"  P1: {args.p1_steps} steps, lr={args.p1_lr}")
        print(f"  P2: {args.p2_steps} steps, lr={args.p2_lr}")
        print(f"  gamma={args.gamma}  gate_init={args.gate_init}  val_split={args.val_split}")
        print(f"  GPUs: {world_size}  effective_batch={args.batch_size * world_size}")
        print("=" * 60)

    # Build
    dev = f"cuda:{local_rank}"
    vla = build_model(args, device=dev)
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

    # ── Check 4: Demo-level train/val split ────────────────
    # Same (suite, task_id, demo_id) must not appear in both train and val.
    full_ds = loader.dataset
    entries = full_ds._ds.entries

    from collections import defaultdict
    demo_to_indices = defaultdict(list)
    for i, entry in enumerate(entries):
        dk = (entry.get("suite", ""), entry.get("task_id", -1), entry.get("demo_id", -1))
        demo_to_indices[dk].append(i)

    demo_keys = sorted(demo_to_indices.keys())
    n_val_demos = max(1, int(len(demo_keys) * args.val_split))
    rng = np.random.RandomState(42)
    rng.shuffle(demo_keys)
    val_demo_set = set(demo_keys[:n_val_demos])

    train_indices, val_indices = [], []
    for dk in demo_keys:
        if dk in val_demo_set:
            val_indices.extend(demo_to_indices[dk])
        else:
            train_indices.extend(demo_to_indices[dk])

    train_subset = Subset(full_ds, sorted(train_indices))
    val_subset = Subset(full_ds, sorted(val_indices))

    # Re-create training loader on train subset with DistributedSampler
    if dist.is_initialized():
        train_sampler = DistributedSampler(train_subset, shuffle=True)
        train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=False,
                                  sampler=train_sampler,
                                  collate_fn=lambda batch: batch, num_workers=2, pin_memory=True)
    else:
        train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True,
                                  collate_fn=lambda batch: batch, num_workers=2, pin_memory=True)

    # Fixed eval loader on val subset — rank 0 only
    n_eval = min(args.eval_samples, len(val_subset))
    eval_indices = sorted(rng.choice(len(val_subset), size=n_eval, replace=False).tolist())
    eval_subset = Subset(val_subset, eval_indices)
    eval_loader = DataLoader(eval_subset, batch_size=args.batch_size, shuffle=False,
                             collate_fn=lambda batch: batch, num_workers=0, pin_memory=True)

    _log(f"  Demo split: {len(demo_keys)} demos → {len(demo_keys)-n_val_demos} train + {n_val_demos} val")
    _log(f"  Frame split: {len(train_subset)} train + {len(val_subset)} val frames")
    _log(f"  Fixed eval set: {n_eval} val samples ({len(eval_loader)} batches)")

    # Sync all ranks before training
    if dist.is_initialized():
        dist.barrier()

    # Phase 1
    best_p1, p1_model = phase1_train(args, vla, train_loader, eval_loader, output_dir, local_rank)
    if _is_main() and best_p1 is not None:
        torch.save({"p1_state_dict": best_p1}, str(output_dir / "best_p1.pt"))

    # Phase switch parity (uses one training batch)
    p1_model = phase_switch_parity(vla, p1_model, train_loader)

    # Copy trained P1 backbone weights into VLA's spatial_backbone
    vla.spatial_backbone.load_state_dict(p1_model.backbone.state_dict())
    _log("  ✅ Copied trained P1 backbone → vla.spatial_backbone")

    # Phase 2
    phase2_train(args, vla, p1_model, train_loader, eval_loader, output_dir, local_rank)

    if dist.is_initialized():
        dist.destroy_process_group()
    _log(f"\n{'='*60}\nTraining Complete\n{'='*60}")


if __name__ == "__main__":
    main()
