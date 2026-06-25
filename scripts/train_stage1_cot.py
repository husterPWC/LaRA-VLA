#!/usr/bin/env python
"""
Stage I-A: Explicit Transition-CoT SFT
======================================
Warm-up phase: trains Qwen3-VL to explicitly generate `cot_text_transition`
(subtask + reasoning + spatial transition) conditioned on image + instruction.

This is a STANDALONE CoT SFT phase — it does NOT use Qwen_GR00T's latent-forward
path, action head, img_next loss, or mask encoder. Those will be integrated in
Stage I-B (weight bridging) → Stage II (mask-conditioned) → Stage III (action).

The trained VLM weights (last 2 LLM layers + lm_head) will be loaded back
into the full Qwen_GR00T framework in Stage I-B via scripts/bridge_stage1a_to_laravla.py.

Roadmap:
    Stage I-A (this script):  Explicit Transition-CoT SFT (standalone Qwen3-VL)
    Stage I-B (bridge):       Load Stage I-A weights → Qwen_GR00T, verify forward
    Stage II:                 Add mask encoder, future/goal mask loss, relation loss
    Stage III:                Transition tokens → action head, latent reasoning

Key features:
- Uses SpatialCoTDataset (dynamic mask filtering for libero_10)
- Loads LaRA-VLA checkpoint via baseframework.from_pretrained (handles key remapping)
- Freezes all but last 2 LLM layers + lm_head (~596M trainable params)
- HuggingFace Accelerate for multi-GPU training
- Config-driven via YAML with CLI dotlist overrides
- W&B logging + periodic checkpointing
- Saves checkpoint compatible with Qwen_GR00T weight bridging (Stage I-B)

Usage:
    # Single GPU sanity
    python scripts/train_stage1_cot.py \\
        --config laravla/config/training/stage1_cot.yaml \\
        --trainer.max_train_steps=50

    # Multi-GPU (server)
    accelerate launch --num_processes=8 scripts/train_stage1_cot.py \\
        --config laravla/config/training/stage1_cot.yaml

    # Resume from checkpoint
    python scripts/train_stage1_cot.py \\
        --config laravla/config/training/stage1_cot.yaml \\
        --trainer.resume_from_checkpoint=results/Stage1A_CoT/checkpoints/steps_5000
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from accelerate import Accelerator
from torch.utils.data import DataLoader

# ── Path setup ──────────────────────────────────────────────────
_THIS = Path(__file__).resolve()
_REPO = _THIS.parents[1]
_LARA_REPRO = _REPO.parent
sys.path.insert(0, str(_REPO))

import warnings
warnings.filterwarnings("ignore")

from lara_vla.data.spatial_cot_dataset import SpatialCoTDataset
from laravla.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)


# ==================================================================
#  Config helpers
# ==================================================================

def resolve_path(p: str, base: Path = None) -> str:
    """Resolve a path. If relative, resolve against base (default: _REPO)."""
    if p.startswith("/"):
        return p
    return str((base or _REPO) / p)


def load_config(config_path: str, cli_overrides: list = None):
    """Load YAML config and apply CLI dotlist overrides."""
    try:
        from omegaconf import OmegaConf
    except ImportError:
        print("ERROR: omegaconf required. Install: pip install omegaconf")
        raise

    cfg = OmegaConf.load(config_path)
    if cli_overrides:
        dotlist = []
        for arg in cli_overrides:
            # Normalize: --key.subkey=value → key.subkey=value
            arg = arg.lstrip("-")
            if "=" not in arg:
                continue
            dotlist.append(arg)
        cli_cfg = OmegaConf.from_dotlist(dotlist)
        cfg = OmegaConf.merge(cfg, cli_cfg)

    # Resolve relative paths to absolute
    # spatial_root, index_path, alignment_path are relative to _REPO
    for key in ["spatial_root", "index_path", "alignment_path"]:
        if key in cfg.data:
            cfg.data[key] = resolve_path(cfg.data[key], base=_REPO)
    # cot_root is relative to _LARA_REPRO (sibling of LaRA-VLA)
    if "cot_root" in cfg.data:
        cfg.data["cot_root"] = resolve_path(cfg.data["cot_root"], base=_LARA_REPRO)
    if "pretrained_checkpoint" in cfg.framework:
        cfg.framework["pretrained_checkpoint"] = resolve_path(
            cfg.framework["pretrained_checkpoint"], base=_LARA_REPRO)

    return cfg


# ==================================================================
#  Collate function
# ==================================================================

def collate_fn(batch):
    """Convert SpatialCoTDataset output to training format.

    Returns lists of PIL images, strings — processor handles batching.
    """
    out = {}
    for key in batch[0]:
        vals = [b.get(key) for b in batch]
        if all(v is not None for v in vals):
            if isinstance(vals[0], np.ndarray):
                out[key] = np.stack(vals, axis=0)
            else:
                out[key] = vals
    return out


# ==================================================================
#  Input building
# ==================================================================

def numpy_to_pil(img_np: np.ndarray):
    """Convert [3, H, W] float32 numpy (0-1) to PIL Image."""
    from PIL import Image
    arr = (img_np.transpose(1, 2, 0) * 255).astype(np.uint8)
    return Image.fromarray(arr)


VLM_USER_TEMPLATE = (
    "You are controlling a robot. Given the image and instruction, "
    "generate the reasoning chain.\n"
    "Instruction: {instruction}"
)


def build_training_inputs(processor, images_np, instructions, cot_texts):
    """Build input_ids + aligned labels for CoT training.

    Strategy:
    1. Build messages [user(image+instruction), assistant(cot_text)]
    2. Apply chat template → input_ids
    3. Tokenize only the assistant CoT text separately
    4. Align labels: prepend -100 for user+image portion, keep CoT token IDs

    Args:
        processor: HuggingFace AutoProcessor
        images_np: [B, 3, H, W] numpy array
        instructions: list of str
        cot_texts: list of str (cot_text_transition)

    Returns:
        dict with input_ids, attention_mask, pixel_values, image_grid_thw, labels
    """
    B = len(instructions)
    images_pil = [numpy_to_pil(img) for img in images_np]

    # Build messages: one per sample, user + assistant
    messages = []
    for i in range(B):
        msg = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": images_pil[i]},
                    {"type": "text", "text": VLM_USER_TEMPLATE.format(instruction=instructions[i])},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": cot_texts[i]}],
            },
        ]
        messages.append(msg)

    # Apply chat template → full input_ids
    # Use left padding so CoT text is right-aligned (consistent with Qwen interface)
    processor.tokenizer.padding_side = "left"
    inputs = processor.apply_chat_template(
        messages, tokenize=True, padding=True, return_tensors="pt",
        add_generation_prompt=False, return_dict=True,
    )

    # Tokenize ONLY the CoT text (assistant response) for labels
    cot_inputs = processor.tokenizer(cot_texts, return_tensors="pt", padding=True)
    labels_raw = cot_inputs["input_ids"]  # [B, L_cot]

    L_in = inputs["input_ids"].shape[1]
    L_cot = labels_raw.shape[1]

    if L_cot >= L_in:
        # CoT is longer than full input: truncate CoT from left
        labels = labels_raw[:, -L_in:]
    else:
        # CoT is shorter: prepend -100 for user+image+header portion
        labels = torch.cat([
            torch.full((B, L_in - L_cot), -100, dtype=labels_raw.dtype),
            labels_raw,
        ], dim=1)

    # Mask padding tokens
    labels[labels == processor.tokenizer.pad_token_id] = -100

    inputs["labels"] = labels
    return inputs


# ==================================================================
#  Checkpoint helpers
# ==================================================================

def save_checkpoint(accelerator, model, step, output_dir, is_final=False):
    """Save model checkpoint."""
    if not accelerator.is_main_process:
        return
    if is_final:
        ckpt_dir = Path(output_dir) / "final_model"
        ckpt_path = ckpt_dir / "pytorch_model.pt"
    else:
        ckpt_dir = Path(output_dir) / "checkpoints"
        ckpt_path = ckpt_dir / f"steps_{step}_pytorch_model.pt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    summary_path = Path(output_dir) / "summary.jsonl"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    state_dict = accelerator.get_state_dict(model)
    torch.save(state_dict, str(ckpt_path))

    # Append to summary
    summary_path = Path(output_dir) / "summary.jsonl"
    with open(summary_path, "a") as f:
        f.write(json.dumps({"steps": step, "checkpoint": str(ckpt_path)}) + "\n")

    logger.info(f"Checkpoint saved: {ckpt_path}")


# ==================================================================
#  Validation & Generation helpers
# ==================================================================

@torch.no_grad()
def _eval_val_loss(model, val_loader, processor, accelerator, max_batches=50):
    """Compute average val loss over max_batches."""
    model.eval()
    total_loss = 0.0
    count = 0
    for batch in val_loader:
        if count >= max_batches:
            break
        images_np = batch["image"]
        instructions = batch["instruction"]
        cot_texts = batch["cot_text_transition"]
        if isinstance(instructions, np.ndarray):
            instructions = list(instructions)
        if isinstance(cot_texts, np.ndarray):
            cot_texts = list(cot_texts)

        inputs = build_training_inputs(processor, images_np, instructions, cot_texts)
        inputs = {k: v.to(accelerator.device) for k, v in inputs.items()}

        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = model(**inputs)
            loss = outputs.loss

        if loss is not None and not torch.isnan(loss):
            total_loss += accelerator.gather(loss).mean().item()
            count += 1

    model.train()
    return total_loss / max(count, 1)


@torch.no_grad()
def _generate_cot_samples(model, val_loader, processor, accelerator, num_samples=4):
    """Generate CoT text samples from val set for qualitative inspection."""
    model.eval()
    samples = []
    for batch in val_loader:
        if len(samples) >= num_samples:
            break
        images_np = batch["image"]
        instructions = batch["instruction"]
        if isinstance(instructions, np.ndarray):
            instructions = list(instructions)

        for i in range(len(instructions)):
            if len(samples) >= num_samples:
                break
            img_pil = numpy_to_pil(images_np[i])
            # Build messages with chat template → text, then process with images
            prompt_text = VLM_USER_TEMPLATE.format(instruction=instructions[i])
            msg = [{"role": "user", "content": [
                {"type": "image", "image": img_pil},
                {"type": "text", "text": prompt_text},
            ]}]
            prompt_str = processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            model_inputs = processor(text=[prompt_str], images=[img_pil], return_tensors="pt").to(accelerator.device)

            with torch.autocast("cuda", dtype=torch.bfloat16):
                generated = model.generate(
                    **model_inputs, max_new_tokens=200, do_sample=False,
                    pad_token_id=processor.tokenizer.pad_token_id,
                )
            # Decode only the new tokens
            gen_text = processor.tokenizer.decode(
                generated[0][model_inputs["input_ids"].shape[1]:], skip_special_tokens=True
            )
            samples.append(f"Instruction: {instructions[i][:100]}\nGenerated: {gen_text[:300]}")

    model.train()
    return samples


# ==================================================================
#  Main
# ==================================================================

def main(cfg):
    # ── Accelerator setup ─────────────────────────────────────
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.trainer.gradient_accumulation_steps,
        mixed_precision="bf16" if cfg.trainer.enable_mixed_precision else "no",
    )
    rank = accelerator.process_index
    world_size = accelerator.num_processes

    if accelerator.is_main_process:
        logger.info("=" * 60)
        logger.info(f"Stage I-A: Explicit Transition-CoT SFT")
        logger.info(f"  Processes: {world_size}")
        logger.info(f"  Batch size (per device): {cfg.trainer.per_device_batch_size}")
        logger.info(f"  Gradient accumulation: {cfg.trainer.gradient_accumulation_steps}")
        logger.info(f"  Max steps: {cfg.trainer.max_train_steps}")
        logger.info("=" * 60)

    # ── Dataset ───────────────────────────────────────────────
    if accelerator.is_main_process:
        logger.info("Loading dataset...")
    full_ds = SpatialCoTDataset(
        spatial_root=cfg.data.spatial_root,
        index_path=cfg.data.index_path,
        cot_root=cfg.data.cot_root,
        alignment_path=cfg.data.alignment_path,
        future_k=cfg.data.future_k,
        cache_size=cfg.data.cache_size,
        enable_dynamic_mask=cfg.data.enable_dynamic_mask,
    )

    # Optional: limit dataset size for quick testing
    max_samples = cfg.data.get("max_samples", 0) if hasattr(cfg.data, "get") else 0
    if max_samples > 0 and max_samples < len(full_ds):
        full_len = len(full_ds)
        indices = np.random.choice(full_len, max_samples, replace=False)
        full_ds = torch.utils.data.Subset(full_ds, indices)
        if accelerator.is_main_process:
            logger.info(f"  Using {max_samples} / {full_len} samples (subset)")

    # ── Train / Val split (stratified by suite) ───────────────
    val_per_suite = cfg.trainer.get("val_samples_per_suite", 500)
    # Access entries from underlying dataset
    if hasattr(full_ds, 'dataset'):
        entries = full_ds.dataset.entries
    else:
        entries = full_ds.entries

    # Index entries by suite
    suite_to_indices = {}
    for i, e in enumerate(entries):
        if hasattr(full_ds, 'indices') and i not in set(full_ds.indices):
            continue  # Skip indices not in subset
        suite = e.get("suite", "unknown")
        suite_to_indices.setdefault(suite, []).append(i)

    # Re-index: if using Subset, map back to Subset indices
    if hasattr(full_ds, 'indices'):
        subset_indices = list(full_ds.indices)
        idx_map = {orig: pos for pos, orig in enumerate(subset_indices)}
        val_subset_indices = []
        train_subset_indices = []
        for suite, idxs in suite_to_indices.items():
            n_val = min(val_per_suite, len(idxs) // 10)  # max 10% per suite
            rng = np.random.RandomState(cfg.trainer.seed + 42)
            rng.shuffle(idxs)
            val_orig = idxs[:n_val]
            train_orig = idxs[n_val:]
            for orig in val_orig:
                if orig in idx_map:
                    val_subset_indices.append(idx_map[orig])
            for orig in train_orig:
                if orig in idx_map:
                    train_subset_indices.append(idx_map[orig])
        train_ds = torch.utils.data.Subset(full_ds, train_subset_indices)
        val_ds = torch.utils.data.Subset(full_ds, val_subset_indices)
    else:
        val_indices = []
        train_indices = []
        for suite, idxs in suite_to_indices.items():
            n_val = min(val_per_suite, len(idxs) // 10)
            rng = np.random.RandomState(cfg.trainer.seed + 42)
            rng.shuffle(idxs)
            val_indices.extend(idxs[:n_val])
            train_indices.extend(idxs[n_val:])
        train_ds = torch.utils.data.Subset(full_ds, train_indices)
        val_ds = torch.utils.data.Subset(full_ds, val_indices)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.trainer.per_device_batch_size, shuffle=True,
        collate_fn=collate_fn,
        num_workers=cfg.trainer.get("num_workers", 2),
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.trainer.per_device_batch_size, shuffle=False,
        collate_fn=collate_fn,
        num_workers=cfg.trainer.get("num_workers", 2),
        pin_memory=True,
    )
    if accelerator.is_main_process:
        logger.info(f"  Train: {len(train_ds)} samples, {len(train_loader)} batches")
        logger.info(f"  Val:   {len(val_ds)} samples, {len(val_loader)} batches")

    # ── Model ─────────────────────────────────────────────────
    if accelerator.is_main_process:
        logger.info(f"Loading model from {cfg.framework.pretrained_checkpoint}...")

    from laravla.model.tools import read_mode_config
    from laravla.model.framework.base_framework import baseframework

    # Load LaRA-VLA (handles qwen_vl_interface.model. prefix remapping)
    # from_pretrained loads to CPU; accelerator.prepare() handles device placement
    model_config, norm_stats = read_mode_config(Path(cfg.framework.pretrained_checkpoint))
    vla = baseframework.from_pretrained(cfg.framework.pretrained_checkpoint)
    model = vla.qwen_vl_interface.model  # Qwen3VLForConditionalGeneration (on CPU)

    # Get processor
    base_vlm = cfg.framework.base_vlm
    if accelerator.is_main_process:
        logger.info(f"  Base VLM: {base_vlm}")
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(base_vlm, trust_remote_code=True)

    # ── Freeze layers ─────────────────────────────────────────
    freeze_cfg = cfg.framework.freeze
    model.train()
    model.gradient_checkpointing_enable()

    n_trainable_layers = freeze_cfg.num_trainable_llm_layers
    total_layers = len(model.model.language_model.layers)
    for i, layer in enumerate(model.model.language_model.layers):
        for p in layer.parameters():
            p.requires_grad = (i >= total_layers - n_trainable_layers)

    if freeze_cfg.train_lm_head:
        for p in model.lm_head.parameters():
            p.requires_grad = True

    if not freeze_cfg.train_visual:
        for p in model.visual.parameters():
            p.requires_grad = False

    # Freeze embedding layer (should not be trained)
    for p in model.model.language_model.embed_tokens.parameters():
        p.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    if accelerator.is_main_process:
        logger.info(f"  Trainable: {trainable/1e6:.1f}M / {total/1e9:.2f}B params "
                     f"(last {n_trainable_layers} LLM layers + lm_head)")

    # ── Optimizer & Scheduler ─────────────────────────────────
    opt_cfg = cfg.trainer.optimizer
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.trainer.learning_rate,
        betas=opt_cfg.betas,
        eps=opt_cfg.eps,
        weight_decay=opt_cfg.weight_decay,
    )

    # Cosine LR scheduler with warmup
    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
    total_steps = cfg.trainer.max_train_steps
    warmup_steps = cfg.trainer.num_warmup_steps
    if warmup_steps > 0 and warmup_steps < total_steps:
        warmup_scheduler = LinearLR(
            optimizer, start_factor=0.1, total_iters=warmup_steps
        )
        cosine_scheduler = CosineAnnealingLR(
            optimizer, T_max=total_steps - warmup_steps, eta_min=cfg.trainer.min_lr
        )
        lr_scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps],
        )
    else:
        lr_scheduler = CosineAnnealingLR(
            optimizer, T_max=total_steps, eta_min=cfg.trainer.min_lr
        )

    # ── Accelerator prepare ───────────────────────────────────
    model, optimizer, train_loader, val_loader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, lr_scheduler
    )

    # ── W&B ───────────────────────────────────────────────────
    if accelerator.is_main_process and cfg.wandb_project:
        try:
            import wandb
            wandb.init(
                name=cfg.run_id,
                dir=str(Path(cfg.run_root_dir) / "wandb"),
                project=cfg.wandb_project,
                entity=cfg.wandb_entity or None,
            )
            wandb_enabled = True
        except Exception as e:
            logger.warning(f"W&B init failed: {e}")
            wandb_enabled = False
    else:
        wandb_enabled = False

    # ── Resume from checkpoint ────────────────────────────────
    start_step = 0
    resume_path = getattr(cfg.trainer, "resume_from_checkpoint", None)
    if resume_path:
        resume_path = resolve_path(resume_path)
        if accelerator.is_main_process:
            logger.info(f"Resuming from {resume_path}")
        accelerator.load_state(resume_path)
        # Extract step number from path
        try:
            start_step = int(Path(resume_path).name.split("_")[-1])
        except (ValueError, IndexError):
            start_step = 0

    # ── Training loop ─────────────────────────────────────────
    # Ensure output directories exist
    output_dir = str(_REPO / cfg.run_root_dir)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if accelerator.is_main_process:
        logger.info(f"\n=== Starting training (step {start_step} → {total_steps}) ===")

    completed_steps = start_step
    losses_history = []
    data_iter = iter(train_loader)
    t0 = time.time()

    while completed_steps < total_steps:
        # Get batch
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        # Build inputs
        images_np = batch["image"]  # [B, 3, 224, 224]
        instructions = batch["instruction"]
        cot_texts = batch["cot_text_transition"]

        # Convert to list of strings (from batch collation)
        if isinstance(instructions, np.ndarray):
            instructions = list(instructions)
        if isinstance(cot_texts, np.ndarray):
            cot_texts = list(cot_texts)

        inputs = build_training_inputs(processor, images_np, instructions, cot_texts)
        inputs = {k: v.to(accelerator.device) for k, v in inputs.items()}

        with accelerator.accumulate(model):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(**inputs)
                loss = outputs.loss

            if loss is None or torch.isnan(loss):
                if accelerator.is_main_process and completed_steps < 5:
                    logger.warning(f"  Step {completed_steps}: loss NaN/None, skipping")
                optimizer.zero_grad()
                continue

            accelerator.backward(loss)

            # Gradient clipping
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    max_norm=cfg.trainer.gradient_clipping,
                )

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        if accelerator.sync_gradients:
            completed_steps += 1
            loss_val = accelerator.gather(loss).mean().item()
            losses_history.append(loss_val)

            # ── Logging ──────────────────────────────────────────
            do_log = (completed_steps <= 5
                      or completed_steps % cfg.trainer.logging_frequency == 0)
            do_eval = (cfg.trainer.eval_interval > 0
                       and completed_steps % cfg.trainer.eval_interval == 0)

            if do_log:
                lr_now = lr_scheduler.get_last_lr()[0]
                log_msg = (f"  Step {completed_steps:5d}: train_loss={loss_val:.4f}  "
                           f"lr={lr_now:.2e}")
                metrics = {"train_loss": loss_val, "learning_rate": lr_now, "step": completed_steps}

            if do_eval:
                val_loss = _eval_val_loss(model, val_loader, processor, accelerator,
                                           max_batches=cfg.trainer.get("val_max_batches", 50))
                cot_samples = _generate_cot_samples(model, val_loader, processor, accelerator,
                                                     num_samples=4)
                if do_log:
                    log_msg += f"  val_loss={val_loss:.4f}"
                else:
                    lr_now = lr_scheduler.get_last_lr()[0]
                    log_msg = (f"  Step {completed_steps:5d}: val_loss={val_loss:.4f}  "
                               f"lr={lr_now:.2e}")
                metrics["val_loss"] = val_loss
                if cot_samples:
                    for ci, cs in enumerate(cot_samples):
                        metrics[f"cot_sample_{ci}"] = cs

            if do_log or do_eval:
                if accelerator.is_main_process:
                    suite_info = batch.get('suite', ['?'])[0] if 'suite' in batch else '?'
                    logger.info(log_msg + f"  suite={suite_info}")
                    if wandb_enabled:
                        wandb.log(metrics)

                    # Print CoT samples to console
                    if do_eval and cot_samples:
                        for ci, cs in enumerate(cot_samples[:2]):  # Print first 2
                            logger.info(f"  [CoT sample {ci}]\n{cs[:300]}...")

            # Checkpoint
            if (completed_steps % cfg.trainer.save_interval == 0
                    and completed_steps > 0):
                save_checkpoint(
                    accelerator, model, completed_steps,
                    str(_REPO / cfg.run_root_dir)
                )

    # ── Final save ────────────────────────────────────────────
    elapsed = time.time() - t0
    save_checkpoint(
        accelerator, model, completed_steps,
        str(_REPO / cfg.run_root_dir), is_final=True
    )

    if accelerator.is_main_process:
        logger.info(f"\n=== Training complete ===")
        logger.info(f"  Steps: {completed_steps}")
        logger.info(f"  Time: {elapsed/60:.1f} min ({elapsed/completed_steps:.2f}s/step)" if completed_steps > 0 else "")
        if losses_history:
            logger.info(f"  Final loss: {losses_history[-1]:.4f}")
            logger.info(f"  Loss trend: {losses_history[0]:.4f} → {losses_history[-1]:.4f}")

        if wandb_enabled:
            wandb.finish()

    accelerator.wait_for_everyone()
    logger.info("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage I CoT Training")
    parser.add_argument("--config", type=str,
                        default=str(_REPO / "laravla/config/training/stage1_cot.yaml"),
                        help="Path to YAML config")
    args, clipargs = parser.parse_known_args()

    config_path = resolve_path(args.config) if not args.config.startswith("/") else args.config
    cfg = load_config(config_path, clipargs)

    main(cfg)
