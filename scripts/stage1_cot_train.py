#!/usr/bin/env python
"""
Stage I minimal training: CoT loss only using SpatialCoTDataset.
Verifies cot_text_transition works with LaRA-VLA VLM.
No mask encoder, no action head, no transition module.

Usage:
    # Sanity: 50 steps
    python scripts/stage1_cot_train.py --max-steps 50
    # Short train: 500 steps
    python scripts/stage1_cot_train.py --max-steps 500
"""
import argparse, os, sys, json, time
from pathlib import Path
import torch
import numpy as np
from torch.utils.data import DataLoader

_THIS = Path(__file__).resolve()
_REPO = _THIS.parents[1]
sys.path.insert(0, str(_REPO))

import warnings; warnings.filterwarnings("ignore")

from lara_vla.data.spatial_cot_dataset import SpatialCoTDataset
from laravla.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)

_LARA_REPRO = _REPO.parent
CKPT = os.environ.get('LARAVLA_CKPT',
                       str(_LARA_REPRO / 'models/LaRA-VLA-libero/checkpoints/steps_25000_pytorch_model.pt'))
SPATIAL = str(_REPO / "output" / "spatial_lara_libero")
INDEX = str(_REPO / "output" / "spatial_lara_libero_no_noops" / "spatial_lara_libero_index_cot_transition_all.jsonl")
COT = os.environ.get('LEROBOT_ROOT', str(_LARA_REPRO / 'datasets/lovejuly/libero_lerobot_all'))

VLM_PROMPT = (
    "Robot task reasoning: first output the Subtask to perform next, "
    "then output the BBox of target object, then generate the Motion Reasoning. "
    "Instruction: {instruction}. @ {cot_text}"
)


def collate_fn(batch):
    """Collate for text-only CoT training — only fields we need."""
    train_keys = ['image', 'instruction', 'cot_text_transition',
                  'relation_label', 'suite', 'image_next', 'actions']
    out = {}
    for key in train_keys:
        if key not in batch[0]:
            continue
        vals = [b.get(key) for b in batch]
        if all(v is not None for v in vals):
            if isinstance(vals[0], np.ndarray):
                out[key] = torch.from_numpy(np.stack(vals, axis=0))
            else:
                out[key] = vals
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-steps", type=int, default=50)
    args = parser.parse_args()

    print("=" * 60)
    print(f"Stage I CoT Training (minimal, max_steps={args.max_steps})")
    print("=" * 60)

    # ── Dataset ────────────────────────────────────────────────
    print("Loading dataset...")
    ds = SpatialCoTDataset(SPATIAL, INDEX, COT, SPATIAL + "/cot_spatial_alignment.json",
                            enable_dynamic_mask=True, cache_size=16)
    # Use small subset for sanity: 1000 samples
    indices = np.random.choice(len(ds), min(1000, len(ds)), replace=False)
    subset = torch.utils.data.Subset(ds, indices)
    loader = DataLoader(subset, batch_size=2, shuffle=True, collate_fn=collate_fn, num_workers=0)
    print(f"Dataset: {len(ds)} total, using {len(indices)} for sanity")

    # ── Model ──────────────────────────────────────────────────
    print(f"\nLoading model from {CKPT}...")
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    # Load base VLM + processor
    from laravla.model.tools import read_mode_config
    model_config, _ = read_mode_config(Path(CKPT))
    base_vlm_path = model_config.get("framework", {}).get("qwenvl", {}).get(
        "base_vlm", "StarVLA/Qwen3-VL-4B-Instruct-Action")

    print(f"Base VLM: {base_vlm_path}")
    processor = AutoProcessor.from_pretrained(base_vlm_path, trust_remote_code=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        base_vlm_path, trust_remote_code=True,
        torch_dtype=torch.float16, device_map="auto"
    )
    model.train()
    model.gradient_checkpointing_enable()

    # Freeze most of the model, train only last 2 LLM layers
    total_layers = len(model.model.language_model.layers)
    for i, layer in enumerate(model.model.language_model.layers):
        for p in layer.parameters():
            p.requires_grad = (i >= total_layers - 2)  # last 2 layers only
    # Also train lm_head
    for p in model.lm_head.parameters():
        p.requires_grad = True
    # Freeze vision
    for p in model.visual.parameters():
        p.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {trainable / 1e6:.1f}M (last 2 layers + lm_head)")

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-6)

    # ── Training sanity ────────────────────────────────────────
    max_steps = args.max_steps
    print(f"\n=== Training {max_steps} steps ===")
    losses = []
    t0 = time.time()
    from PIL import Image

    def to_pil(tensor):
        arr = (tensor.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        return Image.fromarray(arr)

    for step, batch in enumerate(loader):
        if step >= max_steps:
            break

        # Batch together via processor (handles padding)
        texts = []
        images_pil = []
        for i in range(len(batch['instruction'])):
            inst = batch['instruction'][i]
            cot = batch['cot_text_transition'][i]
            img_pil = to_pil(batch['image'][i])
            images_pil.append(img_pil)

            messages = [
                {"role": "user", "content": [
                    {"type": "image", "image": img_pil},
                    {"type": "text", "text": (
                        "You are controlling a robot. Given the image and instruction, "
                        "generate the reasoning chain.\n"
                        f"Instruction: {inst}"
                    )},
                ]},
                {"role": "assistant", "content": [{"type": "text", "text": cot}]},
            ]
            texts.append(processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False))

        inputs = processor(text=texts, images=images_pil, return_tensors="pt", padding=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # Labels: tokenize ONLY assistant CoT, pad/truncate to match input
        cot_only = [batch['cot_text_transition'][i] for i in range(len(batch['instruction']))]
        cot_inputs = processor(text=cot_only, return_tensors="pt", padding=True)
        labels_raw = cot_inputs["input_ids"]
        L_in = inputs["input_ids"].shape[1]
        L_cot = labels_raw.shape[1]
        if L_cot >= L_in:
            # CoT too long: truncate
            labels = labels_raw[:, -L_in:]
        else:
            # Pad: prepend -100 for user+image tokens
            labels = torch.cat([
                torch.full((labels_raw.shape[0], L_in - L_cot), -100, dtype=labels_raw.dtype),
                labels_raw,
            ], dim=1)
        labels[labels == processor.tokenizer.pad_token_id] = -100
        labels = labels.to(model.device)

        # Forward + CoT loss
        with torch.autograd.set_detect_anomaly(True):
            outputs = model(**inputs, labels=labels)
        loss = outputs.loss

        if loss is None or torch.isnan(loss):
            # Debug: check logits for NaN
            if hasattr(outputs, 'logits') and outputs.logits is not None:
                has_nan = torch.isnan(outputs.logits).any().item()
                print(f"  Step {step}: loss NaN, logits NaN={has_nan}, max_logit={outputs.logits.max().item():.2f}")
            else:
                print(f"  Step {step}: loss NaN, no logits")
            # Don't skip on NaN — let it crash with traceback to see origin
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()

        losses.append(loss.item())
        if step < 5 or step % 10 == 0:
            print(f"  Step {step:3d}: loss={loss.item():.4f}  "
                  f"rel={batch['relation_label'][0]}  "
                  f"suite={batch['suite'][0]}")

    elapsed = time.time() - t0

    # ── Summary ────────────────────────────────────────────────
    print(f"\n=== Summary ===")
    print(f"Steps: {len(losses)}, Time: {elapsed:.1f}s")
    if losses:
        print(f"Loss: {losses[0]:.4f} → {losses[-1]:.4f}")
        trending = "↓" if losses[-1] < losses[0] else "→"
        print(f"Trend: {trending}")
    print(f"✅ Stage I CoT training sanity check complete.")
    print(f"\nNext: integrate into full LaRA-VLA training loop with proper tokenizer/pipeline.")


if __name__ == "__main__":
    main()
