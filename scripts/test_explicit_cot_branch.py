#!/usr/bin/env python
"""
Test explicit_transition_cot training branch in Qwen_GR00T.
=============================================================
Verifies:
  1. Labels decode to only assistant CoT text (no user/prompt/pad tokens)
  2. Qwen_GR00T.forward() with training_stage="explicit_transition_cot" works
  3. vlm_loss is computed and valid
  4. 10-step training loop works (loss decreases or stays stable)
  5. Original action_only branch is unaffected

Usage:
    python scripts/test_explicit_cot_branch.py
"""

import sys, os, numpy as np
from pathlib import Path
import torch

_THIS = Path(__file__).resolve()
_REPO = _THIS.parents[1]
sys.path.insert(0, str(_REPO))

import warnings; warnings.filterwarnings("ignore")

CKPT = os.environ.get("LARAVLA_CKPT",
    str(_REPO.parent / "models/LaRA-VLA-libero/checkpoints/steps_25000_pytorch_model.pt"))


def main():
    print("=" * 60)
    print("Test: explicit_transition_cot branch")
    print("=" * 60)

    # ── Load model ────────────────────────────────────────────
    print("\n[1] Loading Qwen_GR00T...")
    from laravla.model.framework.base_framework import baseframework
    vla = baseframework.from_pretrained(CKPT)
    vla = vla.to("cuda")
    vla.eval()
    processor = vla.qwen_vl_interface.processor
    print(f"  training_stage: {vla.training_stage}")
    print(f"  action_model present: {hasattr(vla, 'action_model')}")

    # ── Create test batch ─────────────────────────────────────
    print("\n[2] Creating test batch...")
    from PIL import Image
    from lara_vla.data.spatial_cot_dataset import SpatialCoTDataset

    SPATIAL = str(_REPO / "output" / "spatial_lara_libero")
    INDEX = str(_REPO / "output" / "spatial_lara_libero_no_noops" /
                "spatial_lara_libero_index_cot_transition_all_fixed_v3.jsonl")
    COT = os.environ.get("LEROBOT_ROOT",
                          str(_REPO.parent / "datasets" / "lovejuly" / "libero_lerobot_all"))

    ds = SpatialCoTDataset(SPATIAL, INDEX, COT, SPATIAL + "/cot_spatial_alignment.json",
                            enable_dynamic_mask=True, cache_size=4)
    rng = np.random.RandomState(0)
    indices = rng.choice(len(ds), 3, replace=False)

    examples = []
    for idx in indices:
        s = ds[idx]
        img_np = s["image"]
        img_pil = Image.fromarray((img_np.transpose(1,2,0)*255).astype(np.uint8))
        examples.append({
            "image": [img_pil],
            "lang": s["instruction"] if isinstance(s["instruction"], str) else str(s["instruction"]),
            "action": s["actions"],
            "state": s["robot_state"],
            "cot_text_transition": s["cot_text_transition"],
        })
    print(f"  Batch size: {len(examples)}")
    for i, ex in enumerate(examples):
        print(f"  [{i}] instruction: {ex['lang'][:60]}...")
        print(f"      cot_text:     {ex['cot_text_transition'][:80]}...")

    # ── Test 1: Label correctness ─────────────────────────────
    print("\n[3] Testing label correctness...")
    # Temporarily switch to explicit_transition_cot
    vla.training_stage = "explicit_transition_cot"

    # Build inputs manually to inspect labels
    batch_images = [ex["image"] for ex in examples]
    instructions = [ex["lang"] for ex in examples]
    cot_texts = [ex["cot_text_transition"] for ex in examples]

    qwen_inputs = vla.qwen_vl_interface.build_qwenvl_inputs(
        images=batch_images,
        instructions=instructions,
        solutions=cot_texts,
        cot_mode=True,
    )

    for i in range(len(examples)):
        input_ids = qwen_inputs["input_ids"][i]
        labels = qwen_inputs["labels"][i]
        label_mask = labels != -100
        label_positions = label_mask.nonzero(as_tuple=True)[0]
        if len(label_positions) == 0:
            print(f"  ❌ Sample {i}: NO LABELS!")
            continue
        decoded = processor.tokenizer.decode(labels[label_mask])
        match = (input_ids[label_mask] == labels[label_mask]).float().mean().item()

        # Check for contamination
        has_instruction = "Instruction:" in decoded or "instruction:" in decoded
        has_user = "user" in decoded.lower()[:20]
        has_subtask = "Subtask:" in decoded
        has_spatial = "Spatial transition:" in decoded or "Spatial" in decoded

        print(f"  Sample {i}: {len(label_positions)} labels, match={match:.0%}")
        print(f"    has_Subtask={has_subtask}, has_Spatial={has_spatial}")
        print(f"    contamination: instruction={has_instruction}, user={has_user}")
        print(f"    Decoded: {decoded[:150]}...")

        if not has_subtask:
            print(f"    ❌ Missing 'Subtask:' in labels!")
        if has_instruction:
            print(f"    ❌ Labels contain instruction text!")
        if match < 0.99:
            print(f"    ❌ Token match < 99%!")

    # ── Test 2: Forward pass ──────────────────────────────────
    print("\n[4] Testing Qwen_GR00T.forward()...")
    vla.training_stage = "explicit_transition_cot"

    with torch.no_grad():
        output = vla.forward(examples)

    print(f"  Output keys: {list(output.keys())}")
    if "vlm_loss" in output:
        print(f"  vlm_loss: {output['vlm_loss'].item():.4f}")
    if "total_loss" in output:
        print(f"  total_loss: {output['total_loss'].item():.4f}")

    assert "vlm_loss" in output, "Missing vlm_loss!"
    assert not torch.isnan(output["vlm_loss"]), "vlm_loss is NaN!"
    print("  ✅ Forward pass successful")

    # ── Test 3: 10-step training ──────────────────────────────
    print("\n[5] 10-step training sanity...")
    model = vla.qwen_vl_interface.model
    model.train()
    model.gradient_checkpointing_enable()

    # Freeze most layers, train last 2
    total_layers = len(model.model.language_model.layers)
    for i, layer in enumerate(model.model.language_model.layers):
        for p in layer.parameters():
            p.requires_grad = (i >= total_layers - 2)
    for p in model.lm_head.parameters(): p.requires_grad = True
    for p in model.visual.parameters(): p.requires_grad = False
    for p in model.model.language_model.embed_tokens.parameters(): p.requires_grad = False

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-6)

    # Use 2 fixed samples for overfit test
    test_exs = [examples[0], examples[1]]
    losses = []
    for step in range(10):
        optimizer.zero_grad()
        output = vla.forward(test_exs)
        loss = output["total_loss"]
        if torch.isnan(loss):
            print(f"  Step {step}: NaN! Aborting.")
            break
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        optimizer.step()
        losses.append(loss.item())
        if step < 3 or step % 3 == 0:
            print(f"  Step {step}: loss={loss.item():.4f}")

    print(f"  Loss trend: {losses[0]:.4f} → {losses[-1]:.4f}")

    # ── Test 4: action_only still works ───────────────────────
    print("\n[6] Testing action_only still works...")
    vla.training_stage = "action_only"
    vla.eval()
    with torch.no_grad():
        with torch.autocast("cuda", dtype=torch.float32):
            try:
                output = vla.forward(examples)
                print(f"  action_loss: {output.get('action_loss', 'N/A')}")
                print("  ✅ action_only branch still works")
            except Exception as e:
                print(f"  ❌ action_only failed: {e}")

    # ── Test 5: predict_action still works ────────────────────
    print("\n[7] Testing predict_action still works...")
    try:
        pred = vla.predict_action(
            batch_images=[[ex["image"][0]] for ex in examples],
            instructions=[ex["lang"] for ex in examples],
            use_ddim=True, num_ddim_steps=5,
        )
        print(f"  normalized_actions shape: {pred['normalized_actions'].shape}")
        print("  ✅ predict_action still works")
    except Exception as e:
        print(f"  ❌ predict_action failed: {e}")

    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
