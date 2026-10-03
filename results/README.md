# Reproduction results

This directory contains small, reviewable JSON/CSV summaries only. Raw logs,
videos, checkpoints, and generated artifacts remain under ignored directories.

- `r1_checkpoint_preflight.json`: strict checkpoint, tokenizer, processor,
  model-shape, real-frame preprocessing, and normalization checks.
- `r1_official_checkpoint_smoke.json`: one real LIBERO rollout with the
  released checkpoint on one RTX 3090.
