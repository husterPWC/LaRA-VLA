#!/bin/bash
# ============================================================================
# Stage I-A: Explicit Transition-CoT SFT — Server Launch Script
# ============================================================================
# Usage:
#   bash scripts/run_stage1a_server.sh
#
# Prerequisites:
#   1. datasets downloaded to $LARA_REPRO/datasets/
#   2. Checkpoint at $LARA_REPRO/models/LaRA-VLA-libero/
#   3. spatial NPZ data built (output/spatial_lara_libero/)
#
# NOTE: Stage I-A does NOT need EGL/MuJoCo — it only reads pre-built NPZ data.
# ============================================================================

set -e

_THIS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_REPO="$(cd "$_THIS/.." && pwd)"
_LARA_REPRO="$(cd "$_REPO/.." && pwd)"

echo "========================================"
echo "Stage I-A: Explicit Transition-CoT SFT"
echo "========================================"
echo "  LaRA-VLA: $_REPO"
echo "  lara_repro: $_LARA_REPRO"
echo ""

# Data paths
SPATIAL_ROOT="${SPATIAL_ROOT:-$_REPO/output/spatial_lara_libero}"
INDEX_PATH="${INDEX_PATH:-$_REPO/output/spatial_lara_libero_no_noops/spatial_lara_libero_index_cot_transition_all.jsonl}"
COT_ROOT="${COT_ROOT:-$_LARA_REPRO/datasets/lovejuly/libero_lerobot_all}"
ALIGN_PATH="${ALIGN_PATH:-$SPATIAL_ROOT/cot_spatial_alignment.json}"
CKPT="${CKPT:-$_LARA_REPRO/models/LaRA-VLA-libero/checkpoints/steps_25000_pytorch_model.pt}"
CONFIG="${CONFIG:-$_REPO/laravla/config/training/stage1_cot.yaml}"

echo "Data paths:"
echo "  SPATIAL_ROOT: $SPATIAL_ROOT"
echo "  INDEX_PATH:   $INDEX_PATH"
echo "  COT_ROOT:     $COT_ROOT"
echo "  CKPT:         $CKPT"
echo "  CONFIG:       $CONFIG"
echo ""

# Number of GPUs
NUM_GPUS="${NUM_GPUS:-8}"

# Override paths in config via CLI
OVERRIDES=(
    --data.spatial_root="$SPATIAL_ROOT"
    --data.index_path="$INDEX_PATH"
    --data.cot_root="$COT_ROOT"
    --data.alignment_path="$ALIGN_PATH"
    --framework.pretrained_checkpoint="$CKPT"
)

echo "Launching with $NUM_GPUS GPUs..."
echo "Overrides: ${OVERRIDES[@]}"
echo ""

# Run with accelerate
accelerate launch \
    --num_processes="$NUM_GPUS" \
    --num_machines=1 \
    --mixed_precision=bf16 \
    "$_REPO/scripts/train_stage1_cot.py" \
    --config "$CONFIG" \
    "${OVERRIDES[@]}" \
    "$@"

echo ""
echo "Done. Check results in $_REPO/results/Stage1A_CoT/"
echo ""
echo "Next: run bridge test to verify compatibility with Qwen_GR00T:"
echo "  STAGE1A_CKPT=\$_REPO/results/Stage1A_CoT/final_model/pytorch_model.pt \\"
echo "  python scripts/bridge_stage1a_to_laravla.py"
