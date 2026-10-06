#!/usr/bin/env bash
# Libero-all :: train.py — repository root: bash scripts/run_laravla_libero.sh
set -euo pipefail
export TOKENIZERS_PARALLELISM=false


PRETRAINED_CKPT="${PRETRAINED_CKPT:-}"
RELOAD_MODULES="${RELOAD_MODULES:-qwen_vl_interface}"
NUM_GPUS="${NUM_GPUS:-8}"
MASTER_PORT="${MASTER_PORT:-29513}"
CONFIG_YAML="${CONFIG_YAML:-laravla/config/training/libero.yaml}"
RUN_ROOT="${RUN_ROOT:-results/Libero_VLA}"
RUN_ID="${RUN_ID:-libero_all_vla}"
WANDB_PROJECT="${WANDB_PROJECT:-libero_vla}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-40000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-4000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-20000000}"
MIN_SAVE_STEP="${MIN_SAVE_STEP:-16000}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-16}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
DRY_RUN="${DRY_RUN:-false}"

if [[ "${DRY_RUN}" != "true" && ( -z "${PRETRAINED_CKPT}" || ! -f "${PRETRAINED_CKPT}" ) ]]; then
  echo "Stage III 需要有效的 Stage II 最终 checkpoint: PRETRAINED_CKPT" >&2
  exit 1
fi

if [[ "${DRY_RUN}" != "true" ]]; then
  mkdir -p "${RUN_ROOT}/${RUN_ID}"
fi

args=(
  --config_yaml "${CONFIG_YAML}"
  --run_root_dir "${RUN_ROOT}"
  --run_id "${RUN_ID}"
  --wandb_project "${WANDB_PROJECT}"
  --framework.training_stage full
  --datasets.vla_data.bridge_reasoning.stage 4
  --datasets.vla_data.bridge_reasoning.include_action_tokens false
  --framework.latent_reasoning.vlm_loss_weight 0
  --trainer.max_train_steps "${MAX_TRAIN_STEPS}"
  --trainer.save_interval "${SAVE_INTERVAL}"
  --trainer.eval_interval "${EVAL_INTERVAL}"
  --trainer.min_save_step "${MIN_SAVE_STEP}"
  --datasets.vla_data.per_device_batch_size "${PER_DEVICE_BATCH_SIZE}"
  --trainer.gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
  --framework.img_next.use_teacher false
  --framework.action_model.diffusion_model_cfg.dropout 0.1
  --trainer.pretrained_checkpoint "${PRETRAINED_CKPT}"
  --trainer.reload_modules "${RELOAD_MODULES}"
)

command=(
  torchrun
  --nproc_per_node="${NUM_GPUS}"
  --master_port="${MASTER_PORT}"
  laravla/training/train.py
  "${args[@]}"
  "$@"
)
if [[ "${DRY_RUN}" == "true" ]]; then
  printf 'stage III command:'
  printf ' %q' "${command[@]}"
  printf '\n'
else
  exec "${command[@]}"
fi
