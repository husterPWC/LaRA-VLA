#!/usr/bin/env bash
# Libero-all :: 四阶段课程训练（均为 reasoning_only）
# - Stage 1：无 pretrained_checkpoint
# - Stage 2–4：加载上一阶段 checkpoints/steps_<CKPT_STEP[上一阶段]>_pytorch_model.pt
# train：training_stage!=full 时不覆盖 bridge_reasoning.stage，每阶段由下方 BRIDGE_STAGE 指定。
# 仓库根目录: bash scripts/run_libero_multistage.sh
set -euo pipefail
export TOKENIZERS_PARALLELISM=false

# ===================== 按需改这里 =====================
CONFIG_YAML="${CONFIG_YAML:-laravla/config/training/libero.yaml}"
RUN_ROOT="${RUN_ROOT:-results/LiberoVLM}"
RUN_ID_PREFIX="${RUN_ID_PREFIX:-libero_vlm}"
NUM_GPUS="${NUM_GPUS:-8}"
MASTER_PORT="${MASTER_PORT:-29513}"
WANDB_PROJECT="${WANDB_PROJECT:-libero_vlm}"
WANDB_ENTITY="${WANDB_ENTITY:-}"

# 非空则只加载部分模块；空 = 整模加载
RELOAD_MODULES="${RELOAD_MODULES:-}"
IMG_NEXT_USE_TEACHER="${IMG_NEXT_USE_TEACHER:-true}"

STEPS_CACHE_PATH="${STEPS_CACHE_PATH:-${RUN_ROOT}/steps_cache/libero_vlm}"
WRITE_STEPS_CACHE="${WRITE_STEPS_CACHE:-true}"
DRY_RUN="${DRY_RUN:-false}"

declare -A BRIDGE_STAGE=( [1]=1 [2]=2 [3]=3 [4]=4 )

declare -A VLM_LOSS_WEIGHT=( [1]=1.0 [2]=1.0 [3]=1.0 [4]=1.0 )
declare -A IMG_NEXT_LOSS_WEIGHT=( [1]=0.1 [2]=0.2 [3]=0.2 [4]=0.2 )

declare -A PER_DEVICE_BATCH=(
  [1]="${STAGE1_PER_DEVICE_BATCH:-12}"
  [2]="${STAGE2_PER_DEVICE_BATCH:-16}"
  [3]="${STAGE2_PER_DEVICE_BATCH:-16}"
  [4]="${STAGE2_PER_DEVICE_BATCH:-16}"
)
declare -A GRADIENT_ACCUMULATION=(
  [1]="${STAGE1_GRADIENT_ACCUMULATION:-1}"
  [2]="${STAGE2_GRADIENT_ACCUMULATION:-1}"
  [3]="${STAGE2_GRADIENT_ACCUMULATION:-1}"
  [4]="${STAGE2_GRADIENT_ACCUMULATION:-1}"
)
declare -A MAX_STEPS=( [1]=5000 [2]=2000 [3]=2000 [4]=2000 )

# 须满足 MAX_STEPS[s] % SAVE_INTERVAL[s] == 0（与 train 存盘条件一致）
declare -A SAVE_INTERVAL=( [1]=5000 [2]=2000 [3]=2000 [4]=2000 )

# 第 s 阶段结束时文件名 steps_<CKPT_STEP[s]>_pytorch_model.pt 中的步数
declare -A CKPT_STEP=( [1]=5000 [2]=2000 [3]=2000 [4]=2000 )

START_STAGE="${START_STAGE:-1}"
END_STAGE="${END_STAGE:-4}"
# ====================================================

if (( START_STAGE < 1 || START_STAGE > 4 || END_STAGE < 1 || END_STAGE > 4 || START_STAGE > END_STAGE )); then
  echo "阶段范围无效: START_STAGE=${START_STAGE}, END_STAGE=${END_STAGE}" >&2
  exit 2
fi

if [[ "${DRY_RUN}" != "true" ]]; then
  mkdir -p "${STEPS_CACHE_PATH}"
fi

run_one_stage() {
  local stage="$1"
  local load_ckpt="$2"
  shift 2

  local run_id="${RUN_ID_PREFIX}_stage_${stage}"
  local out="${RUN_ROOT}/${run_id}"

  if [[ "${DRY_RUN}" != "true" ]]; then
    mkdir -p "${out}"
    cp "$0" "${out}/run_command.sh"
  fi

  local args=(
    --config_yaml "${CONFIG_YAML}"
    --run_root_dir "${RUN_ROOT}"
    --run_id "${run_id}"
    --wandb_project "${WANDB_PROJECT}"
    --framework.training_stage reasoning_only
    --datasets.vla_data.bridge_reasoning.stage "${BRIDGE_STAGE[$stage]}"
    --trainer.max_train_steps "${MAX_STEPS[$stage]}"
    --trainer.save_interval "${SAVE_INTERVAL[$stage]}"
    --datasets.vla_data.per_device_batch_size "${PER_DEVICE_BATCH[$stage]}"
    --trainer.gradient_accumulation_steps "${GRADIENT_ACCUMULATION[$stage]}"
    --framework.latent_reasoning.vlm_loss_weight "${VLM_LOSS_WEIGHT[$stage]}"
    --framework.img_next.loss_weight "${IMG_NEXT_LOSS_WEIGHT[$stage]}"
    --framework.img_next.use_teacher "${IMG_NEXT_USE_TEACHER}"
    --datasets.vla_data.bridge_annotations.steps_cache_path "${STEPS_CACHE_PATH}"
    --datasets.vla_data.bridge_annotations.write_steps_cache "${WRITE_STEPS_CACHE}"
  )
  [[ -n "${WANDB_ENTITY}" ]] && args+=( --wandb_entity "${WANDB_ENTITY}" )

  if [[ -n "${load_ckpt}" ]]; then
    args+=( --trainer.pretrained_checkpoint "${load_ckpt}" )
    [[ -n "${RELOAD_MODULES}" ]] && args+=( --trainer.reload_modules "${RELOAD_MODULES}" )
  fi

  local command=(
    torchrun
    --nproc_per_node="${NUM_GPUS}"
    --master_port="${MASTER_PORT}"
    laravla/training/train.py
    "${args[@]}"
    "$@"
  )
  if [[ "${DRY_RUN}" == "true" ]]; then
    printf 'stage %s command:' "${stage}"
    printf ' %q' "${command[@]}"
    printf '\n'
  else
    "${command[@]}"
  fi
}

for stage in 1 2 3 4; do
  (( stage < START_STAGE )) && continue
  (( stage > END_STAGE )) && continue

  load_ckpt=""
  if (( stage > 1 )); then
    prev=$((stage - 1))
    load_ckpt="${RUN_ROOT}/${RUN_ID_PREFIX}_stage_${prev}/checkpoints/steps_${CKPT_STEP[$prev]}_pytorch_model.pt"
    if [[ "${DRY_RUN}" != "true" && ! -f "${load_ckpt}" ]]; then
      echo "缺少上一阶段权重: ${load_ckpt}" >&2
      exit 1
    fi
  fi

  run_one_stage "${stage}" "${load_ckpt}" "$@"
done
