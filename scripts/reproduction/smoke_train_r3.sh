#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-}"
if [[ -z "${MODE}" ]]; then
  echo "用法: $0 {stage1|resume-stage1|stage2-1|stage2-2|stage2-3|stage3}" >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REPRO_ROOT="${LARA_REPRO_ROOT:-$(dirname "${REPO_ROOT}")}"
PYTHON_BIN="${LARAVLA_PYTHON:-/home/robot/miniconda3/envs/lara-vla/bin/python}"
DATASET_ROOT="${LARA_DATASET_ROOT:-/data/CodePWC/lara_datasets/libero_lerobot_all}"
BACKBONE="${LARA_BACKBONE:-${REPRO_ROOT}/StarVLA-Qwen3-VL-4B-Instruct-Action}"
FAST_REVISION="ec4d7aa71691cac0b8bed6942be45684db2110f4"
FAST_TOKENIZER="${LARA_FAST_TOKENIZER:-${REPRO_ROOT}/dependencies/physical-intelligence-fast/${FAST_REVISION}}"
RUN_ROOT="${R3_RUN_ROOT:-${REPRO_ROOT}/runs/repro_r3_official_smoke}"
GPU_ID="${R3_GPU_ID:-0}"
MASTER_PORT="${R3_MASTER_PORT:-29613}"
GRAD_ACCUM="${R3_GRAD_ACCUM:-1}"

for path in "${PYTHON_BIN}" "${DATASET_ROOT}" "${BACKBONE}" "${FAST_TOKENIZER}"; do
  if [[ ! -e "${path}" ]]; then
    echo "缺少 R3 必需路径: ${path}" >&2
    exit 1
  fi
done

mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/steps_cache" \
  "${REPRO_ROOT}/cache/huggingface" "${REPRO_ROOT}/cache/torch_extensions"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HOME="${REPRO_ROOT}/cache/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled
export TORCH_EXTENSIONS_DIR="${REPRO_ROOT}/cache/torch_extensions"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

COMMON_ARGS=(
  --config_yaml "${REPO_ROOT}/laravla/config/training/libero.yaml"
  --run_root_dir "${RUN_ROOT}"
  --wandb_project lara_r3_official_smoke
  --datasets.vla_data.data_root_dir "${DATASET_ROOT}"
  --datasets.vla_data.per_device_batch_size 1
  --datasets.vla_data.num_workers 0
  --datasets.vla_data.bridge_annotations.fast_tokenizer_name "${FAST_TOKENIZER}"
  --datasets.vla_data.bridge_annotations.steps_cache_path "${RUN_ROOT}/steps_cache"
  --datasets.vla_data.bridge_annotations.write_steps_cache true
  --framework.qwenvl.base_vlm "${BACKBONE}"
  --framework.qwenvl.cache_dir "${REPRO_ROOT}/cache/huggingface"
  --trainer.gradient_accumulation_steps "${GRAD_ACCUM}"
  --trainer.logging_frequency 1
  --trainer.eval_interval 20000000
  --trainer.min_save_step 0
  --trainer.deepspeed_zero_stage 3
  --trainer.deepspeed_offload_optimizer_device cpu
  --trainer.deepspeed_offload_param_device cpu
  --trainer.deepspeed_zero3_init_flag true
  --trainer.deepspeed_zero3_save_16bit_model true
  --trainer.enable_gradient_checkpointing false
  --trainer.save_final_model false
)

run_training() {
  local log_name="$1"
  shift
  cd "${REPO_ROOT}"
  set +e
  "${PYTHON_BIN}" -m torch.distributed.run \
    --nproc_per_node=1 \
    --master_port="${MASTER_PORT}" \
    laravla/training/train.py \
    "${COMMON_ARGS[@]}" \
    "$@" \
    2>&1 | tee "${RUN_ROOT}/logs/${log_name}.log"
  local status="${PIPESTATUS[0]}"
  set -e
  if [[ "${status}" -ne 0 ]]; then
    echo "R3 ${MODE} 失败，退出码 ${status}" >&2
    exit "${status}"
  fi
}

STAGE1_DIR="${RUN_ROOT}/stage1"
STAGE1_STEP2="${STAGE1_DIR}/checkpoints/steps_2_pytorch_model.pt"
STAGE1_STATE2="${STAGE1_DIR}/checkpoints/steps_2_training_state"
STAGE1_STEP3="${STAGE1_DIR}/checkpoints/steps_3_pytorch_model.pt"
STAGE21_STEP2="${RUN_ROOT}/stage2_1/checkpoints/steps_2_pytorch_model.pt"
STAGE22_STEP2="${RUN_ROOT}/stage2_2/checkpoints/steps_2_pytorch_model.pt"
STAGE23_STEP2="${RUN_ROOT}/stage2_3/checkpoints/steps_2_pytorch_model.pt"

case "${MODE}" in
  stage1)
    run_training stage1 \
      --run_id stage1 \
      --framework.training_stage reasoning_only \
      --datasets.vla_data.bridge_reasoning.stage 1 \
      --datasets.vla_data.bridge_reasoning.include_action_tokens true \
      --framework.latent_reasoning.vlm_loss_weight 1.0 \
      --framework.img_next.use_teacher true \
      --framework.img_next.loss_weight 0.1 \
      --trainer.max_train_steps 2 \
      --trainer.save_interval 2 \
      --trainer.save_training_state true
    ;;
  resume-stage1)
    test -f "${STAGE1_STEP2}"
    test -f "${STAGE1_STATE2}/trainer_state.json"
    run_training stage1_resume \
      --run_id stage1 \
      --framework.training_stage reasoning_only \
      --datasets.vla_data.bridge_reasoning.stage 1 \
      --datasets.vla_data.bridge_reasoning.include_action_tokens true \
      --framework.latent_reasoning.vlm_loss_weight 1.0 \
      --framework.img_next.use_teacher true \
      --framework.img_next.loss_weight 0.1 \
      --trainer.max_train_steps 3 \
      --trainer.save_interval 3 \
      --trainer.is_resume true \
      --trainer.resume_from_checkpoint "${STAGE1_STATE2}" \
      --trainer.save_training_state false
    ;;
  stage2-1)
    test -f "${STAGE1_STEP3}"
    run_training stage2_1 \
      --run_id stage2_1 \
      --framework.training_stage reasoning_only \
      --datasets.vla_data.bridge_reasoning.stage 2 \
      --datasets.vla_data.bridge_reasoning.include_action_tokens true \
      --framework.latent_reasoning.vlm_loss_weight 1.0 \
      --framework.img_next.use_teacher true \
      --framework.img_next.loss_weight 0.2 \
      --trainer.pretrained_checkpoint "${STAGE1_STEP3}" \
      --trainer.max_train_steps 2 \
      --trainer.save_interval 2 \
      --trainer.save_training_state false
    ;;
  stage2-2)
    test -f "${STAGE21_STEP2}"
    run_training stage2_2 \
      --run_id stage2_2 \
      --framework.training_stage reasoning_only \
      --datasets.vla_data.bridge_reasoning.stage 3 \
      --datasets.vla_data.bridge_reasoning.include_action_tokens true \
      --framework.latent_reasoning.vlm_loss_weight 1.0 \
      --framework.img_next.use_teacher true \
      --framework.img_next.loss_weight 0.2 \
      --trainer.pretrained_checkpoint "${STAGE21_STEP2}" \
      --trainer.max_train_steps 2 \
      --trainer.save_interval 2 \
      --trainer.save_training_state false
    ;;
  stage2-3)
    test -f "${STAGE22_STEP2}"
    run_training stage2_3 \
      --run_id stage2_3 \
      --framework.training_stage reasoning_only \
      --datasets.vla_data.bridge_reasoning.stage 4 \
      --datasets.vla_data.bridge_reasoning.include_action_tokens true \
      --framework.latent_reasoning.vlm_loss_weight 1.0 \
      --framework.img_next.use_teacher true \
      --framework.img_next.loss_weight 0.2 \
      --trainer.pretrained_checkpoint "${STAGE22_STEP2}" \
      --trainer.max_train_steps 2 \
      --trainer.save_interval 2 \
      --trainer.save_training_state false
    ;;
  stage3)
    test -f "${STAGE23_STEP2}"
    run_training stage3 \
      --run_id stage3 \
      --framework.training_stage full \
      --datasets.vla_data.bridge_reasoning.stage 4 \
      --datasets.vla_data.bridge_reasoning.include_action_tokens false \
      --framework.latent_reasoning.vlm_loss_weight 0 \
      --framework.img_next.use_teacher false \
      --framework.action_model.diffusion_model_cfg.dropout 0.1 \
      --trainer.pretrained_checkpoint "${STAGE23_STEP2}" \
      --trainer.reload_modules qwen_vl_interface \
      --trainer.max_train_steps 2 \
      --trainer.save_interval 2 \
      --trainer.save_training_state false
    ;;
  *)
    echo "未知模式: ${MODE}" >&2
    exit 2
    ;;
esac
