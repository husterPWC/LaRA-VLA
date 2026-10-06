#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-}"
if [[ -z "${MODE}" ]]; then
  echo "用法: $0 {preflight|dry-run|distributed-preflight|distributed-resume-preflight|reasoning|stage3} [额外训练参数]" >&2
  exit 2
fi
shift
EXTRA_ARGS=("$@")

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REPRO_ROOT="${LARA_REPRO_ROOT:-$(dirname "${REPO_ROOT}")}"
PYTHON_BIN="${LARAVLA_PYTHON:-$(command -v python)}"
DATASET_ROOT="${LARA_DATASET_ROOT:-${REPRO_ROOT}/datasets/libero_lerobot_all}"
BACKBONE="${LARA_BACKBONE:-${REPRO_ROOT}/StarVLA-Qwen3-VL-4B-Instruct-Action}"
FAST_REVISION="ec4d7aa71691cac0b8bed6942be45684db2110f4"
FAST_TOKENIZER="${LARA_FAST_TOKENIZER:-${REPRO_ROOT}/dependencies/physical-intelligence-fast/${FAST_REVISION}}"
DATASET_REVISION="fbe4f71c2fd5a6e4f9171c78e51e2f6567277fff"
RUN_ROOT="${R4_RUN_ROOT:-${REPRO_ROOT}/runs/repro_r4_official}"
STEPS_CACHE_PATH="${R4_STEPS_CACHE_PATH:-${RUN_ROOT}/steps_cache}"
NUM_GPUS="${R4_NUM_GPUS:-8}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
MASTER_PORT="${R4_MASTER_PORT:-29614}"
NUM_WORKERS="${R4_NUM_WORKERS:-4}"
MAX_IDLE_MEMORY_MIB="${R4_MAX_USED_MEMORY_MIB_BEFORE_START:-1024}"
STATE_RETENTION="${R4_STATE_RETENTION:-2}"
ZERO_STAGE="${R4_ZERO_STAGE:-2}"
OFFLOAD_OPTIMIZER="${R4_OFFLOAD_OPTIMIZER:-none}"
OFFLOAD_PARAM="${R4_OFFLOAD_PARAM:-none}"
ZERO3_INIT="${R4_ZERO3_INIT:-false}"
ZERO3_SAVE="${R4_ZERO3_SAVE_16BIT:-false}"
GRADIENT_CHECKPOINTING="${R4_GRADIENT_CHECKPOINTING:-false}"

cd "${REPO_ROOT}"

SUITES=(
  libero_goal_no_noops_1.0.0_lerobot
  libero_object_no_noops_1.0.0_lerobot
  libero_spatial_no_noops_1.0.0_lerobot
  libero_10_no_noops_1.0.0_lerobot
)

export CUDA_VISIBLE_DEVICES
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HOME="${R4_HF_HOME:-${REPRO_ROOT}/cache/huggingface}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE="${WANDB_MODE:-offline}"
export TORCH_EXTENSIONS_DIR="${R4_TORCH_EXTENSIONS_DIR:-${REPRO_ROOT}/cache/torch_extensions}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

dataset_revision() {
  local metadata_root="${DATASET_ROOT}/.cache/huggingface/download"
  [[ -d "${metadata_root}" ]] || return 1
  find "${metadata_root}" -type f -name '*.metadata' -print0 \
    | xargs -0 -r -n1 head -n1 \
    | sort -u
}

check_paths() {
  [[ -x "${PYTHON_BIN}" ]] || { echo "Python 不可执行: ${PYTHON_BIN}" >&2; exit 1; }
  [[ -d "${DATASET_ROOT}" ]] || { echo "缺少数据集: ${DATASET_ROOT}" >&2; exit 1; }
  [[ -d "${BACKBONE}" ]] || { echo "缺少 backbone: ${BACKBONE}" >&2; exit 1; }
  [[ -f "${BACKBONE}/config.json" ]] || { echo "backbone 缺少 config.json" >&2; exit 1; }
  [[ -f "${BACKBONE}/model.safetensors.index.json" ]] || { echo "backbone 缺少权重索引" >&2; exit 1; }
  [[ -f "${FAST_TOKENIZER}/tokenizer.json" ]] || { echo "缺少 FAST tokenizer: ${FAST_TOKENIZER}" >&2; exit 1; }
  for suite in "${SUITES[@]}"; do
    [[ -d "${DATASET_ROOT}/${suite}/data" ]] || { echo "${suite} 缺少 data" >&2; exit 1; }
    [[ -d "${DATASET_ROOT}/${suite}/videos" ]] || { echo "${suite} 缺少 videos" >&2; exit 1; }
    [[ -f "${DATASET_ROOT}/${suite}/annotations/episode_dense_captions_full.jsonl" ]] || { echo "${suite} 缺少 CoT" >&2; exit 1; }
    [[ -f "${DATASET_ROOT}/${suite}/annotations/episode_sam3_bboxes_from_dino_final.jsonl" ]] || { echo "${suite} 缺少 bbox" >&2; exit 1; }
  done
  local revisions
  revisions="$(dataset_revision)" || { echo "数据集缺少 Hugging Face revision metadata" >&2; exit 1; }
  [[ "${revisions}" == "${DATASET_REVISION}" ]] || {
    echo "数据集 revision 不一致: ${revisions}" >&2
    exit 1
  }
  [[ -z "$(git -C "${REPO_ROOT}" status --porcelain)" ]] || {
    echo "R4 要求干净 worktree" >&2
    exit 1
  }
}

check_gpus() {
  command -v nvidia-smi >/dev/null
  command -v torchrun >/dev/null
  local count
  count="$(${PYTHON_BIN} -c 'import torch; print(torch.cuda.device_count())')"
  [[ "${count}" -eq "${NUM_GPUS}" ]] || {
    echo "可见 GPU 数量为 ${count}，期望 ${NUM_GPUS}" >&2
    exit 1
  }
  local gpu used
  IFS=',' read -r -a visible_gpus <<< "${CUDA_VISIBLE_DEVICES}"
  [[ "${#visible_gpus[@]}" -eq "${NUM_GPUS}" ]] || {
    echo "CUDA_VISIBLE_DEVICES 与 R4_NUM_GPUS 数量不一致" >&2
    exit 1
  }
  for gpu in "${visible_gpus[@]}"; do
    used="$(nvidia-smi -i "${gpu}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
    if (( used > MAX_IDLE_MEMORY_MIB )); then
      echo "GPU ${gpu} 已使用 ${used} MiB，超过启动阈值 ${MAX_IDLE_MEMORY_MIB} MiB" >&2
      exit 1
    fi
  done
}

check_steps_cache() {
  [[ -d "${STEPS_CACHE_PATH}" ]] || {
    echo "缺少 R4 steps cache，请先运行 preflight: ${STEPS_CACHE_PATH}" >&2
    exit 1
  }
  local count
  count="$(find "${STEPS_CACHE_PATH}" -maxdepth 1 -type f -name 'steps_*.pkl' | wc -l)"
  [[ "${count}" -eq 4 ]] || {
    echo "R4 steps cache 应恰好包含 4 个文件，实际为 ${count}" >&2
    exit 1
  }
}

write_metadata() {
  local mode="$1"
  local out="${RUN_ROOT}/metadata/${mode}"
  mkdir -p "${out}"
  {
    echo "timestamp=$(date --iso-8601=seconds)"
    echo "git_commit=$(git -C "${REPO_ROOT}" rev-parse HEAD)"
    echo "dataset_revision=${DATASET_REVISION}"
    echo "dataset_root=${DATASET_ROOT}"
    echo "backbone=${BACKBONE}"
    echo "fast_tokenizer=${FAST_TOKENIZER}"
    echo "fast_tokenizer_sha256=$(sha256sum "${FAST_TOKENIZER}/tokenizer.json" | awk '{print $1}')"
    echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
    echo "num_gpus=${NUM_GPUS}"
    echo "zero_stage=${ZERO_STAGE}"
    echo "gradient_checkpointing=${GRADIENT_CHECKPOINTING}"
    echo "state_retention=${STATE_RETENTION}"
  } > "${out}/run.env"
  nvidia-smi > "${out}/nvidia-smi.txt"
  "${PYTHON_BIN}" -m pip freeze > "${out}/pip-freeze.txt"
}

check_effective_batch() {
  local label="$1"
  local per_device="$2"
  local accumulation="$3"
  local expected="$4"
  local actual=$((per_device * NUM_GPUS * accumulation))
  if (( actual != expected )) && [[ "${R4_ALLOW_EFFECTIVE_BATCH_CHANGE:-false}" != "true" ]]; then
    echo "${label} effective global batch=${actual}，官方目标=${expected}" >&2
    echo "请成对调整 per-device batch/gradient accumulation；若确需改变数学 batch，显式设置 R4_ALLOW_EFFECTIVE_BATCH_CHANGE=true" >&2
    exit 1
  fi
}

COMMON_ARGS=(
  --datasets.vla_data.data_root_dir "${DATASET_ROOT}"
  --datasets.vla_data.num_workers "${NUM_WORKERS}"
  --datasets.vla_data.bridge_annotations.fast_tokenizer_name "${FAST_TOKENIZER}"
  --datasets.vla_data.bridge_annotations.steps_cache_path "${STEPS_CACHE_PATH}"
  --datasets.vla_data.bridge_annotations.write_steps_cache false
  --framework.qwenvl.base_vlm "${BACKBONE}"
  --framework.qwenvl.cache_dir "${HF_HOME}"
  --trainer.eval_interval 20000000
  --trainer.save_final_model false
  --trainer.save_training_state true
  --trainer.max_training_state_checkpoints "${STATE_RETENTION}"
  --trainer.deepspeed_zero_stage "${ZERO_STAGE}"
  --trainer.deepspeed_offload_optimizer_device "${OFFLOAD_OPTIMIZER}"
  --trainer.deepspeed_offload_param_device "${OFFLOAD_PARAM}"
  --trainer.deepspeed_zero3_init_flag "${ZERO3_INIT}"
  --trainer.deepspeed_zero3_save_16bit_model "${ZERO3_SAVE}"
  --trainer.enable_gradient_checkpointing "${GRADIENT_CHECKPOINTING}"
)

run_reasoning() {
  local target_root="$1"
  shift
  mkdir -p "${target_root}/logs"
  set +e
  RUN_ROOT="${target_root}" \
  NUM_GPUS="${NUM_GPUS}" \
  MASTER_PORT="${MASTER_PORT}" \
  STEPS_CACHE_PATH="${STEPS_CACHE_PATH}" \
  WRITE_STEPS_CACHE=false \
  bash "${REPO_ROOT}/scripts/run_libero_multistage.sh" \
    "${COMMON_ARGS[@]}" "$@" \
    2>&1 | tee -a "${target_root}/logs/train.stdout.log"
  local status="${PIPESTATUS[0]}"
  set -e
  return "${status}"
}

case "${MODE}" in
  preflight)
    check_paths
    check_gpus
    mkdir -p "${RUN_ROOT}/preflight" "${STEPS_CACHE_PATH}" "${HF_HOME}" "${TORCH_EXTENSIONS_DIR}"
    "${PYTHON_BIN}" -m pip check
    for stage in 1 2 3 4; do
      cache_args=(--steps-cache-path "${STEPS_CACHE_PATH}")
      (( stage == 1 )) && cache_args+=(--write-steps-cache)
      "${PYTHON_BIN}" "${REPO_ROOT}/scripts/reproduction/check_training_data.py" \
        --config "${REPO_ROOT}/laravla/config/training/libero.yaml" \
        --dataset-root "${DATASET_ROOT}" \
        --fast-tokenizer "${FAST_TOKENIZER}" \
        --stage "${stage}" \
        --output "${RUN_ROOT}/preflight/stage_${stage}.json" \
        "${cache_args[@]}"
    done
    [[ "$(find "${STEPS_CACHE_PATH}" -maxdepth 1 -type f -name 'steps_*.pkl' | wc -l)" -eq 4 ]]
    write_metadata preflight
    echo "R4 PREFLIGHT PASS: ${RUN_ROOT}/preflight"
    ;;
  dry-run)
    DRY_RUN=true RUN_ROOT="${RUN_ROOT}/reasoning" STEPS_CACHE_PATH="${STEPS_CACHE_PATH}" \
      bash "${REPO_ROOT}/scripts/run_libero_multistage.sh" "${COMMON_ARGS[@]}" "${EXTRA_ARGS[@]}"
    stage2_checkpoint="${RUN_ROOT}/reasoning/libero_vlm_stage_4/checkpoints/steps_2000_pytorch_model.pt"
    DRY_RUN=true RUN_ROOT="${RUN_ROOT}/action" \
      PRETRAINED_CKPT="${stage2_checkpoint}" \
      bash "${REPO_ROOT}/scripts/run_laravla_libero.sh" "${COMMON_ARGS[@]}" "${EXTRA_ARGS[@]}"
    ;;
  distributed-preflight)
    check_paths
    check_gpus
    check_steps_cache
    write_metadata distributed-preflight
    START_STAGE=1 END_STAGE=1 \
    STAGE1_PER_DEVICE_BATCH="${R4_PREFLIGHT_PER_DEVICE_BATCH:-12}" \
    STAGE1_GRADIENT_ACCUMULATION="${R4_PREFLIGHT_GRADIENT_ACCUMULATION:-1}" \
      run_reasoning "${RUN_ROOT}/distributed_preflight" \
        --trainer.max_train_steps 1 \
        --trainer.save_interval 1 \
        --trainer.min_save_step 0 \
        --trainer.logging_frequency 1 \
        --trainer.max_training_state_checkpoints 1 \
        "${EXTRA_ARGS[@]}"
    ;;
  distributed-resume-preflight)
    check_paths
    check_gpus
    check_steps_cache
    resume_state="${RUN_ROOT}/distributed_preflight/libero_vlm_stage_1/checkpoints/steps_1_training_state"
    [[ -f "${resume_state}/trainer_state.json" ]] || {
      echo "缺少分布式预检状态: ${resume_state}" >&2
      exit 1
    }
    write_metadata distributed-resume-preflight
    START_STAGE=1 END_STAGE=1 \
    STAGE1_PER_DEVICE_BATCH="${R4_PREFLIGHT_PER_DEVICE_BATCH:-12}" \
    STAGE1_GRADIENT_ACCUMULATION="${R4_PREFLIGHT_GRADIENT_ACCUMULATION:-1}" \
      run_reasoning "${RUN_ROOT}/distributed_preflight" \
        --trainer.max_train_steps 2 \
        --trainer.save_interval 2 \
        --trainer.min_save_step 0 \
        --trainer.logging_frequency 1 \
        --trainer.is_resume true \
        --trainer.resume_from_checkpoint "${resume_state}" \
        --trainer.max_training_state_checkpoints 1 \
        "${EXTRA_ARGS[@]}"
    ;;
  reasoning)
    check_paths
    check_gpus
    check_steps_cache
    check_effective_batch Stage-I "${R4_STAGE1_PER_DEVICE_BATCH:-12}" "${R4_STAGE1_GRADIENT_ACCUMULATION:-1}" 96
    check_effective_batch Stage-II "${R4_STAGE2_PER_DEVICE_BATCH:-16}" "${R4_STAGE2_GRADIENT_ACCUMULATION:-1}" 128
    write_metadata reasoning
    START_STAGE="${R4_START_STAGE:-1}" END_STAGE="${R4_END_STAGE:-4}" \
    STAGE1_PER_DEVICE_BATCH="${R4_STAGE1_PER_DEVICE_BATCH:-12}" \
    STAGE1_GRADIENT_ACCUMULATION="${R4_STAGE1_GRADIENT_ACCUMULATION:-1}" \
    STAGE2_PER_DEVICE_BATCH="${R4_STAGE2_PER_DEVICE_BATCH:-16}" \
    STAGE2_GRADIENT_ACCUMULATION="${R4_STAGE2_GRADIENT_ACCUMULATION:-1}" \
      run_reasoning "${RUN_ROOT}/reasoning" "${EXTRA_ARGS[@]}"
    ;;
  stage3)
    check_paths
    check_gpus
    check_steps_cache
    check_effective_batch Stage-III "${R4_STAGE3_PER_DEVICE_BATCH:-16}" "${R4_STAGE3_GRADIENT_ACCUMULATION:-1}" 128
    write_metadata stage3
    mkdir -p "${RUN_ROOT}/action/logs"
    stage2_checkpoint="${RUN_ROOT}/reasoning/libero_vlm_stage_4/checkpoints/steps_2000_pytorch_model.pt"
    set +e
    RUN_ROOT="${RUN_ROOT}/action" \
    PRETRAINED_CKPT="${stage2_checkpoint}" \
    NUM_GPUS="${NUM_GPUS}" \
    MASTER_PORT="$((MASTER_PORT + 1))" \
    PER_DEVICE_BATCH_SIZE="${R4_STAGE3_PER_DEVICE_BATCH:-16}" \
    GRADIENT_ACCUMULATION_STEPS="${R4_STAGE3_GRADIENT_ACCUMULATION:-1}" \
      bash "${REPO_ROOT}/scripts/run_laravla_libero.sh" \
        "${COMMON_ARGS[@]}" "${EXTRA_ARGS[@]}" \
        2>&1 | tee -a "${RUN_ROOT}/action/logs/train.stdout.log"
    status="${PIPESTATUS[0]}"
    set -e
    exit "${status}"
    ;;
  *)
    echo "未知 R4 模式: ${MODE}" >&2
    exit 2
    ;;
esac
