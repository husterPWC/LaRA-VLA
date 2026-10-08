#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "请使用 source $0 <gpu0,gpu1,gpu2,gpu3>" >&2
  exit 2
fi

gpu_set="${1:-}"
if [[ -z "${gpu_set}" ]]; then
  echo "用法: source scripts/reproduction/setup_r4_4gpu_env.sh <gpu0,gpu1,gpu2,gpu3>" >&2
  return 2
fi

IFS=',' read -r -a gpu_ids <<< "${gpu_set}"
if [[ "${#gpu_ids[@]}" -ne 4 ]]; then
  echo "必须提供恰好 4 个 GPU 编号，当前为: ${gpu_set}" >&2
  return 2
fi

declare -A seen_gpu_ids=()
for gpu_id in "${gpu_ids[@]}"; do
  if [[ ! "${gpu_id}" =~ ^[0-9]+$ ]]; then
    echo "GPU 编号必须是非负整数: ${gpu_id}" >&2
    unset seen_gpu_ids
    return 2
  fi
  if [[ -n "${seen_gpu_ids[${gpu_id}]:-}" ]]; then
    echo "GPU 编号重复: ${gpu_id}" >&2
    unset seen_gpu_ids
    return 2
  fi
  seen_gpu_ids["${gpu_id}"]=1
done
unset seen_gpu_ids

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
repro_root="${LARA_REPRO_ROOT:-$(dirname "${repo_root}")}"
fast_revision="ec4d7aa71691cac0b8bed6942be45684db2110f4"

export REPRO_ROOT="${repro_root}"
export REPO="${repo_root}"
export LARA_REPRO_ROOT="${repro_root}"
export LARAVLA_PYTHON="${LARAVLA_PYTHON:-$(command -v python)}"
export LARA_DATASET_ROOT="${repro_root}/datasets/libero_lerobot_all"
export LARA_BACKBONE="${repro_root}/StarVLA-Qwen3-VL-4B-Instruct-Action"
export FAST_REV="${fast_revision}"
export FAST_BASE="${repro_root}/dependencies/physical-intelligence-fast"
export LARA_FAST_TOKENIZER="${FAST_BASE}/${FAST_REV}"
export R4_RUN_ROOT="${repro_root}/runs/repro_r4_official_4gpu"
export CUDA_VISIBLE_DEVICES="${gpu_set}"
export R4_NUM_GPUS=4
export R4_PREFLIGHT_PER_DEVICE_BATCH=12
export R4_PREFLIGHT_GRADIENT_ACCUMULATION=2
export R4_STAGE1_PER_DEVICE_BATCH=12
export R4_STAGE1_GRADIENT_ACCUMULATION=2
export R4_STAGE2_PER_DEVICE_BATCH=16
export R4_STAGE2_GRADIENT_ACCUMULATION=2
export R4_STAGE3_PER_DEVICE_BATCH=16
export R4_STAGE3_GRADIENT_ACCUMULATION=2
export R4_GRADIENT_CHECKPOINTING="${R4_GRADIENT_CHECKPOINTING:-false}"

# 该服务器报告支持 CUDA peer access，但同 NUMA GPU 对的真实 NCCL P2P/IPC
# collective 会挂起；SHM 传输已通过四 rank 验证。该兼容项保持显式且可覆盖，
# 便于管理员修复主机后用 R4_NCCL_P2P_DISABLE=0 重新验证。
export R4_NCCL_P2P_DISABLE="${R4_NCCL_P2P_DISABLE:-1}"
if [[ "${R4_NCCL_P2P_DISABLE}" != "0" && "${R4_NCCL_P2P_DISABLE}" != "1" ]]; then
  echo "R4_NCCL_P2P_DISABLE 必须为 0 或 1" >&2
  return 2
fi
export NCCL_P2P_DISABLE="${R4_NCCL_P2P_DISABLE}"

# 清除定位 P2P 故障时使用的诊断覆盖。已验收的最小配置保留 CUMEM 自动检测，
# 只关闭 P2P。
unset NCCL_CUMEM_ENABLE
unset NCCL_P2P_LEVEL
export NCCL_DEBUG="${R4_NCCL_DEBUG:-WARN}"
unset NCCL_DEBUG_SUBSYS

printf 'R4 4-GPU profile: GPUs=%s run_root=%s NCCL_P2P_DISABLE=%s\n' \
  "${CUDA_VISIBLE_DEVICES}" "${R4_RUN_ROOT}" "${NCCL_P2P_DISABLE}"

unset gpu_set gpu_ids gpu_id repo_root repro_root fast_revision
