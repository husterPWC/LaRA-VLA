#!/usr/bin/env bash
set -euo pipefail

component="${1:-}"
if [[ "${component}" != "server" && "${component}" != "client" ]]; then
  echo "usage: $0 {server|client}" >&2
  exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
workspace="$(cd "${repo_root}/.." && pwd)"
checkpoint="${LARA_CHECKPOINT:-${workspace}/checkpoints/repro_r1_official/checkpoints/steps_25000_pytorch_model.pt}"
port="${LARA_PORT:-10093}"
gpu_id="${LARA_GPU_ID:-0}"
log_root="${LARA_R1_LOG_ROOT:-${repo_root}/logs/repro_r1}"

if [[ ! -f "${checkpoint}" ]]; then
  echo "checkpoint does not exist: ${checkpoint}" >&2
  exit 1
fi
mkdir -p "${log_root}/client" /tmp/lara-numba-cache /tmp/lara-matplotlib

if [[ "${component}" == "server" ]]; then
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  export CUDA_VISIBLE_DEVICES="${gpu_id}"
  export PYTHONPATH="${repo_root}"
  exec /home/robot/miniconda3/envs/lara-vla/bin/python -u \
    "${repo_root}/deployment/model_server/server_policy.py" \
    --ckpt_path "${checkpoint}" \
    --port "${port}" \
    --use_bf16
fi

libero_home="${LARA_LIBERO_HOME:-${workspace}/LIBERO}"
export PYTHONPATH="${repo_root}:${libero_home}"
export LIBERO_CONFIG_PATH="${libero_home}/libero"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID=0
export CUDA_VISIBLE_DEVICES="${gpu_id}"
export NUMBA_CACHE_DIR=/tmp/lara-numba-cache
export MPLCONFIGDIR=/tmp/lara-matplotlib

exec /usr/bin/time -v /home/robot/miniconda3/envs/libero/bin/python -u \
  "${repo_root}/examples/LIBERO/eval_libero.py" \
  --args.pretrained-path "${checkpoint}" \
  --args.host 127.0.0.1 \
  --args.port "${port}" \
  --args.task-suite-name libero_goal \
  --args.task-id 0 \
  --args.num-trials-per-task 1 \
  --args.seed 7 \
  --args.enable-latent-reasoning \
  --args.cot-mode implicit \
  --args.log-path "${log_root}/client"
