#!/usr/bin/env bash
# Run the released-checkpoint R2 protocol through the official evaluator.
set -euo pipefail

: "${LARAVLA_PYTHON:?Set LARAVLA_PYTHON to the lara-vla Python executable}"
: "${LIBERO_PYTHON:?Set LIBERO_PYTHON to the libero Python executable}"
: "${LIBERO_HOME:?Set LIBERO_HOME to the pinned LIBERO checkout}"
: "${LARA_CHECKPOINT:?Set LARA_CHECKPOINT to the prepared checkpoint path}"
: "${R2_RUN_ROOT:?Set R2_RUN_ROOT to an output directory outside Git}"
: "${CUDA_VISIBLE_DEVICES:?Expose exactly four GPUs for the official first run}"
RESULT_PREFIX="${RESULT_PREFIX:-official_checkpoint}"
EVAL_LABEL="${EVAL_LABEL:-R2}"

if [[ ! "${RESULT_PREFIX}" =~ ^[a-zA-Z0-9_-]+$ ]]; then
  echo "RESULT_PREFIX 只能包含字母、数字、下划线和连字符: ${RESULT_PREFIX}" >&2
  exit 2
fi

IFS=',' read -r -a gpu_ids <<< "${CUDA_VISIBLE_DEVICES}"
if [[ "${#gpu_ids[@]}" -ne 4 ]]; then
  echo "R2 official first run requires exactly four visible GPUs; got: ${CUDA_VISIBLE_DEVICES}" >&2
  exit 1
fi
for executable in "${LARAVLA_PYTHON}" "${LIBERO_PYTHON}"; do
  if [[ ! -x "${executable}" ]]; then
    echo "Python is not executable: ${executable}" >&2
    exit 1
  fi
done

# robosuite 1.4.0 interprets MUJOCO_EGL_DEVICE_ID as a physical GPU ID and
# requires it to be present in CUDA_VISIBLE_DEVICES.  Default to the first GPU
# in the explicitly selected R2 pool instead of assuming physical GPU 0.
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-${gpu_ids[0]}}"
egl_device_visible=false
for gpu_id in "${gpu_ids[@]}"; do
  if [[ "${gpu_id}" == "${MUJOCO_EGL_DEVICE_ID}" ]]; then
    egl_device_visible=true
    break
  fi
done
if [[ "${egl_device_visible}" != "true" ]]; then
  echo "MUJOCO_EGL_DEVICE_ID=${MUJOCO_EGL_DEVICE_ID} is not in CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" >&2
  exit 1
fi
if [[ ! -f "${LARA_CHECKPOINT}" ]]; then
  echo "Checkpoint does not exist: ${LARA_CHECKPOINT}" >&2
  exit 1
fi
if [[ ! -d "${LIBERO_HOME}" ]]; then
  echo "LIBERO_HOME does not exist: ${LIBERO_HOME}" >&2
  exit 1
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
eval_dir="${R2_RUN_ROOT}/eval"
if [[ -d "${R2_RUN_ROOT}" ]] && [[ -n "$(find "${R2_RUN_ROOT}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "R2_RUN_ROOT must be new or empty: ${R2_RUN_ROOT}" >&2
  exit 1
fi
mkdir -p "${R2_RUN_ROOT}" "${eval_dir}" "${R2_RUN_ROOT}/cache/numba" \
  "${R2_RUN_ROOT}/cache/matplotlib" "${R2_RUN_ROOT}/cache/huggingface"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME="${R2_RUN_ROOT}/cache/huggingface"
export LIBERO_CONFIG_PATH="${LIBERO_HOME}/libero"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export NUMBA_CACHE_DIR="${R2_RUN_ROOT}/cache/numba"
export MPLCONFIGDIR="${R2_RUN_ROOT}/cache/matplotlib"

cat > "${R2_RUN_ROOT}/run.env" <<EOF
git_commit=$(git -C "${repo_root}" rev-parse HEAD)
libero_commit=$(git -C "${LIBERO_HOME}" rev-parse HEAD)
checkpoint=${LARA_CHECKPOINT}
checkpoint_sha256=$(sha256sum "${LARA_CHECKPOINT}" | cut -d' ' -f1)
cuda_visible_devices=${CUDA_VISIBLE_DEVICES}
mujoco_egl_device_id=${MUJOCO_EGL_DEVICE_ID}
task_suites=libero_goal,libero_spatial,libero_object,libero_10
rollouts_per_task=50
save_videos=false
result_prefix=${RESULT_PREFIX}
EOF

{
  date -Is
  hostname
  uname -a
  nvidia-smi
  "${LARAVLA_PYTHON}" -c 'import torch, transformers; print("server python/torch/transformers", torch.__version__, torch.version.cuda, transformers.__version__)'
  "${LIBERO_PYTHON}" -c 'import mujoco, numpy, torch; print("client torch/numpy/mujoco", torch.__version__, numpy.__version__, mujoco.__version__)'
} > "${R2_RUN_ROOT}/system-info.txt" 2>&1

monitor_pid=""
cleanup_monitor() {
  if [[ -n "${monitor_pid}" ]] && kill -0 "${monitor_pid}" 2>/dev/null; then
    kill "${monitor_pid}" 2>/dev/null || true
    wait "${monitor_pid}" 2>/dev/null || true
  fi
}
trap cleanup_monitor EXIT
nvidia-smi \
  --query-gpu=timestamp,index,name,memory.used,memory.free,utilization.gpu \
  --format=csv,noheader,nounits -l 5 \
  > "${R2_RUN_ROOT}/gpu.csv" &
monitor_pid="$!"

set +e
TASK_SUITES=libero_goal,libero_spatial,libero_object,libero_10 \
NUM_TRIALS_PER_TASK=50 \
SAVE_VIDEOS=false \
EVAL_DIR="${eval_dir}" \
bash "${repo_root}/examples/LIBERO/eval_libero_all.sh" "${LARA_CHECKPOINT}" \
  2>&1 | tee "${R2_RUN_ROOT}/eval_all.stdout.log"
eval_status="${PIPESTATUS[0]}"
set -e
cleanup_monitor
monitor_pid=""
if [[ "${eval_status}" -ne 0 ]]; then
  echo "Official evaluator failed with exit status ${eval_status}" >&2
  exit "${eval_status}"
fi

"${LARAVLA_PYTHON}" "${script_dir}/summarize_libero_eval.py" \
  --log-dir "${eval_dir}/logs" \
  --expected-rollouts-per-task 50 \
  --output-json "${R2_RUN_ROOT}/${RESULT_PREFIX}_libero_eval.json" \
  --output-csv "${R2_RUN_ROOT}/${RESULT_PREFIX}_libero_eval.csv" \
  > "${R2_RUN_ROOT}/summary.stdout.log"

echo "${EVAL_LABEL} PASS: ${R2_RUN_ROOT}"
