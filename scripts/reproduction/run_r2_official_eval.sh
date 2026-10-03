#!/usr/bin/env bash
# Run the released-checkpoint R2 protocol through the official evaluator.
set -euo pipefail

: "${LARAVLA_PYTHON:?Set LARAVLA_PYTHON to the lara-vla Python executable}"
: "${LIBERO_PYTHON:?Set LIBERO_PYTHON to the libero Python executable}"
: "${LIBERO_HOME:?Set LIBERO_HOME to the pinned LIBERO checkout}"
: "${LARA_CHECKPOINT:?Set LARA_CHECKPOINT to the prepared checkpoint path}"
: "${R2_RUN_ROOT:?Set R2_RUN_ROOT to an output directory outside Git}"
: "${CUDA_VISIBLE_DEVICES:?Expose exactly four GPUs for the official first run}"

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
export MUJOCO_EGL_DEVICE_ID=0
export NUMBA_CACHE_DIR="${R2_RUN_ROOT}/cache/numba"
export MPLCONFIGDIR="${R2_RUN_ROOT}/cache/matplotlib"

cat > "${R2_RUN_ROOT}/run.env" <<EOF
git_commit=$(git -C "${repo_root}" rev-parse HEAD)
libero_commit=$(git -C "${LIBERO_HOME}" rev-parse HEAD)
checkpoint=${LARA_CHECKPOINT}
checkpoint_sha256=$(sha256sum "${LARA_CHECKPOINT}" | cut -d' ' -f1)
cuda_visible_devices=${CUDA_VISIBLE_DEVICES}
task_suites=libero_goal,libero_spatial,libero_object,libero_10
rollouts_per_task=50
save_videos=false
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
  --output-json "${R2_RUN_ROOT}/official_checkpoint_libero_eval.json" \
  --output-csv "${R2_RUN_ROOT}/official_checkpoint_libero_eval.csv" \
  > "${R2_RUN_ROOT}/summary.stdout.log"

echo "R2 PASS: ${R2_RUN_ROOT}"
