#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export LIBERO_HOME="${LIBERO_HOME:-${REPO_ROOT}/../LIBERO}"
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-${LIBERO_HOME}/libero}"
export PYTHONPATH="${REPO_ROOT}:${LIBERO_HOME}${PYTHONPATH:+:${PYTHONPATH}}"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export LARA_BACKBONE="${LARA_BACKBONE:-${REPO_ROOT}/../StarVLA-Qwen3-VL-4B-Instruct-Action}"
export LARA_DATASET_ROOT="${LARA_DATASET_ROOT:-/data/CodePWC/lara_datasets/libero_lerobot_all}"
CONDA_BASE="${CONDA_BASE:-${HOME}/miniconda3}"
LARAVLA_PYTHON="${LARAVLA_PYTHON:-${CONDA_BASE}/envs/lara-vla/bin/python}"
LIBERO_PYTHON="${LIBERO_PYTHON:-${CONDA_BASE}/envs/libero/bin/python}"
"${LARAVLA_PYTHON}" -m pip check
"${LIBERO_PYTHON}" -m pip check
"${LARAVLA_PYTHON}" "${REPO_ROOT}/scripts/reproduction/env_check.py" server
"${LIBERO_PYTHON}" "${REPO_ROOT}/scripts/reproduction/env_check.py" client
