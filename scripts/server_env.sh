#!/usr/bin/env bash
# Server environment setup for LaRA-VLA spatial pipeline
# Usage: source scripts/server_env.sh

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export __GLX_VENDOR_LIBRARY_NAME=nvidia
export __EGL_VENDOR_LIBRARY_FILENAMES=/data/peixingxing/.local/share/glvnd/egl_vendor.d/10_nvidia.json
unset MUJOCO_EGL_DEVICE_ID

export LIBERO_HOME=/data/peixingxing/codevla/lara_repro/LIBERO
export LIBERO_CONFIG_PATH=$LIBERO_HOME/libero
export PYTHONPATH=$LIBERO_HOME:$PYTHONPATH

echo "✅ Server env ready: MUJOCO_GL=$MUJOCO_GL, LIBERO_HOME=$LIBERO_HOME"
