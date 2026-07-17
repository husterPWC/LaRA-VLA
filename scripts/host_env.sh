#!/usr/bin/env bash
# Server environment setup for LaRA-VLA spatial pipeline
# Usage: source scripts/server_env.sh

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export __GLX_VENDOR_LIBRARY_NAME=nvidia

export LIBERO_HOME=/data/peixingxing/codevla/lara_repro/LIBERO
export LIBERO_CONFIG_PATH=$LIBERO_HOME/libero
export PYTHONPATH=$LIBERO_HOME:$PYTHONPATH

echo "✅ Server env ready: MUJOCO_GL=$MUJOCO_GL, LIBERO_HOME=$LIBERO_HOME"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export LIBERO_HOME=/home/robot/codePWC/lara_repro/LIBERO
export LIBERO_CONFIG_PATH=$LIBERO_HOME/libero
export PYTHONPATH=$LIBERO_HOME:$PYTHONPATH
echo "✅ Host env ready: LIBERO_HOME=$LIBERO_HOME"
