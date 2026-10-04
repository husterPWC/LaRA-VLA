# R2 服务器复现手册：使用官方 checkpoint 完成 LIBERO 正式评估

本文档面向服务器 `5880-1`，按照当前已知目录结构编写：

```text
/home/peixingxing/codevla/LaRA/
├── checkpoints/
│   └── LaRA-VLA-libero/
│       ├── checkpoints/
│       ├── config.yaml
│       ├── config.json
│       ├── dataset_statistics.json
│       ├── README.md
│       ├── summary.jsonl
│       └── eval_libero_implicit_parallel/   # 已有旧结果，不用于本次 R2
└── StarVLA-Qwen3-VL-4B-Instruct-Action/
```

目标是从干净环境开始，依次完成：环境安装、LIBERO 固定版本克隆、官方
checkpoint 本地路径映射、服务器单卡 R1 复验、官方 4 GPU/2000 rollout
R2 正式评估。R1 单卡复验通过前，不要启动正式 2000 rollouts。

第一次正式评估严格使用官方脚本的一套 suite 对应一个 policy server 的
方式，因此即使服务器有 8 张 GPU，也先只使用 4 张，不自行改写成 8 卡。

## 1. 记录服务器硬件信息

```bash
mkdir -p "$HOME/lara-r2-bootstrap"
cd "$HOME/lara-r2-bootstrap"
{
  date -Is
  hostname
  uname -a
  nvidia-smi
  nvidia-smi --query-gpu=index,name,compute_cap,memory.total,memory.used,driver_version --format=csv
  command -v nvcc && nvcc --version || true
  df -h
  free -h
  conda info --base
} | tee server-hardware.txt
```

当前已验证的软件组合是 PyTorch 2.6.0+cu124，所以编译 PyTorch3D 时需要
驱动支持 CUDA 12.4、本机有 CUDA 12.4 toolkit、选中 GPU 至少约 12 GiB
可用显存，并且主机内存足以让四个 server 同时加载 10.3 GB checkpoint。

`nvidia-smi` 顶部显示的是驱动支持的最高 CUDA 版本，不等于 `nvcc` 版本。
如果下面检查不是 `12.4`，先停止并把 `server-hardware.txt` 发回来，不要
自行替换 PyTorch 或 CUDA 版本。

```bash
export CUDA_HOME="$(dirname "$(dirname "$(readlink -f "$(command -v nvcc)")")")"
export NVCC_VERSION="$($CUDA_HOME/bin/nvcc --version | sed -n 's/.*release \([0-9.]*\),.*/\1/p')"
echo "CUDA_HOME=$CUDA_HOME"
echo "NVCC_VERSION=$NVCC_VERSION"
test "$NVCC_VERSION" = "12.4"
```

如果系统缺少编译或 EGL 依赖，并且你有 sudo 权限：

```bash
sudo apt-get update
sudo apt-get install -y \
  git git-lfs curl build-essential cmake ninja-build unzip ripgrep patchelf \
  libgl1-mesa-dev libegl1-mesa-dev libgles2-mesa-dev \
  libglfw3 libglfw3-dev libosmesa6-dev
```

没有 sudo 权限时先让管理员安装，不要临时换成未记录的渲染后端。

## 2. 设置服务器固定路径

服务器已经有 Conda，不需要重新安装 Miniconda：

```bash
export REPRO_ROOT="$HOME/codevla/LaRA"
export REPO="$REPRO_ROOT/LaRA-VLA"
export LIBERO_HOME="$REPRO_ROOT/LIBERO"
export OFFICIAL_RUN="$REPRO_ROOT/checkpoints/LaRA-VLA-libero"
export BACKBONE="$REPRO_ROOT/StarVLA-Qwen3-VL-4B-Instruct-Action"
export CKPT_NAME="steps_25000_pytorch_model.pt"
export CONDA_BASE="$(conda info --base)"

test -f "$OFFICIAL_RUN/config.yaml"
test -f "$OFFICIAL_RUN/config.json"
test -f "$OFFICIAL_RUN/dataset_statistics.json"
test -f "$OFFICIAL_RUN/checkpoints/$CKPT_NAME"
test -f "$BACKBONE/config.json"
test -f "$BACKBONE/model.safetensors.index.json"
```

如果 checkpoint 文件名不一致，先检查再修改 `CKPT_NAME`，不要重命名：

```bash
find "$OFFICIAL_RUN" -maxdepth 2 -type f -printf '%P %s bytes\n' | sort
```

记录已有权重的 SHA256。读取大文件需要一些时间：

```bash
{
  sha256sum "$OFFICIAL_RUN/checkpoints/$CKPT_NAME"
  find "$BACKBONE" -maxdepth 1 -name 'model-*.safetensors' -print0 \
    | sort -z | xargs -0 sha256sum
} | tee "$HOME/lara-r2-bootstrap/weights.sha256"
```

## 3. 克隆 LaRA-VLA 和固定版本 LIBERO

```bash
git clone https://github.com/husterPWC/LaRA-VLA.git "$REPO"
git -C "$REPO" checkout main
git -C "$REPO" pull --ff-only origin main

git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git "$LIBERO_HOME"
git -C "$LIBERO_HOME" checkout --detach 8f1084e3132a39270c3a13ebe37270a43ece2a01

git -C "$REPO" status --short
git -C "$REPO" rev-parse HEAD
git -C "$LIBERO_HOME" status --short
git -C "$LIBERO_HOME" rev-parse HEAD
```

记录 LaRA commit。LIBERO commit 必须严格等于：

```text
8f1084e3132a39270c3a13ebe37270a43ece2a01
```

设置两个环境的 Python 路径：

```bash
source "$CONDA_BASE/etc/profile.d/conda.sh"
export LARAVLA_PYTHON="$CONDA_BASE/envs/lara-vla/bin/python"
export LIBERO_PYTHON="$CONDA_BASE/envs/libero/bin/python"
```

## 4. 创建 LaRA policy server 环境

如果存在同名旧环境，先停止确认，不要直接覆盖：

```bash
conda env list
conda create -y -n lara-vla python=3.10.22 pip

"$LARAVLA_PYTHON" -m pip install --upgrade \
  pip==26.2.1 setuptools==75.8.0 wheel==0.47.0 ninja==1.13.2

"$LARAVLA_PYTHON" -m pip install \
  torch==2.6.0 torchvision==0.21.0 \
  --index-url https://download.pytorch.org/whl/cu124

"$LARAVLA_PYTHON" -m pip install \
  -r "$REPO/environment/lara-requirements.txt"
```

确认 PyTorch 使用 CUDA 12.4，并根据真实 GPU 自动生成编译架构：

```bash
"$LARAVLA_PYTHON" - <<'PY'
import torch
print("torch", torch.__version__)
print("torch CUDA", torch.version.cuda)
print("GPUs", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
print("capabilities", [torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count())])
assert torch.cuda.is_available()
assert torch.version.cuda == "12.4"
PY

"$CUDA_HOME/bin/nvcc" --version

export TORCH_CUDA_ARCH_LIST="$($LARAVLA_PYTHON - <<'PY'
import torch
caps = sorted({f"{a}.{b}" for a, b in (torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count()))})
if not caps:
    raise SystemExit("没有可见 CUDA GPU")
print(";".join(caps))
PY
)"
echo "TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"
```

构建固定版本官方 PyTorch3D 和修复元数据后的 decord wheel：

```bash
export BUILD_ROOT="$REPRO_ROOT/dependency-build"
export WHEEL_DIR="$BUILD_ROOT/wheels"
export MAX_JOBS="${MAX_JOBS:-4}"
export LARAVLA_PYTHON CUDA_HOME TORCH_CUDA_ARCH_LIST BUILD_ROOT WHEEL_DIR MAX_JOBS

bash "$REPO/scripts/reproduction/build_dependency_wheels.sh"

P3D_WHEEL="$(find "$WHEEL_DIR" -maxdepth 1 -name 'pytorch3d-0.7.6-*.whl' -print -quit)"
DECORD_WHEEL="$(find "$WHEEL_DIR" -maxdepth 1 -name 'decord-0.6.0-*.whl' -print -quit)"
test -n "$P3D_WHEEL"
test -n "$DECORD_WHEEL"

"$LARAVLA_PYTHON" -m pip install "$P3D_WHEEL" "$DECORD_WHEEL"
"$LARAVLA_PYTHON" -m pip install --no-deps -e "$REPO"
"$LARAVLA_PYTHON" -m pip check
```

不要安装 `pipablepytorch3d` 或 `eva-decord`：前者的通用 wheel 实际包含
CPython 3.11 扩展，Python 3.10 无法加载；后者会覆盖 `decord` 所拥有的
94 个文件。这是主机 R0 已定位的依赖打包根因，不是可选优化。

## 5. 创建独立的 LIBERO client 环境

```bash
conda env list
conda create -y -n libero python=3.10.22 pip

"$LIBERO_PYTHON" -m pip install --upgrade \
  pip==26.2.1 setuptools==75.8.0 wheel==0.47.0

"$LIBERO_PYTHON" -m pip install \
  torch==2.5.1 torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cpu

"$LIBERO_PYTHON" -m pip install \
  -r "$REPO/environment/libero-requirements.txt"
"$LIBERO_PYTHON" -m pip install --no-deps -e "$LIBERO_HOME"
"$LIBERO_PYTHON" -m pip check
```

client 使用 CPU torch 是有意设计：policy 在 LaRA server 中使用 GPU，
MuJoCo 仍通过 EGL 使用 GPU 渲染。torch 2.5.1 还能避免 torch 2.6 对
LIBERO 可信 init-state 文件默认 `weights_only=True` 的兼容补丁。

## 6. 生成服务器本地 LIBERO 配置

R2 不读取 demonstration HDF5，但 LIBERO 配置要求 `datasets` 指向存在的
目录。因此创建空目录占位，不下载训练数据：

```bash
mkdir -p "$REPRO_ROOT/eval-only-libero-datasets"
"$LARAVLA_PYTHON" "$REPO/scripts/reproduction/configure_libero.py" \
  --libero-home "$LIBERO_HOME" \
  --datasets "$REPRO_ROOT/eval-only-libero-datasets"

cat "$LIBERO_HOME/libero/config.yaml"
git -C "$LIBERO_HOME" status --short
```

LIBERO 工作树中只应出现未跟踪的 `libero/config.yaml`。

## 7. 为官方 checkpoint 创建本地路径视图

保持原始下载目录不变，创建只修改 `framework.qwenvl.base_vlm` 的隔离视图：

```bash
export PREPARED_RUN="$REPRO_ROOT/checkpoints/repro_r2_official"

"$LARAVLA_PYTHON" "$REPO/scripts/reproduction/prepare_official_checkpoint.py" \
  --source-run "$OFFICIAL_RUN" \
  --backbone "$BACKBONE" \
  --output-run "$PREPARED_RUN" \
  --checkpoint-name "$CKPT_NAME"

export PREPARED_CKPT="$PREPARED_RUN/checkpoints/$CKPT_NAME"

cat "$PREPARED_RUN/reproduction_manifest.json"
sha256sum "$OFFICIAL_RUN/checkpoints/$CKPT_NAME" "$PREPARED_CKPT"
stat -c '%h %i %n' \
  "$OFFICIAL_RUN/checkpoints/$CKPT_NAME" "$PREPARED_CKPT"
```

两个 checkpoint 的 SHA256 和 inode 必须相同。这里必须用硬链接：官方
loader 会先 `resolve()` checkpoint 路径再找相邻 `config.yaml`；符号链接
会跳回原始目录并读到远端 backbone 名称。两个目录都在
`$REPRO_ROOT/checkpoints` 下，不会复制第二份 10.3 GB 权重。

已有的 `$OFFICIAL_RUN/eval_libero_implicit_parallel` 是旧输出，本轮不读、
不删，也不向里面写入新结果。

## 8. 执行评估专用环境验收

```bash
mkdir -p "$REPRO_ROOT/r2-preflight/cache/numba" \
  "$REPRO_ROOT/r2-preflight/cache/matplotlib"

export PYTHONPATH="$REPO"
export LARA_BACKBONE="$BACKBONE"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0

"$LARAVLA_PYTHON" "$REPO/scripts/reproduction/env_check.py" server-eval \
  | tee "$REPRO_ROOT/r2-preflight/server-env.log"

export PYTHONPATH="$REPO:$LIBERO_HOME"
export LIBERO_CONFIG_PATH="$LIBERO_HOME/libero"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID=0
export NUMBA_CACHE_DIR="$REPRO_ROOT/r2-preflight/cache/numba"
export MPLCONFIGDIR="$REPRO_ROOT/r2-preflight/cache/matplotlib"

"$LIBERO_PYTHON" "$REPO/scripts/reproduction/env_check.py" client \
  | tee "$REPRO_ROOT/r2-preflight/client-env.log"
```

验收标准：正式模块导入、CUDA BF16、PyTorch3D native CUDA、本地 Qwen
processor、全部 40 个任务 init states，以及 Goal task 0 的 EGL 两路
256×256 图像 reset/render 全部通过。`server-eval` 只明确跳过训练数据视频
解码，因为当前服务器阶段只做 R2。

若出现 Numba `no locator available`，确认 `NUMBA_CACHE_DIR` 指向可写目录，
不要修改 robosuite。Gym、private macro、matplotlib/pyparsing 提示是已知
warning，需要保留日志，但不等于失败。

## 9. 在服务器单张 GPU 上复验 R1

```bash
export CONDA_BASE LARAVLA_PYTHON LIBERO_PYTHON
export LARA_CHECKPOINT="$PREPARED_CKPT"
export LARA_LIBERO_HOME="$LIBERO_HOME"
export LARA_GPU_ID=0
export LARA_PORT=10093
export LARA_R1_LOG_ROOT="$REPRO_ROOT/r2-preflight/r1"
mkdir -p "$LARA_R1_LOG_ROOT"

"$REPO/scripts/reproduction/smoke_inference.sh" server \
  > "$LARA_R1_LOG_ROOT/server.log" 2>&1 &
SERVER_PID=$!

for _ in $(seq 1 180); do
  grep -q 'server listening' "$LARA_R1_LOG_ROOT/server.log" && break
  kill -0 "$SERVER_PID"
  sleep 1
done
grep -q 'server listening' "$LARA_R1_LOG_ROOT/server.log"

set +e
"$REPO/scripts/reproduction/smoke_inference.sh" client \
  > "$LARA_R1_LOG_ROOT/client.stdout.log" 2>&1
CLIENT_STATUS=$?
set -e

kill -INT "$SERVER_PID" 2>/dev/null || true
wait "$SERVER_PID" 2>/dev/null || true
test "$CLIENT_STATUS" -eq 0

cat "$LARA_R1_LOG_ROOT/client/libero_goal.log"
grep -c 'Completed 4 reasoning passes' "$LARA_R1_LOG_ROOT/server.log"
nvidia-smi
```

验收重点是完整执行真实 episode，而不是强制单次 rollout 成功：checkpoint
完整加载、通信正常、两路图像、latent reasoning、action head 和 MuJoCo step
都实际执行，episode 正常结束，无 NaN/Inf、fallback 或 OOM，停止后显存释放。

主机 R1 恰好成功并记录 16 次 action-chunk inference，每次 4 个 latent
reasoning passes。服务器因硬件非确定性，轨迹长度和成功与否可能不同。

episode 完成且 client exit 0 后，robosuite EGL 析构器可能打印
`EGL_NOT_INITIALIZED`，要保留 warning，不做静默补丁。官方 evaluator 还会
构造 8 维 robot state 但不发送给 policy，并用 `min/max` 而非 `q01/q99`
反归一化 action。R2 先保持官方行为，不在此阶段修改。

## 10. 使用官方协议执行 4 GPU R2 正式评估

协议为 `4 suites × 10 tasks × 50 rollouts = 2000 rollouts`。

```bash
nvidia-smi
tmux new -s lara-r2
```

在 tmux 内重新设置变量。新结果写入 `$REPRO_ROOT/runs/`，不会读取或覆盖
checkpoint 目录中已有的 `eval_libero_implicit_parallel`：

```bash
export REPRO_ROOT="$HOME/codevla/LaRA"
export REPO="$REPRO_ROOT/LaRA-VLA"
export LIBERO_HOME="$REPRO_ROOT/LIBERO"
export CONDA_BASE="$(conda info --base)"
export LARAVLA_PYTHON="$CONDA_BASE/envs/lara-vla/bin/python"
export LIBERO_PYTHON="$CONDA_BASE/envs/libero/bin/python"
export PREPARED_RUN="$REPRO_ROOT/checkpoints/repro_r2_official"
export PREPARED_CKPT="$PREPARED_RUN/checkpoints/steps_25000_pytorch_model.pt"
export LARA_CHECKPOINT="$PREPARED_CKPT"
export R2_RUN_ROOT="$REPRO_ROOT/runs/r2-official-$(date +%Y%m%d-%H%M%S)"

# 专用 8 GPU 服务器第一次官方复现只使用前 4 张卡。
# 若由调度器分配 GPU，请改成调度器实际分配的 4 个 ID。
export CUDA_VISIBLE_DEVICES=0,1,2,3

cd "$REPO"
bash scripts/reproduction/run_r2_official_eval.sh
```

wrapper 会强制 4 GPU、固定完整协议、关闭视频、每卡一个 policy server、
启用 implicit latent reasoning、每 5 秒记录 GPU 状态，并保存 commit 和
checkpoint SHA256。只有 2000 条结果全部存在才生成最终 JSON/CSV。

官方并行脚本原来传 `--args.log_path`，Tyro 实际参数是 `--args.log-path`。
commit `9778eec` 只修正参数名，不改变评估语义。R1 wrapper 原来写死主机
Conda 路径，现在会读取服务器的 `CONDA_BASE` 和两个 Python。

## 11. 验证并保存正式结果

```bash
cat "$R2_RUN_ROOT/run.env"
cat "$R2_RUN_ROOT/system-info.txt"
cat "$R2_RUN_ROOT/official_checkpoint_libero_eval.csv"

"$LARAVLA_PYTHON" - <<'PY'
import json
import os

path = os.path.join(os.environ["R2_RUN_ROOT"], "official_checkpoint_libero_eval.json")
result = json.load(open(path))
assert result["total_episodes"] == 2000
assert set(result["suites"]) == {
    "libero_spatial", "libero_goal", "libero_object", "libero_10"
}
print(json.dumps(
    {name: data["success_percent"] for name, data in result["suites"].items()},
    indent=2,
    ensure_ascii=False,
))
print("四个 suite 宏平均：", result["macro_average_success_percent"])
PY

cp "$R2_RUN_ROOT/official_checkpoint_libero_eval.json" \
  "$REPO/results/official_checkpoint_libero_eval.json"
cp "$R2_RUN_ROOT/official_checkpoint_libero_eval.csv" \
  "$REPO/results/official_checkpoint_libero_eval.csv"
```

不要提交原始 rollout 日志、cache、视频、checkpoint 或 GPU 采样文件，只
提交审查过的小型 JSON/CSV 摘要和必要复现元数据。

| Suite | 论文成功率 |
|---|---:|
| Spatial | 96.4 |
| Goal | 98.6 |
| Object | 99.8 |
| Long / `libero_10` | 96.6 |
| 四个 suite 宏平均 | 97.9 |

若结果有差异，依次检查 checkpoint/LIBERO revision、reset state、seed、
两路图像预处理、action 反归一化、action horizon、episode 长度和 success
criterion。在这些一致性问题排除前，不修改模型。
