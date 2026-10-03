# R2 server runbook: released checkpoint on LIBERO

This runbook starts from an otherwise unconfigured Linux server. It installs
two clean Python 3.10 environments, checks out the exact LIBERO revision,
prepares the already-downloaded released checkpoint without changing model
semantics, repeats the one-GPU R1 smoke test, and then runs the official
four-GPU, 2,000-rollout R2 protocol.

Do not start the 2,000 rollouts until every check through the server R1 smoke
test passes. The first formal run deliberately uses four GPUs even on an
eight-GPU server because the released evaluator starts one policy server for
each of the four suites.

## 1. Record hardware before installation

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
} | tee server-hardware.txt
```

The validated build uses PyTorch 2.6.0 cu124 and therefore needs a driver that
supports CUDA 12.4 plus a CUDA 12.4 toolkit containing `nvcc` to compile
PyTorch3D. If `/usr/local/cuda-12.4/bin/nvcc` is absent, stop and identify the
installed toolkit before changing package versions. A newer driver is fine;
the CUDA version shown at the top of `nvidia-smi` is driver capability, not the
toolkit used to compile extensions.

Install system build and headless-rendering prerequisites if they are absent:

```bash
sudo apt-get update
sudo apt-get install -y \
  git git-lfs curl build-essential cmake ninja-build unzip ripgrep patchelf \
  libgl1-mesa-dev libegl1-mesa-dev libgles2-mesa-dev \
  libglfw3 libglfw3-dev libosmesa6-dev
```

On a managed cluster without `sudo`, ask the administrator for these packages;
do not replace EGL with an unrecorded rendering backend.

## 2. Define machine-local paths

Edit only the two existing weight paths and, if necessary, `CUDA_HOME`:

```bash
export REPRO_ROOT="$HOME/LaRA-reproduction"
export REPO="$REPRO_ROOT/LaRA-VLA"
export LIBERO_HOME="$REPRO_ROOT/LIBERO"
export CONDA_BASE="$HOME/miniconda3"
export CUDA_HOME="/usr/local/cuda-12.4"

# Existing downloads on the server. OFFICIAL_RUN must contain config.yaml,
# config.json, dataset_statistics.json, and checkpoints/<checkpoint>.pt.
export OFFICIAL_RUN="/ABSOLUTE/PATH/TO/LaRA-VLA-libero"
export BACKBONE="/ABSOLUTE/PATH/TO/StarVLA-Qwen3-VL-4B-Instruct-Action"
export CKPT_NAME="steps_25000_pytorch_model.pt"

mkdir -p "$REPRO_ROOT"
test -x "$CUDA_HOME/bin/nvcc"
test -f "$OFFICIAL_RUN/config.yaml"
test -f "$OFFICIAL_RUN/config.json"
test -f "$OFFICIAL_RUN/dataset_statistics.json"
test -f "$OFFICIAL_RUN/checkpoints/$CKPT_NAME"
test -f "$BACKBONE/config.json"
test -f "$BACKBONE/model.safetensors.index.json"

{
  sha256sum "$OFFICIAL_RUN/checkpoints/$CKPT_NAME"
  find "$BACKBONE" -maxdepth 1 -name 'model-*.safetensors' -print0 \
    | sort -z | xargs -0 sha256sum
} | tee "$HOME/lara-r2-bootstrap/weights.sha256"
```

If the checkpoint filename differs, inspect the downloaded repository before
setting `CKPT_NAME`; do not rename or guess it:

```bash
find "$OFFICIAL_RUN" -maxdepth 2 -type f -printf '%P %s bytes\n' | sort
```

## 3. Clone the pinned source trees

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

Record the LaRA commit printed here. The LIBERO SHA must be exactly
`8f1084e3132a39270c3a13ebe37270a43ece2a01`.

## 4. Install Miniconda if needed

Skip the download when the server already has Conda. Otherwise:

```bash
cd "$REPRO_ROOT"
curl -fL -o Miniconda3-latest-Linux-x86_64.sh \
  https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p "$CONDA_BASE"
```

Initialize the current shell and define interpreter paths:

```bash
source "$CONDA_BASE/etc/profile.d/conda.sh"
export LARAVLA_PYTHON="$CONDA_BASE/envs/lara-vla/bin/python"
export LIBERO_PYTHON="$CONDA_BASE/envs/libero/bin/python"
```

## 5. Build the LaRA policy-server environment

```bash
conda create -y -n lara-vla python=3.10.22 pip
"$LARAVLA_PYTHON" -m pip install --upgrade \
  pip==26.2.1 setuptools==75.8.0 wheel==0.47.0 ninja==1.13.2

"$LARAVLA_PYTHON" -m pip install \
  torch==2.6.0 torchvision==0.21.0 \
  --index-url https://download.pytorch.org/whl/cu124

"$LARAVLA_PYTHON" -m pip install \
  -r "$REPO/environment/lara-requirements.txt"
```

Verify that torch and `nvcc` both report CUDA 12.4, then derive the compilation
architectures from the actual server GPUs. This avoids assuming H100, A100, or
RTX 3090:

```bash
"$LARAVLA_PYTHON" - <<'PY'
import torch
print("torch", torch.__version__, "torch CUDA", torch.version.cuda)
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
    raise SystemExit("No CUDA GPU visible")
print(";".join(caps))
PY
)"
echo "TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"
```

Build the pinned official PyTorch3D source and repaired decord wheel:

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

Do not install upstream `pipablepytorch3d` or `eva-decord`. The former embeds a
CPython 3.11 extension in its nominally universal wheel, and the latter
overwrites 94 files owned by `decord`. These were the two R0 dependency root
causes, not optional cleanup.

## 6. Build the separate LIBERO client environment

```bash
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

CPU torch in the client is intentional: the neural policy lives in the LaRA
server process, while MuJoCo renders through EGL. Torch 2.5.1 also preserves
loading of LIBERO's trusted init-state files without patching the benchmark for
the torch 2.6 `weights_only` default change.

## 7. Generate the machine-local LIBERO configuration

R2 does not consume demonstration HDF5 data, but LIBERO requires the datasets
entry to name an existing directory. Create an empty placeholder rather than
downloading training data:

```bash
mkdir -p "$REPRO_ROOT/eval-only-libero-datasets"
"$LARAVLA_PYTHON" "$REPO/scripts/reproduction/configure_libero.py" \
  --libero-home "$LIBERO_HOME" \
  --datasets "$REPRO_ROOT/eval-only-libero-datasets"
cat "$LIBERO_HOME/libero/config.yaml"
git -C "$LIBERO_HOME" status --short
```

Only `libero/config.yaml` should be untracked in the LIBERO checkout.

## 8. Prepare a path-only local view of the released checkpoint

The active config must reference the absolute local backbone path. Keep the
downloaded run untouched:

```bash
export PREPARED_RUN="$(dirname "$OFFICIAL_RUN")/repro_r2_official"
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

The two files must have identical SHA256 and inode numbers. The hard link is
required because the official loader resolves the checkpoint path before
looking for sibling `config.yaml`; a symbolic link silently jumps back to the
unmodified config and tries to load the remote `StarVLA/...` name. Keep
`PREPARED_RUN` on the same filesystem as `OFFICIAL_RUN` so the hard link works.

## 9. Evaluation-only environment acceptance

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

`server-eval` intentionally skips only training-dataset video decoding. It
still imports the real training/model/dataset modules, exercises CUDA BF16 and
the native PyTorch3D extension, and loads the local Qwen processor. The client
loads all 40 task initial-state sets and performs a real EGL reset/render.

If Numba reports “no locator available” while importing robosuite, ensure
`NUMBA_CACHE_DIR` points to a writable directory as above. Do not edit
robosuite. Warnings about Gym maintenance, missing optional robosuite private
macros, and matplotlib/pyparsing deprecations are expected.

## 10. Repeat R1 once on one server GPU

Run this in one shell; it starts the official policy server, waits for readiness,
runs Goal task 0 once, and then stops the server:

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
grep -F 'Success: True' "$LARA_R1_LOG_ROOT/client/libero_goal.log"
grep -c 'Completed 4 reasoning passes' "$LARA_R1_LOG_ROOT/server.log"
nvidia-smi
```

Acceptance: strict server load, real client/server communication, one completed
episode, `Success: True`, 16 inference calls for the observed R1 trajectory,
four latent passes per call, no OOM, and released GPU memory after shutdown.
Success itself may vary with hardware nondeterminism; completing an actual
episode without pipeline errors is the smoke-test requirement.

The robosuite EGL destructor may print `EGL_NOT_INITIALIZED` after an already
completed episode and still exit 0. Record it; do not suppress it. Also record
that the unchanged evaluator constructs an eight-value robot state but does not
send it to the policy, and uses `min/max` rather than `q01/q99` for action
unnormalization. These are protocol-consistency risks for interpreting R2, not
reasons to alter the model during setup.

## 11. Run the formal four-GPU R2 evaluation

Use `tmux` or the cluster scheduler so a terminal disconnect cannot kill the
2,000-rollout job. Confirm the selected four GPUs are free before starting:

```bash
nvidia-smi
tmux new -s lara-r2
```

Inside tmux:

```bash
source "$CONDA_BASE/etc/profile.d/conda.sh"
export LARAVLA_PYTHON="$CONDA_BASE/envs/lara-vla/bin/python"
export LIBERO_PYTHON="$CONDA_BASE/envs/libero/bin/python"
export LIBERO_HOME="$REPRO_ROOT/LIBERO"
export LARA_CHECKPOINT="$PREPARED_CKPT"
export R2_RUN_ROOT="$REPRO_ROOT/runs/r2-official-$(date +%Y%m%d-%H%M%S)"

# Use the four GPU IDs assigned by the scheduler. For an unscheduled dedicated
# eight-GPU host, the official first run uses 0,1,2,3.
export CUDA_VISIBLE_DEVICES=0,1,2,3

cd "$REPO"
bash scripts/reproduction/run_r2_official_eval.sh
```

The wrapper fixes the protocol to four suites, ten tasks per suite, and 50
rollouts per task; disables videos; uses one policy server per GPU; records GPU
telemetry, commits and checkpoint SHA256; calls the unchanged official parallel
evaluator; and refuses to write a summary unless all 2,000 outcomes are present.
The fork changes the released launcher argument `--args.log_path` to Tyro's
actual `--args.log-path`; without this minimal fix the parallel clients may
reject the output-path argument before evaluation. It also removes the R1
wrapper's machine-specific `/home/robot/miniconda3` assumption by honoring
`CONDA_BASE`, `LARAVLA_PYTHON`, and `LIBERO_PYTHON`.

## 12. Verify and preserve results

```bash
cat "$R2_RUN_ROOT/run.env"
cat "$R2_RUN_ROOT/official_checkpoint_libero_eval.csv"
"$LARAVLA_PYTHON" - <<'PY'
import json, os
p = os.path.join(os.environ["R2_RUN_ROOT"], "official_checkpoint_libero_eval.json")
d = json.load(open(p))
assert d["total_episodes"] == 2000
assert set(d["suites"]) == {"libero_spatial", "libero_goal", "libero_object", "libero_10"}
print(json.dumps({k: v["success_percent"] for k, v in d["suites"].items()}, indent=2))
print("macro average", d["macro_average_success_percent"])
PY

cp "$R2_RUN_ROOT/official_checkpoint_libero_eval.json" \
  "$REPO/results/official_checkpoint_libero_eval.json"
cp "$R2_RUN_ROOT/official_checkpoint_libero_eval.csv" \
  "$REPO/results/official_checkpoint_libero_eval.csv"
```

Do not commit raw rollout logs, caches, videos, checkpoint files, or GPU traces.
Review the two small result summaries and the recorded revisions before making
the R2 results commit.

For comparison only after all protocol checks pass, the paper reports Spatial
96.4, Goal 98.6, Object 99.8, Long (`libero_10`) 96.6, and macro average 97.9.
Investigate version, reset, preprocessing, normalization, horizon, and success
criterion differences before considering any model change.
