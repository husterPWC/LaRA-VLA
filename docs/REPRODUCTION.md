# LaRA-VLA official reproduction

## Scope and baseline

Rebuilt on 2026-10-03 after the old checkout and environments were removed.
R0 is environment validation; R1 is an actual released-checkpoint rollout.
R2 is official 2,000-rollout evaluation on the server. R3/R4 training remains
blocked until checkpoint evaluation is validated. No Spatial-LaRA changes.

- Fork: https://github.com/husterPWC/LaRA-VLA
- Upstream: https://github.com/LoveJu1y/LaRA-VLA.git
- Initial fork, origin/main and upstream/main: `93b5c03c691e38a3c9e90878e76557009e2df261`
- LIBERO: https://github.com/Lifelong-Robot-Learning/LIBERO.git
- LIBERO commit: `8f1084e3132a39270c3a13ebe37270a43ece2a01`
- Initial tracked worktrees were clean. No merge, rebase or reset was performed.
- Ubuntu 22.04.5; two RTX 3090 24 GiB GPUs; R0/R1 use GPU 0 only.
- Driver 570.144; system CUDA toolkit 12.4 (nvcc V12.4.99).
  The 12.8 label in nvidia-smi is driver capability, not the torch runtime.

## Environment decisions

Two newly created Conda environments: `lara-vla` and `libero`, Python 3.10.
This follows LaRA's LIBERO README recommendation to separate client/server.
Unrelated Conda environments were not changed.

- Server: torch 2.6.0+cu124 / torchvision 0.21.0+cu124, paired with the
  upstream torchvision pin; transformers 4.57.0, accelerate 1.5.2,
  DeepSpeed 0.16.9, NumPy 1.26.4 as upstream requires.
- Client: torch 2.5.1+cpu / torchvision 0.20.1+cpu. The client has no neural
  policy; GPU policy runs in the server. This explicit compatibility choice
  preserves loading of LIBERO's trusted initial-state files without the
  `weights_only=False` source patches needed with torch 2.6 defaults.
  CPU torch does not prevent GPU EGL rendering.
- Client NumPy 1.24.4 follows LaRA's evaluation README, overriding LIBERO's
  original 1.22.4 pin. Other original LIBERO direct pins are preserved.
- MuJoCo 2.3.7 is an explicit reproduction choice with robosuite 1.4.0,
  not a claim that the paper specified this exact patch version.
- Client supplemental packages are recorded in `environment/libero-requirements.txt`.
  Numba 0.60.0 is pinned for the NumPy 1.24 stack. Setuptools 75.8.0 retains
  legacy packaging APIs used by the older robotics dependencies.
- No model, loss, curriculum, evaluation action semantics, or LIBERO benchmark
  source has been changed by this rebuild.

## Local resources and paths

Workspace: `/home/robot/codePWC/LaRA`; code: `LaRA-VLA/`; simulator: `LIBERO/`.
LaRA training data: `/data/CodePWC/lara_datasets/libero_lerobot_all`.
Existing data and weights were retained; no dataset or model was downloaded.

```
../StarVLA-Qwen3-VL-4B-Instruct-Action/
../checkpoints/LaRA-VLA-libero/config.yaml
../checkpoints/LaRA-VLA-libero/config.json
../checkpoints/LaRA-VLA-libero/dataset_statistics.json
../checkpoints/LaRA-VLA-libero/checkpoints/steps_25000_pytorch_model.pt
```

`environment/local-weights.sha256` identifies existing local weight bytes;
it does not prove equality with a verified Hugging Face revision.
Checkpoint remote revision is pending verification. Its filename says 25k,
while accompanying configuration says max 40k and summary lists 16k–40k;
do not infer provenance or final training step from either alone.

Configure the simulator paths (JSON syntax is valid YAML):

```bash
python scripts/reproduction/configure_libero.py \
  --libero-home ../LIBERO \
  --datasets /data/CodePWC/lara_datasets/clip-rt/modified_libero_hdf5
```

This generates `../LIBERO/libero/config.yaml`, an untracked machine-local file.
The `datasets` entry is the HDF5 root, not LaRA's LeRobot training root;
R0 reset/render and R1 evaluation do not consume demonstration HDF5 files.
Set `LIBERO_CONFIG_PATH=$LIBERO_HOME/libero` when running the official client.

The previous fork's `LARAVLA_VLM_PATH` patch is absent from this clean fork.
R1 must explicitly configure the existing backbone through the official
`framework.qwenvl.base_vlm` mechanism; do not assume that old variable works.
Do not overwrite released configuration without preserving its original and
recording the path-only change. Prepare an isolated run view with:

```bash
python scripts/reproduction/prepare_official_checkpoint.py \
  --source-run ../checkpoints/LaRA-VLA-libero \
  --backbone ../StarVLA-Qwen3-VL-4B-Instruct-Action \
  --output-run ../checkpoints/repro_r1_official
```

The generated manifest states that only `framework.qwenvl.base_vlm` changed;
the active checkpoint is an absolute symlink to the released bytes.

## R0 acceptance

```bash
bash scripts/reproduction/env_check.sh
```

The script checks both environments with pip check, imports the actual training
entry, model, dataset loader and Qwen3 class, runs BF16 CUDA arithmetic, and
loads the local processor. It then loads initial states for all 40 LIBERO tasks
and resets/renders the real Goal task 0 using both 256x256 cameras through EGL.
It does not replace a policy, perform a rollout, or instantiate a small model.
R0 passing cannot be used as evidence of checkpoint loading or R1 success.

Status: **PASS on 2026-10-03**. Both pip checks, actual imports, GPU/native
extension probes, real AV1 decode, all task initial states, and EGL rendering
completed successfully. This closes environment R0 only; checkpoint loading and
an actual rollout remain R1 work.

### Dependency packaging root causes (2026-10-03)

The initial installation used the unmodified upstream requirements with the
server constraints. pip 26.2.1 check found two real packaging defects:

1. `pipablepytorch3d==0.7.6` advertises a universal wheel but embeds a
   `cp311-cp311-linux_x86_64` WHEEL tag and `_C.cpython-311-...so`. Python 3.10
   imports its pure Python transforms but cannot use that extension.
2. `decord==0.6.0` plus `eva-decord==0.6.1` share 94 installed RECORD paths,
   including decord's dist-info files. The published decord filename is tagged
   `py3-none-manylinux2010`, while its internal WHEEL metadata incorrectly says
   `cp36-cp36m-manylinux2010`. Import alone is not sufficient validation.

No pip downgrade or suppressed dependency-check failure is used. The proposed
repair is a separately recorded dependency build, leaving upstream
requirements.txt and all LaRA source unchanged:

- Build official PyTorch3D v0.7.6 at
  `f34104cf6ebefacd7b7e07955ee7aaa823e616ac` against torch 2.6/cu124 for Python
  3.10 and CUDA architecture 8.6. All six transforms Python files were AST
  identical to the original installed package (only comments differed).
  Build succeeded; the incompatible package was replaced by the official wheel.
- Use a single decord provider, removing the overlapping eva-decord
  distribution. The official decord 0.6.0 wheel bytes are unpacked, only the
  internal tag is changed to match its published ABI-neutral filename, and
  RECORD is regenerated. The original wheel SHA256 is
  `51997f20be8958e23b7c4061ba45d0efcd86bffd5fe81c695d0befee0d442976`;
  the repaired wheel SHA256 in this run is
  `991dd62b390a2393ad0e531f2dd8c6bc20dd26dc89ad85cff363b0de0323952c`.
  Its bundled 2021 FFmpeg cannot decode the current AV1 dataset, so no such
  capability is claimed. The unchanged official loader selects
  `torchvision_av`; that path decoded three real 256x256 AV1 frames correctly.

The build recipe is `scripts/reproduction/build_dependency_wheels.sh`;
`environment/lara-requirements.txt` is the upstream list with these three
wheel entries removed and explicit communication/build dependencies added.
Use the built wheels alongside that file, not that file alone. Native wheels
are machine-specific artifacts, excluded from Git; rebuild on the 8-GPU
server with its actual CUDA architectures rather than assuming H100.

Checks already completed: server/model/Qwen3/dataset imports, local processor,
GPU 0 BF16 matrix arithmetic, all 40 tasks' initial-state loads and real EGL
reset/render of both cameras. Initial server log reports cuDNN 90100. Local
backbone tokenizer lacks thinking/img_next tokens before model initialization;
upstream QWen3 adds those in its initializer, to be validated with R1 loading.
Client completed without rendering cleanup errors. Expected upstream warnings
include Gym maintenance notice, missing optional robosuite private macros,
matplotlib pyparsing deprecations, and the trusted-init-state torch.load future
warning. None were hidden or used to replace a failed check.

## Upstream issues retained for later stages

These were identified at the baseline above; rebuilding the environments does
not fix them. Investigate and document separately, with minimal independent
commits when needed, rather than silently changing the method.

- Official evaluator loops all tasks; one rollout/task is ten episodes, not a
  single task. R1 needs a task-selector parameter retaining default behavior.
- Evaluator constructs state but does not send it to the policy. Preserve the
  official behavior until its role is established.
- Checkpoint loader contains optional-module strict=False compatibility logic;
  record and explain any missing/unexpected keys during R1.
- README's LIBERO_LEROBOT_ROOT variable is not consumed by the training path;
  use existing `--datasets.vla_data.data_root_dir` override.
- Multistage launcher covers 5k/2k/2k/2k reasoning training only. Stage III is a
  separate script defaulting to 60k/one process and an empty pretrained path.
- Multistage `img_next.use_teacher=false` gates off visual-supervision loss in
  current forward. The paper describes visual supervision in Stage I/II.
- Stage III YAML defaults retain discrete action supervision and VLM weight 1;
  released checkpoint config disables action tokens and sets VLM weight 0.
- Gradient accumulation and gradient-checkpointing YAML fields are not visibly
  wired through; num_workers is hardcoded. Verify before memory tuning.
- Saved model state dict is not a complete optimizer/scheduler/RNG resume.
- Training actually uses torchrun + Accelerate + DeepSpeedPlugin. Validate
  rank/device placement, sharding, scheduler stepping and full resume before R4.

## Change policy

Each source repair requires root cause, reason, minimal diff and independent
commit. Environment artifacts and R0 checks add no alternate model forward.
Ignore datasets, caches, weights and large logs; commit source, configurations,
version locks, small verification logs and result summaries only.
