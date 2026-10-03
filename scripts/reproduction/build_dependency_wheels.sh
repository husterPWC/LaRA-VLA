#!/usr/bin/env bash
# Rebuild the upstream numerical/video dependencies for this Python ABI.
# Prerequisites: git, unzip, CUDA toolkit matching torch, and torch already
# installed in LARAVLA_PYTHON. decord code is not rebuilt; its published wheel
# is repacked only to make its internal ABI tag match its ABI-neutral filename.
set -euo pipefail
: "${LARAVLA_PYTHON:?Set LARAVLA_PYTHON to the lara-vla Python executable}"
: "${CUDA_HOME:?Set CUDA_HOME to the CUDA toolkit matching torch}"
: "${TORCH_CUDA_ARCH_LIST:?Set CUDA architectures for the target machine; 3090 is 8.6}"
BUILD_ROOT="${BUILD_ROOT:-/tmp/lara-dependency-build}"
WHEEL_DIR="${WHEEL_DIR:-${BUILD_ROOT}/wheels}"
mkdir -p "${BUILD_ROOT}" "${WHEEL_DIR}"
checkout() {
  local name="$1" url="$2" revision="$3"
  if [[ ! -d "${BUILD_ROOT}/${name}/.git" ]]; then
    git clone "${url}" "${BUILD_ROOT}/${name}"
  fi
  if [[ -n "$(git -C "${BUILD_ROOT}/${name}" status --porcelain --untracked-files=no)" ]]; then
    echo "Refusing to overwrite modified dependency source: ${name}" >&2
    exit 1
  fi
  git -C "${BUILD_ROOT}/${name}" checkout --detach "${revision}"
  test "$(git -C "${BUILD_ROOT}/${name}" rev-parse HEAD)" = "${revision}"
}
checkout pytorch3d https://github.com/facebookresearch/pytorch3d.git f34104cf6ebefacd7b7e07955ee7aaa823e616ac
MAX_JOBS="${MAX_JOBS:-4}" FORCE_CUDA=1 \
  "${LARAVLA_PYTHON}" -m pip wheel --no-deps --no-build-isolation \
  "${BUILD_ROOT}/pytorch3d" --wheel-dir "${WHEEL_DIR}"
"${LARAVLA_PYTHON}" -m pip download --no-deps decord==0.6.0 \
  --dest "${BUILD_ROOT}"
DECORD_WHEEL="${BUILD_ROOT}/decord-0.6.0-py3-none-manylinux2010_x86_64.whl"
test "$(sha256sum "${DECORD_WHEEL}" | cut -d' ' -f1)" = \
  "51997f20be8958e23b7c4061ba45d0efcd86bffd5fe81c695d0befee0d442976"
REPACK_DIR="$(mktemp -d "${BUILD_ROOT}/decord-repack.XXXXXX")"
trap 'rm -rf "${REPACK_DIR}"' EXIT
unzip -q "${DECORD_WHEEL}" -d "${REPACK_DIR}"
sed -i \
  's/Tag: cp36-cp36m-manylinux2010_x86_64/Tag: py3-none-manylinux2010_x86_64/' \
  "${REPACK_DIR}/decord-0.6.0.dist-info/WHEEL"
rg -Fx 'Tag: py3-none-manylinux2010_x86_64' \
  "${REPACK_DIR}/decord-0.6.0.dist-info/WHEEL"
"${LARAVLA_PYTHON}" -m wheel pack "${REPACK_DIR}" --dest-dir "${WHEEL_DIR}"
echo "Built wheels: ${WHEEL_DIR}. Install these before environment acceptance."
