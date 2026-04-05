#!/usr/bin/env bash
set -euo pipefail

# Stage 1:
# - System deps
# - Use current cloned repo
# - Install Python deps WITHOUT torch stack
# - Optional Forward-Warp CUDA extension (disabled by default)

SCRIPT_DIR="$(
  cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1
  pwd -P
)"
REPO_DIR="${REPO_DIR:-${SCRIPT_DIR}}"
VENV_PATH="${VENV_PATH:-/venv/main}"
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"
MAX_JOBS="${MAX_JOBS:-8}"
REQ_FILE_REL="requirements.docker.no_torch.txt"
BUILD_FORWARD_WARP=false

if [[ -x "${VENV_PATH}/bin/activate" ]]; then
  # shellcheck disable=SC1090
  source "${VENV_PATH}/bin/activate"
else
  echo "[WARN] Venv not found at ${VENV_PATH}. Continuing with current Python."
fi

if [[ "$(id -u)" -eq 0 ]]; then
  SUDO=""
else
  SUDO="sudo"
fi

echo "[INFO] Installing system packages..."
export DEBIAN_FRONTEND=noninteractive
${SUDO} apt-get update
${SUDO} apt-get install -y software-properties-common
runtime_packages=(git git-lfs ffmpeg libgl1 libglib2.0-0)
build_packages=()
if [[ "${BUILD_FORWARD_WARP}" == "true" ]]; then
  build_packages=(build-essential cmake ninja-build pkg-config)
fi
${SUDO} apt-get install -y "${runtime_packages[@]}"
if (( ${#build_packages[@]} > 0 )); then
  echo "[INFO] Installing optional Forward-Warp build packages..."
  ${SUDO} apt-get install -y "${build_packages[@]}"
fi

echo "[INFO] Using repository at: ${REPO_DIR}"
if [[ ! -d "${REPO_DIR}/.git" ]]; then
  echo "[ERR] No git repo found at ${REPO_DIR}"
  echo "[ERR] Clone first, then run this script from inside StereoCrafter."
  echo "[ERR] Example:"
  echo "      cd /workspace"
  echo "      git clone --recursive https://github.com/tig3rmast3r/StereoCrafter"
  echo "      cd StereoCrafter && ./setup_vast_stage1.sh"
  exit 2
fi

git -C "${REPO_DIR}" submodule update --init --recursive

cd "${REPO_DIR}"
python -m pip install -U pip setuptools wheel

if [[ ! -f "${REQ_FILE_REL}" ]]; then
  echo "[ERR] Missing ${REQ_FILE_REL} in ${REPO_DIR}"
  exit 2
fi

echo "[INFO] Validating preinstalled torch stack (must already be in the base Docker image)..."
python - <<'PY'
import importlib.metadata as md
import sys

required = ("torch", "torchvision", "torchaudio")
missing = []
for name in required:
    try:
        print(f"[OK] {name}=={md.version(name)}")
    except Exception:
        missing.append(name)

if missing:
    print(f"[ERR] Missing preinstalled packages: {', '.join(missing)}")
    print("[ERR] Preinstall torch/torchvision/torchaudio in the Docker image, then rerun stage1.")
    sys.exit(2)
PY

python - <<'PY' > /tmp/torch_stack_constraints.txt
import importlib.metadata as md
for name in ("torch", "torchvision", "torchaudio", "xformers"):
    try:
        print(f"{name}=={md.version(name)}")
    except Exception:
        pass
PY

echo "[INFO] Installing Python dependencies (no torch reinstall)..."
python -m pip install -r "${REQ_FILE_REL}" -c /tmp/torch_stack_constraints.txt

if [[ "${BUILD_FORWARD_WARP}" == "true" ]]; then
  echo "[INFO] Building Forward-Warp CUDA extension..."
  export TORCH_CUDA_ARCH_LIST
  export MAX_JOBS
  export TORCH_LIB="$(
python - <<'PY'
import os
import torch
print(os.path.join(os.path.dirname(torch.__file__), "lib"))
PY
)"
  export LD_LIBRARY_PATH="${TORCH_LIB}:${LD_LIBRARY_PATH:-}"

  cd "${REPO_DIR}/dependency/Forward-Warp/Forward_Warp/cuda"
  python -m pip install -v --no-build-isolation .
  python - <<'PY'
import forward_warp_cuda as m
print("[OK] forward_warp_cuda:", m.__file__)
PY

  cd "${REPO_DIR}/dependency/Forward-Warp"
  python -m pip install -v --no-build-isolation --no-cache-dir .

  if ! grep -q '^source /venv/main/bin/activate$' "${HOME}/.bashrc" 2>/dev/null; then
    echo 'source /venv/main/bin/activate' >> "${HOME}/.bashrc"
  fi
  if ! grep -q '/venv/main/lib/python3.10/site-packages/torch/lib' "${HOME}/.bashrc" 2>/dev/null; then
    echo 'export LD_LIBRARY_PATH=/venv/main/lib/python3.10/site-packages/torch/lib:${LD_LIBRARY_PATH:-}' >> "${HOME}/.bashrc"
  fi
else
  echo "[INFO] BUILD_FORWARD_WARP=false -> skipping optional Forward-Warp CUDA build."
fi

cd "${REPO_DIR}"
git lfs install
mkdir -p work

echo "[DONE] Stage 1 complete."
echo "[NEXT] Run: ./setup_vast_stage2_weights.sh"
