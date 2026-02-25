#!/usr/bin/env bash
set -euo pipefail

# Stage 1:
# - System deps
# - Clone/update your fork
# - Install Python deps WITHOUT torch stack
# - Build/install Forward-Warp CUDA extension

REPO_URL="${REPO_URL:-https://github.com/tig3rmast3r/StereoCrafter.git}"
REPO_DIR="${REPO_DIR:-/workspace/StereoCrafter}"
VENV_PATH="${VENV_PATH:-/venv/main}"
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"
MAX_JOBS="${MAX_JOBS:-8}"
REQ_FILE_REL="requirements.docker.no_torch.txt"

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
${SUDO} apt-get install -y git git-lfs ffmpeg build-essential cmake ninja-build pkg-config libgl1 libglib2.0-0

echo "[INFO] Cloning/updating repository..."
if [[ ! -d "${REPO_DIR}/.git" ]]; then
  git clone --recursive "${REPO_URL}" "${REPO_DIR}"
else
  git -C "${REPO_DIR}" remote set-url origin "${REPO_URL}" || true
  git -C "${REPO_DIR}" submodule update --init --recursive
fi

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

cd "${REPO_DIR}"
git lfs install
mkdir -p work

echo "[DONE] Stage 1 complete."
echo "[NEXT] Run: ./setup_vast_stage2_weights.sh"
