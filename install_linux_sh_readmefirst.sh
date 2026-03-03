#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(
  cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1
  pwd -P
)"
cd "$SCRIPT_DIR"

TARGET_TORCH="2.9.1"
TARGET_TORCHVISION="0.24.1"
TARGET_TORCHAUDIO="2.9.1"
TARGET_CUDA="12.8"
TARGET_TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"
TARGET_MAX_JOBS="${MAX_JOBS:-8}"

if ! command -v uv >/dev/null 2>&1; then
  echo "[ERR] uv not found. Install uv first: https://docs.astral.sh/uv/"
  exit 1
fi

run_py() {
  uv run python "$@"
}

install_system_packages_if_requested() {
  if ! command -v apt-get >/dev/null 2>&1; then
    echo "[INFO] apt-get not found. Skipping Linux system package installation step."
    return 0
  fi

  local sudo_cmd=""
  if [[ "$(id -u)" -ne 0 ]]; then
    if command -v sudo >/dev/null 2>&1; then
      sudo_cmd="sudo"
    else
      echo "[WARN] Running as non-root and sudo is not available. Skipping apt install step."
      return 0
    fi
  fi

  read -r -p "Install required Linux build/runtime packages via apt? [Y/n]: " APT_CHOICE
  case "${APT_CHOICE}" in
    n|N|no|NO)
      echo "[INFO] Skipping apt package installation."
      return 0
      ;;
  esac

  echo "[STEP] Installing Linux packages required for Forward-Warp build and runtime..."
  ${sudo_cmd} apt-get update
  ${sudo_cmd} apt-get install -y \
    software-properties-common \
    git \
    git-lfs \
    ffmpeg \
    mkvtoolnix \
    build-essential \
    cmake \
    ninja-build \
    pkg-config \
    libgl1 \
    libglib2.0-0
}

echo "[INFO] Syncing project dependencies (no forced torch reinstall)..."
uv sync --inexact

version_lt() {
  local a="$1"
  local b="$2"
  [[ -z "$a" ]] && return 0
  [[ -z "$b" ]] && return 1
  [[ "$(printf '%s\n' "$a" "$b" | sort -V | head -n1)" == "$a" && "$a" != "$b" ]]
}

version_gt() {
  local a="$1"
  local b="$2"
  [[ -z "$a" ]] && return 1
  [[ -z "$b" ]] && return 0
  [[ "$(printf '%s\n' "$a" "$b" | sort -V | tail -n1)" == "$a" && "$a" != "$b" ]]
}

install_recommended_torch() {
  echo "[STEP] Installing recommended torch stack (cu128)..."
  run_py -m pip install \
    --index-url https://download.pytorch.org/whl/cu128 \
    --extra-index-url https://pypi.org/simple \
    "torch==${TARGET_TORCH}" \
    "torchvision==${TARGET_TORCHVISION}" \
    "torchaudio==${TARGET_TORCHAUDIO}"
}

TORCH_LIB=""
export_torch_lib_path() {
  TORCH_LIB="$(
    run_py - <<'PY'
import os
try:
    import torch
    print(os.path.join(os.path.dirname(torch.__file__), "lib"))
except Exception:
    print("")
PY
  )"

  if [[ -n "${TORCH_LIB}" && -d "${TORCH_LIB}" ]]; then
    export LD_LIBRARY_PATH="${TORCH_LIB}:${LD_LIBRARY_PATH:-}"
    echo "[INFO] Exported LD_LIBRARY_PATH for current shell:"
    echo "       ${TORCH_LIB}"
  else
    echo "[WARN] Could not resolve torch lib path for LD_LIBRARY_PATH export."
  fi
}

build_forward_warp_if_requested() {
  local cuda_dir="dependency/Forward-Warp/Forward_Warp/cuda"
  local fw_root="dependency/Forward-Warp"
  if [[ ! -d "${cuda_dir}" || ! -d "${fw_root}" ]]; then
    echo "[WARN] Forward-Warp sources not found in dependency/. Skipping build."
    return 0
  fi

  read -r -p "Build Forward-Warp CUDA extension now? [Y/n]: " FW_CHOICE
  case "${FW_CHOICE}" in
    n|N|no|NO)
      echo "[INFO] Skipping Forward-Warp build."
      return 0
      ;;
  esac

  export TORCH_CUDA_ARCH_LIST="${TARGET_TORCH_CUDA_ARCH_LIST}"
  export MAX_JOBS="${TARGET_MAX_JOBS}"
  echo "[STEP] Building Forward-Warp with TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}, MAX_JOBS=${MAX_JOBS}..."

  (
    cd "${cuda_dir}"
    run_py -m pip install -v --no-build-isolation .
  )
  (
    cd "${fw_root}"
    run_py -m pip install -v --no-build-isolation --no-cache-dir .
  )

  local fw_so
  fw_so="$(
    run_py - <<'PY'
try:
    import forward_warp_cuda as m
    print(m.__file__)
except Exception:
    print("")
PY
  )"

  if [[ -n "${fw_so}" ]]; then
    echo "[OK] Forward-Warp built and importable."
    echo "[INFO] forward_warp_cuda .so path:"
    echo "       ${fw_so}"
  else
    echo "[WARN] Forward-Warp build completed but import check failed in current shell."
    echo "[WARN] Reopen terminal, reactivate your environment, and retry import."
  fi
}

install_system_packages_if_requested

TORCH_INFO="$(
  run_py - <<'PY'
import importlib.metadata as md
import re
import shutil
import subprocess

def pkg_ver(name: str) -> str:
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return ""

def base_ver(v: str) -> str:
    return v.split("+", 1)[0] if v else ""

torch_v = pkg_ver("torch")
torchvision_v = pkg_ver("torchvision")
torchaudio_v = pkg_ver("torchaudio")
torch_cuda = ""

if torch_v:
    try:
        import torch
        torch_cuda = torch.version.cuda or ""
    except Exception:
        torch_cuda = ""

nvcc_release = ""
if shutil.which("nvcc"):
    try:
        out = subprocess.check_output(["nvcc", "--version"], text=True, stderr=subprocess.STDOUT)
        m = re.search(r"release\s+([0-9]+\.[0-9]+)", out)
        if m:
            nvcc_release = m.group(1)
    except Exception:
        nvcc_release = ""

print(f"HAS_TORCH={'1' if torch_v else '0'}")
print(f"TORCH_VER={torch_v}")
print(f"TORCH_BASE={base_ver(torch_v)}")
print(f"TORCHVISION_VER={torchvision_v}")
print(f"TORCHVISION_BASE={base_ver(torchvision_v)}")
print(f"TORCHAUDIO_VER={torchaudio_v}")
print(f"TORCHAUDIO_BASE={base_ver(torchaudio_v)}")
print(f"TORCH_CUDA={torch_cuda}")
print(f"NVCC_RELEASE={nvcc_release}")
PY
)"

HAS_TORCH="0"
TORCH_VER=""
TORCH_BASE=""
TORCHVISION_VER=""
TORCHVISION_BASE=""
TORCHAUDIO_VER=""
TORCHAUDIO_BASE=""
TORCH_CUDA=""
NVCC_RELEASE=""

while IFS='=' read -r k v; do
  case "$k" in
    HAS_TORCH) HAS_TORCH="$v" ;;
    TORCH_VER) TORCH_VER="$v" ;;
    TORCH_BASE) TORCH_BASE="$v" ;;
    TORCHVISION_VER) TORCHVISION_VER="$v" ;;
    TORCHVISION_BASE) TORCHVISION_BASE="$v" ;;
    TORCHAUDIO_VER) TORCHAUDIO_VER="$v" ;;
    TORCHAUDIO_BASE) TORCHAUDIO_BASE="$v" ;;
    TORCH_CUDA) TORCH_CUDA="$v" ;;
    NVCC_RELEASE) NVCC_RELEASE="$v" ;;
  esac
done <<< "$TORCH_INFO"

echo
echo "[INFO] Current runtime detected in project env:"
echo "       torch=${TORCH_VER:-n.d.}"
echo "       torchvision=${TORCHVISION_VER:-n.d.}"
echo "       torchaudio=${TORCHAUDIO_VER:-n.d.}"
echo "       torch cuda=${TORCH_CUDA:-n.d.}"
echo "       nvcc release=${NVCC_RELEASE:-n.d.}"
echo "       target torch=${TARGET_TORCH} torchvision=${TARGET_TORCHVISION} torchaudio=${TARGET_TORCHAUDIO} cuda=${TARGET_CUDA}"
echo

if [[ "$HAS_TORCH" != "1" ]]; then
  echo "[INFO] Torch stack not found. Installing recommended stack."
  install_recommended_torch
else
  aligned="0"
  if [[ "$TORCH_BASE" == "$TARGET_TORCH" && "$TORCHVISION_BASE" == "$TARGET_TORCHVISION" && "$TORCHAUDIO_BASE" == "$TARGET_TORCHAUDIO" && "$TORCH_CUDA" == "$TARGET_CUDA" ]]; then
    aligned="1"
  fi

  if [[ "$aligned" == "1" ]]; then
    echo "[OK] Torch stack already aligned to recommended versions."
  else
    older_stack="0"
    newer_stack="0"

    if version_lt "$TORCH_BASE" "$TARGET_TORCH" || version_lt "$TORCHVISION_BASE" "$TARGET_TORCHVISION" || version_lt "$TORCHAUDIO_BASE" "$TARGET_TORCHAUDIO" || version_lt "$TORCH_CUDA" "$TARGET_CUDA"; then
      older_stack="1"
    fi
    if version_gt "$TORCH_BASE" "$TARGET_TORCH" || version_gt "$TORCHVISION_BASE" "$TARGET_TORCHVISION" || version_gt "$TORCHAUDIO_BASE" "$TARGET_TORCHAUDIO" || version_gt "$TORCH_CUDA" "$TARGET_CUDA"; then
      newer_stack="1"
    fi

    if [[ "$TORCH_CUDA" == "11.8" || "$older_stack" == "1" ]]; then
      echo "[WARN] Older torch/cuda stack detected. CUDA 11.8 is usually much slower for this fork."
      echo "[WARN] Recommended target is torch ${TARGET_TORCH} + cu${TARGET_CUDA}."
    elif [[ "$newer_stack" == "1" ]]; then
      echo "[INFO] Newer torch/cuda stack detected."
    else
      echo "[INFO] Different torch/cuda stack detected."
    fi

    read -r -p "Align to recommended torch/cu128 stack now? [y/N]: " ALIGN_CHOICE
    case "${ALIGN_CHOICE}" in
      y|Y|yes|YES)
        install_recommended_torch
        ;;
      *)
        echo "[INFO] Keeping current torch stack."
        ;;
    esac
  fi
fi

export_torch_lib_path
build_forward_warp_if_requested

echo
echo "[DONE] Linux setup completed."
echo "[NEXT] Close and reopen your terminal, then reactivate your preferred Python environment."
if [[ -n "${TORCH_LIB}" && -d "${TORCH_LIB}" ]]; then
  echo "[INFO] If needed, persist this line in your shell rc (~/.bashrc or ~/.zshrc):"
  echo "       export LD_LIBRARY_PATH=\"${TORCH_LIB}:\${LD_LIBRARY_PATH:-}\""
fi
echo "[NEXT] After reopening terminal and reactivating env, launch pipeline:"
echo "       python pipeline_master_gui.py"
