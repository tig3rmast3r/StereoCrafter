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
BUILD_FORWARD_WARP=false
INSTALL_DISTRO_FFMPEG=1
TORCH_LIB=""
export BUILD_FORWARD_WARP

print_uv_install_suggestion() {
  echo "[INFO] Suggested install:"
  if command -v curl >/dev/null 2>&1; then
    echo '       curl -LsSf https://astral.sh/uv/install.sh | env UV_NO_MODIFY_PATH=1 sh'
    echo '       export PATH="$HOME/.local/bin:$PATH"'
  elif command -v wget >/dev/null 2>&1; then
    echo '       wget -qO- https://astral.sh/uv/install.sh | env UV_NO_MODIFY_PATH=1 sh'
    echo '       export PATH="$HOME/.local/bin:$PATH"'
  else
    echo '       curl -LsSf https://astral.sh/uv/install.sh | env UV_NO_MODIFY_PATH=1 sh'
    echo '       # or'
    echo '       wget -qO- https://astral.sh/uv/install.sh | env UV_NO_MODIFY_PATH=1 sh'
    echo '       export PATH="$HOME/.local/bin:$PATH"'
  fi
}

is_wsl() {
  grep -qiE '(microsoft|wsl)' /proc/version 2>/dev/null || grep -qiE '(microsoft|wsl)' /proc/sys/kernel/osrelease 2>/dev/null
}

run_py() {
  uv run python "$@"
}

resolve_realpath() {
  local path="$1"
  if command -v readlink >/dev/null 2>&1; then
    readlink -f "$path" 2>/dev/null || printf '%s\n' "$path"
  else
    printf '%s\n' "$path"
  fi
}

ffmpeg_version_line() {
  local ffmpeg_bin="$1"
  local line=""
  line="$("$ffmpeg_bin" -version 2>/dev/null | head -n 1 || true)"
  if [[ -n "$line" ]]; then
    printf '%s\n' "$line"
  else
    printf 'unknown\n'
  fi
}

ffmpeg_nvenc_status() {
  local ffmpeg_bin="$1"
  local encoders=""
  local has_h264="0"
  local has_hevc="0"
  local encoder_label=""

  encoders="$("$ffmpeg_bin" -hide_banner -encoders 2>/dev/null || true)"
  if grep -q '\<h264_nvenc\>' <<< "$encoders"; then
    has_h264="1"
  fi
  if grep -q '\<hevc_nvenc\>' <<< "$encoders"; then
    has_hevc="1"
  fi

  if [[ "$has_h264" != "1" && "$has_hevc" != "1" ]]; then
    printf 'not available\n'
    return 0
  fi

  if [[ "$has_h264" == "1" && "$has_hevc" == "1" ]]; then
    encoder_label="h264_nvenc, hevc_nvenc"
  elif [[ "$has_h264" == "1" ]]; then
    encoder_label="h264_nvenc"
  else
    encoder_label="hevc_nvenc"
  fi

  if ! command -v nvidia-smi >/dev/null 2>&1; then
    printf 'encoders present (%s); runtime not verified (nvidia-smi not found)\n' "$encoder_label"
    return 0
  fi

  local smoke_encoder="h264_nvenc"
  if [[ "$has_h264" != "1" && "$has_hevc" == "1" ]]; then
    smoke_encoder="hevc_nvenc"
  fi

  if "$ffmpeg_bin" -hide_banner -loglevel error -f lavfi -i color=c=black:s=256x144:d=0.1 -frames:v 1 -c:v "$smoke_encoder" -f null - >/dev/null 2>&1; then
    printf 'encoders present (%s); runtime ok (%s)\n' "$encoder_label" "$smoke_encoder"
  else
    printf 'encoders present (%s); runtime FAILED (%s)\n' "$encoder_label" "$smoke_encoder"
  fi
}

discover_ffmpeg_candidates() {
  FFMPEG_CANDIDATE_DISPLAYS=()
  FFMPEG_CANDIDATE_REALS=()

  local old_ifs="$IFS"
  local dir=""
  local candidate=""
  local real=""
  local seen="|"
  IFS=':'
  for dir in $PATH; do
    [[ -n "$dir" ]] || dir="."
    candidate="$dir/ffmpeg"
    if [[ -x "$candidate" && ! -d "$candidate" ]]; then
      real="$(resolve_realpath "$candidate")"
      if [[ "$seen" != *"|${real}|"* ]]; then
        seen="${seen}${real}|"
        FFMPEG_CANDIDATE_DISPLAYS+=("$candidate")
        FFMPEG_CANDIDATE_REALS+=("$real")
      fi
    fi
  done
  IFS="$old_ifs"
}

prompt_ffmpeg_policy_if_found() {
  discover_ffmpeg_candidates

  if (( ${#FFMPEG_CANDIDATE_REALS[@]} == 0 )); then
    INSTALL_DISTRO_FFMPEG=1
    echo "[INFO] No ffmpeg found in PATH."
    echo "[INFO] Distro ffmpeg will be installed via apt if you keep the apt step enabled."
    echo "[WARN] NVENC compatibility depends on the actual ffmpeg build provided by your distro/repo."
    echo "[WARN] If you specifically need NVENC and the installed binary still lacks it, you may need to install or build an NVENC-capable ffmpeg manually."
    return 0
  fi

  echo "[INFO] Found ffmpeg binaries in PATH:"
  local idx=0
  local display=""
  local real=""
  local version=""
  local nvenc=""
  local nvenc_capable_count=0
  for idx in "${!FFMPEG_CANDIDATE_REALS[@]}"; do
    display="${FFMPEG_CANDIDATE_DISPLAYS[$idx]}"
    real="${FFMPEG_CANDIDATE_REALS[$idx]}"
    version="$(ffmpeg_version_line "$display")"
    nvenc="$(ffmpeg_nvenc_status "$display")"
    echo "       [$((idx + 1))] path=${display}"
    if [[ "$real" != "$display" ]]; then
      echo "           real=${real}"
    fi
    echo "           version=${version}"
    echo "           nvenc=${nvenc}"
    if [[ "$nvenc" != "not available" ]]; then
      nvenc_capable_count=$((nvenc_capable_count + 1))
    fi
  done

  echo "[INFO] Active ffmpeg is the first match in PATH:"
  echo "       ${FFMPEG_CANDIDATE_DISPLAYS[0]}"
  echo "[INFO] If a custom ffmpeg comes before /usr/bin, 'apt install ffmpeg' may not change the binary actually used at runtime."
  if (( nvenc_capable_count == 0 )); then
    echo "[WARN] None of the ffmpeg binaries currently found in PATH report NVENC encoders."
    echo "[WARN] Installing distro ffmpeg via apt may or may not add NVENC support, depending on the distro/repo build."
    echo "[WARN] If you need NVENC and still do not get it, install or build an NVENC-capable ffmpeg manually."
  fi

  while true; do
    read -r -p "ffmpeg already found. Use current PATH ffmpeg, install distro ffmpeg anyway, or stop installer? [U/i/s]: " FFMPEG_CHOICE
    case "${FFMPEG_CHOICE:-U}" in
      u|U|use|USE)
        INSTALL_DISTRO_FFMPEG=0
        echo "[INFO] Keeping the current PATH ffmpeg."
        return 0
        ;;
      i|I|install|INSTALL)
        INSTALL_DISTRO_FFMPEG=1
        echo "[INFO] Distro ffmpeg will be installed via apt."
        echo "[INFO] This does not guarantee NVENC support or that /usr/bin/ffmpeg becomes the active binary if another ffmpeg stays earlier in PATH."
        return 0
        ;;
      s|S|stop|STOP)
        echo "[INFO] Installer stopped by user."
        exit 1
        ;;
      *)
        echo "[WARN] Invalid choice. Use 'U', 'i', or 's'."
        ;;
    esac
  done
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

  prompt_ffmpeg_policy_if_found

  local package_label="runtime"
  if [[ "$BUILD_FORWARD_WARP" == "true" ]]; then
    package_label="runtime/build"
  fi

  read -r -p "Install required Linux ${package_label} packages via apt? [Y/n]: " APT_CHOICE
  case "${APT_CHOICE}" in
    n|N|no|NO)
      echo "[INFO] Skipping apt package installation."
      return 0
      ;;
  esac

  local runtime_packages=(
    software-properties-common
    git
    git-lfs
    mkvtoolnix
    libgl1
    libglib2.0-0
  )
  local build_packages=()

  if [[ "$INSTALL_DISTRO_FFMPEG" == "1" ]]; then
    runtime_packages+=(ffmpeg)
  else
    echo "[INFO] Skipping distro ffmpeg install and keeping PATH ffmpeg."
  fi

  if [[ "$BUILD_FORWARD_WARP" == "true" ]]; then
    build_packages+=(build-essential cmake ninja-build pkg-config)
  fi

  echo "[STEP] Installing Linux packages required for the standard runtime..."
  ${sudo_cmd} apt-get update
  ${sudo_cmd} apt-get install -y "${runtime_packages[@]}"
  if (( ${#build_packages[@]} > 0 )); then
    echo "[STEP] Installing optional Forward-Warp build packages..."
    ${sudo_cmd} apt-get install -y "${build_packages[@]}"
  fi
}

if is_wsl; then
  echo "[INFO] WSL detected."
  if [[ "$BUILD_FORWARD_WARP" == "true" ]]; then
    echo "[WARN] BUILD_FORWARD_WARP=true on WSL expects a WSL-safe CUDA toolkit setup."
  else
    echo "[INFO] Standard install will not touch Linux CUDA drivers/toolkit. The optional Forward-Warp build stays disabled."
  fi
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "[ERR] uv not found."
  print_uv_install_suggestion
  exit 1
fi

echo "[INFO] Syncing project dependencies with uv (no forced torch reinstall)..."
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

build_forward_warp_if_enabled() {
  if [[ "$BUILD_FORWARD_WARP" != "true" ]]; then
    echo "[INFO] BUILD_FORWARD_WARP=false -> skipping optional Forward-Warp CUDA build."
    return 0
  fi

  local cuda_dir="dependency/Forward-Warp/Forward_Warp/cuda"
  local fw_root="dependency/Forward-Warp"
  if [[ ! -d "${cuda_dir}" || ! -d "${fw_root}" ]]; then
    echo "[WARN] Forward-Warp sources not found in dependency/. Skipping build."
    return 0
  fi

  export_torch_lib_path
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
import os
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
if os.environ.get("BUILD_FORWARD_WARP", "").lower() == "true" and shutil.which("nvcc"):
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
if [[ "$BUILD_FORWARD_WARP" == "true" ]]; then
  echo "       nvcc release=${NVCC_RELEASE:-n.d.}"
fi
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

build_forward_warp_if_enabled

echo
echo "[DONE] Linux setup completed."
echo "[NEXT] Close and reopen your terminal, then reactivate your preferred Python environment."
if [[ "$BUILD_FORWARD_WARP" == "true" && -n "${TORCH_LIB}" && -d "${TORCH_LIB}" ]]; then
  echo "[INFO] If needed, persist this line in your shell rc (~/.bashrc or ~/.zshrc):"
  echo "       export LD_LIBRARY_PATH=\"${TORCH_LIB}:\${LD_LIBRARY_PATH:-}\""
fi
echo "[NEXT] After reopening terminal and reactivating env, launch pipeline:"
echo "       python pipeline_master_gui.py"
