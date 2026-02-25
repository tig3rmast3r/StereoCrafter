#!/usr/bin/env bash
set -euo pipefail

# Stage 2:
# - Optional/interactive download of model weights from Hugging Face

SCRIPT_DIR="$(
  cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1
  pwd -P
)"
REPO_DIR="${REPO_DIR:-${SCRIPT_DIR}}"

# Ensure CLI scripts installed for current Python are discoverable in PATH.
PY_BIN_DIR="$(
  python - <<'PY'
import os, sys
print(os.path.dirname(sys.executable))
PY
)"
export PATH="${PY_BIN_DIR}:${PATH}"

HF_CLI=""
detect_hf_cli() {
  if command -v hf >/dev/null 2>&1; then
    HF_CLI="hf"
  elif command -v huggingface-cli >/dev/null 2>&1; then
    HF_CLI="huggingface-cli"
  else
    HF_CLI=""
  fi
}

run_hf_login() {
  local -a login_cmd
  if [[ "${HF_CLI}" == "hf" ]]; then
    login_cmd=(hf auth login)
  else
    login_cmd=(huggingface-cli login)
  fi

  if ! "${login_cmd[@]}"; then
    echo "[ERR] Hugging Face login failed."
    exit 2
  fi
}

clone_and_pull_repo() {
  local repo_url="$1"
  local dst="$2"
  if [[ -e "${dst}" ]]; then
    echo "[ERR] ${dst} already exists in $(pwd)"
    echo "[ERR] For first-install flow, remove existing folders and rerun stage2."
    exit 3
  fi

  git clone "${repo_url}" "${dst}"
  git -C "${dst}" lfs install --local
  git -C "${dst}" lfs pull
}

cat <<'EOF'
[WARNING]
This Stage 2 script ONLY downloads model weights.
It requires:
  1) A Hugging Face account
  2) A valid Hugging Face token (Read access)

If you plan to provide weights by other methods, you can skip this stage.
EOF

read -r -p "Continue with Hugging Face weight download? [Y/N]: " CONTINUE_STAGE2
case "${CONTINUE_STAGE2}" in
  Y|y|YES|Yes|yes) ;;
  *)
    echo "[SKIP] Stage 2 skipped by user."
    exit 0
    ;;
esac

if [[ ! -d "${REPO_DIR}/.git" ]]; then
  echo "[ERR] Repository not found at ${REPO_DIR}. Run stage1 first."
  exit 2
fi

cd "${REPO_DIR}"

detect_hf_cli
if [[ -z "${HF_CLI}" ]]; then
  echo "[INFO] Hugging Face CLI not found. Installing huggingface_hub..."
  python -m pip install -U huggingface_hub
  hash -r
  detect_hf_cli
fi

if [[ -z "${HF_CLI}" ]]; then
  echo "[ERR] Could not find Hugging Face CLI after install."
  echo "[ERR] Expected one of: hf, huggingface-cli"
  exit 2
fi

git lfs install
mkdir -p weights
cd weights

git config --global credential.helper store

echo "[ACTION] Hugging Face login required."
echo "[ACTION] Paste token when prompted; answer YES to 'add token to git credential' for git-lfs."
run_hf_login

clone_and_pull_repo "https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt-1-1" "stable-video-diffusion-img2vid-xt-1-1"
clone_and_pull_repo "https://huggingface.co/tencent/DepthCrafter" "DepthCrafter"
clone_and_pull_repo "https://huggingface.co/TencentARC/StereoCrafter" "StereoCrafter"

cd "${REPO_DIR}"
mkdir -p work

echo "[DONE] Stage 2 complete."
