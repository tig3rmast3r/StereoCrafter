#!/usr/bin/env bash
set -euo pipefail

# Stage 2:
# - Optional/interactive download of model weights from Hugging Face

REPO_DIR="${REPO_DIR:-/workspace/StereoCrafter}"
VENV_PATH="${VENV_PATH:-/venv/main}"

if [[ -x "${VENV_PATH}/bin/activate" ]]; then
  # shellcheck disable=SC1090
  source "${VENV_PATH}/bin/activate"
else
  echo "[WARN] Venv not found at ${VENV_PATH}. Continuing with current Python."
fi

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
  Y|y) ;;
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

if ! command -v huggingface-cli >/dev/null 2>&1; then
  echo "[INFO] huggingface-cli not found. Installing huggingface_hub..."
  python -m pip install -U huggingface_hub
fi

git lfs install
mkdir -p weights
cd weights

git config --global credential.helper store

if huggingface-cli whoami >/dev/null 2>&1; then
  echo "[INFO] Hugging Face login already active."
else
  echo "[ACTION] Please login with your Hugging Face token."
  huggingface-cli login
fi

clone_lfs_repo() {
  local repo_url="$1"
  local dst
  dst="$(basename "${repo_url}")"
  if [[ -d "${dst}" ]]; then
    echo "[SKIP] ${dst} already exists."
  else
    git lfs clone "${repo_url}"
  fi
}

clone_lfs_repo "https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt-1-1"
clone_lfs_repo "https://huggingface.co/tencent/DepthCrafter"
clone_lfs_repo "https://huggingface.co/TencentARC/StereoCrafter"

cd "${REPO_DIR}"
mkdir -p work

echo "[DONE] Stage 2 complete."
