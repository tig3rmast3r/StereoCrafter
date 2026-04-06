#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(
  cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1
  pwd -P
)"
REPO_DIR="${REPO_DIR:-${SCRIPT_DIR}}"

usage() {
  cat <<'EOF'
Usage:
  setup_vast_stage2_weights_temp.sh [mode] [--dry-run]

Modes:
  1 | depth | depthcrafter            Download only DepthCrafter + shared SVD base
  2 | inpaint | inpainting            Download only StereoCrafter inpaint + shared SVD base
  3 | both | depth+inpaint            Download DepthCrafter + StereoCrafter inpaint + shared SVD base

Token:
  Actual downloads always use a Hugging Face token.
  The script reads one of:
    HF_TOKEN
    HUGGINGFACE_TOKEN
    HUGGING_FACE_HUB_TOKEN
  If none is set, it prompts for the token interactively.

Notes:
  - This temp script performs targeted file downloads only.
  - It does not clone Hugging Face repos.
  - Managed folders are pruned so they contain only the required files for the selected mode.
  - --dry-run prints the exact download manifest without contacting Hugging Face.
EOF
}

normalize_mode() {
  local raw="${1,,}"
  case "${raw}" in
    1|depth|depthcrafter|depth-only|depthcrafter-only)
      echo "depth"
      ;;
    2|inpaint|inpainting|inpaint-only|inpainting-only)
      echo "inpaint"
      ;;
    3|both|depth+inpaint|depthcrafter+inpaint|depth-and-inpaint|depthcrafter-and-inpaint)
      echo "both"
      ;;
    *)
      return 1
      ;;
  esac
}

MODE_RAW=""
DRY_RUN=0
for arg in "$@"; do
  case "${arg}" in
    -h|--help)
      usage
      exit 0
      ;;
    --dry-run)
      DRY_RUN=1
      ;;
    *)
      if [[ -n "${MODE_RAW}" ]]; then
        echo "[ERR] Too many positional arguments."
        usage
        exit 2
      fi
      MODE_RAW="${arg}"
      ;;
  esac
done

if [[ -z "${MODE_RAW}" ]]; then
  cat <<'EOF'
Select Stage 2 download mode:
  1) depthcrafter only
  2) inpaint only
  3) depthcrafter + inpaint
EOF
  read -r -p "Choice [1-3]: " MODE_RAW
fi

if ! MODE="$(normalize_mode "${MODE_RAW}")"; then
  echo "[ERR] Unsupported mode: ${MODE_RAW}"
  usage
  exit 2
fi

if [[ ! -d "${REPO_DIR}/.git" ]]; then
  echo "[ERR] Repository not found at ${REPO_DIR}. Run from repo root or set REPO_DIR."
  exit 2
fi

if ! python - <<'PY' >/dev/null 2>&1
import huggingface_hub  # noqa: F401
PY
then
  echo "[ERR] Missing Python module: huggingface_hub"
  echo "[ERR] Install it first, then rerun:"
  echo "       source \"${REPO_DIR}/.venv/bin/activate\" && python -m pip install -U huggingface_hub"
  exit 2
fi

HF_TOKEN_VALUE=""
if [[ "${DRY_RUN}" -eq 0 ]]; then
  HF_TOKEN_VALUE="${HF_TOKEN:-${HUGGINGFACE_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}}"
  if [[ -z "${HF_TOKEN_VALUE}" ]]; then
    read -r -s -p "Hugging Face token: " HF_TOKEN_VALUE
    echo
  fi
  if [[ -z "${HF_TOKEN_VALUE}" ]]; then
    echo "[ERR] Hugging Face token is required for actual downloads."
    exit 2
  fi
fi

mkdir -p "${REPO_DIR}/weights" "${REPO_DIR}/work"

export STAGE2_MODE="${MODE}"
export STAGE2_DRY_RUN="${DRY_RUN}"
export STAGE2_REPO_DIR="${REPO_DIR}"
export STAGE2_HF_TOKEN="${HF_TOKEN_VALUE}"

python - <<'PY'
import os
import shutil
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

MODE = os.environ["STAGE2_MODE"]
DRY_RUN = os.environ["STAGE2_DRY_RUN"] == "1"
REPO_DIR = Path(os.environ["STAGE2_REPO_DIR"])
WEIGHTS_DIR = REPO_DIR / "weights"
HF_TOKEN = os.environ.get("STAGE2_HF_TOKEN", "")

MANIFESTS = {
    "stable-video-diffusion-img2vid-xt-1-1": {
        "repo_id": "stabilityai/stable-video-diffusion-img2vid-xt-1-1",
        "files": [
            "model_index.json",
            "feature_extractor/preprocessor_config.json",
            "image_encoder/config.json",
            "image_encoder/model.fp16.safetensors",
            "scheduler/scheduler_config.json",
            "vae/config.json",
            "vae/diffusion_pytorch_model.fp16.safetensors",
        ],
    },
    "DepthCrafter": {
        "repo_id": "tencent/DepthCrafter",
        "files": [
            "config.json",
            "diffusion_pytorch_model.safetensors",
        ],
    },
    "StereoCrafter": {
        "repo_id": "TencentARC/StereoCrafter",
        "files": [
            "config.json",
            "diffusion_pytorch_model.safetensors",
        ],
    },
}

MODE_TARGETS = {
    "depth": ["stable-video-diffusion-img2vid-xt-1-1", "DepthCrafter"],
    "inpaint": ["stable-video-diffusion-img2vid-xt-1-1", "StereoCrafter"],
    "both": ["stable-video-diffusion-img2vid-xt-1-1", "DepthCrafter", "StereoCrafter"],
}


def build_allowed_dirs(files: list[str]) -> set[str]:
    allowed_dirs = {""}
    for rel in files:
        parent = Path(rel).parent
        while str(parent) != ".":
            allowed_dirs.add(parent.as_posix())
            parent = parent.parent
    return allowed_dirs


def prune_managed_tree(root: Path, files: list[str]) -> list[str]:
    removed: list[str] = []
    if not root.exists():
        return removed
    allowed_files = set(files)
    allowed_dirs = build_allowed_dirs(files)

    file_paths = sorted(
        [p for p in root.rglob("*") if p.is_file() or p.is_symlink()],
        key=lambda p: len(p.parts),
        reverse=True,
    )
    for path in file_paths:
        rel = path.relative_to(root).as_posix()
        if rel not in allowed_files:
            path.unlink()
            removed.append(rel)

    dir_paths = sorted(
        [p for p in root.rglob("*") if p.is_dir()],
        key=lambda p: len(p.parts),
        reverse=True,
    )
    for path in dir_paths:
        rel = path.relative_to(root).as_posix()
        if rel not in allowed_dirs:
            shutil.rmtree(path, ignore_errors=True)
            removed.append(rel + "/")

    return removed


def remove_root_garbage(weights_dir: Path) -> None:
    junk = [
        weights_dir / ".ipynb_checkpoints",
    ]
    for path in junk:
        if path.exists():
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            else:
                path.unlink()
            print(f"[PRUNE] removed {path.relative_to(weights_dir).as_posix()}")


def sync_model(model_name: str) -> None:
    spec = MANIFESTS[model_name]
    repo_id = spec["repo_id"]
    files = spec["files"]
    target_dir = WEIGHTS_DIR / model_name
    target_dir.mkdir(parents=True, exist_ok=True)

    print(f"[SYNC] {model_name} <- {repo_id}")
    for rel in files:
        print(f"       - {rel}")

    if DRY_RUN:
        return

    for rel in files:
        try:
            cached = Path(
                hf_hub_download(
                    repo_id=repo_id,
                    filename=rel,
                    token=HF_TOKEN,
                )
            )
        except Exception as exc:
            print(f"[ERR] Failed to download {repo_id}:{rel}")
            print(f"[ERR] {type(exc).__name__}: {exc}")
            sys.exit(3)

        dst = target_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() and dst.stat().st_size == cached.stat().st_size:
            print(f"[KEEP] {model_name}/{rel}")
        else:
            shutil.copy2(cached, dst)
            print(f"[COPY] {model_name}/{rel}")

    removed = prune_managed_tree(target_dir, files)
    if removed:
        print(f"[PRUNE] {model_name}: removed {len(removed)} extra entries")
    else:
        print(f"[PRUNE] {model_name}: already minimal")


selected = MODE_TARGETS[MODE]
print(f"[INFO] Repo dir: {REPO_DIR}")
print(f"[INFO] Weights dir: {WEIGHTS_DIR}")
print(f"[INFO] Mode: {MODE}")
if DRY_RUN:
    print("[INFO] Dry run only; no Hugging Face requests will be made.")

remove_root_garbage(WEIGHTS_DIR)

for name in selected:
    sync_model(name)

print("[DONE] Stage 2 targeted sync complete.")
PY
