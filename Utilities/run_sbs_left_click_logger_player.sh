#!/usr/bin/env bash
set -euo pipefail

UTILS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$UTILS_DIR/.." && pwd)"
LUA_SCRIPT="${LUA_SCRIPT:-$UTILS_DIR/lua/sbs_click_logger.lua}"

DEFAULT_SBS_DIR="${1:-${DIR_SBS:-$REPO_ROOT/work/sbs}}"
DEFAULT_CSV_PATH="${2:-${CSV_PATH:-$REPO_ROOT/sbs_click_annotations.csv}}"

read -r -p "SBS folder [${DEFAULT_SBS_DIR}]: " SBS_DIR_INPUT
SBS_DIR="${SBS_DIR_INPUT:-$DEFAULT_SBS_DIR}"

read -r -p "CSV output path [${DEFAULT_CSV_PATH}]: " CSV_PATH_INPUT
CSV_PATH="${CSV_PATH_INPUT:-$DEFAULT_CSV_PATH}"

if ! command -v mpv >/dev/null 2>&1; then
  echo "[ERR] mpv not found in PATH" >&2
  exit 1
fi

if [[ ! -d "$SBS_DIR" ]]; then
  echo "[ERR] folder not found: $SBS_DIR" >&2
  exit 1
fi

if [[ ! -f "$LUA_SCRIPT" ]]; then
  echo "[ERR] Lua script not found: $LUA_SCRIPT" >&2
  exit 1
fi

mapfile -d '' -t files < <(
  find "$SBS_DIR" -maxdepth 1 -type f \
    \( -iname '*.mp4' -o -iname '*.mkv' -o -iname '*.mov' -o -iname '*.webm' -o -iname '*.avi' -o -iname '*.m4v' \) \
    -print0 | sort -z
)

if (( ${#files[@]} == 0 )); then
  echo "[ERR] no video files found in: $SBS_DIR" >&2
  exit 1
fi

mkdir -p "$(dirname "$CSV_PATH")"

echo "[INFO] SBS folder: $SBS_DIR"
echo "[INFO] CSV append file: $CSV_PATH"
echo "[INFO] Files in playlist: ${#files[@]}"

mpv \
  --keep-open=always \
  --idle=yes \
  --force-window=yes \
  --vf-add="crop=iw/2:ih:iw/2:0" \
  --script="$LUA_SCRIPT" \
  --script-opts="sbs_click_logger-csv_path=$CSV_PATH,sbs_click_logger-source_mode=right_half,sbs_click_logger-show_osd=yes" \
  -- "${files[@]}"
