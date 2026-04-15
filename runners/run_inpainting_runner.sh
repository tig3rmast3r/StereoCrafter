#!/usr/bin/env bash
set -euo pipefail

# Edit these paths/values as needed.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-python3}"
RUNNER="${RUNNER:-$SCRIPT_DIR/batch_inpainting_runner.py}"
INPUT_DIR="${INPUT_DIR:-./work/splat/}"
INPUT_VIDEO="${INPUT_VIDEO:-}"                       # if set (non-empty), overrides INPUT_DIR
OUTPUT_DIR="${OUTPUT_DIR:-./work/output/}"
GLOB="${GLOB:-*.mp4}"

HIRES_BLEND_FOLDER="${HIRES_BLEND_FOLDER:-}"                # optional
REPLACE_MASK_FOLDER="${REPLACE_MASK_FOLDER:-./work/mask/}"              # optional; folder with <splatted_stem>_replace_mask.*
USE_REPLACE_MASK="${USE_REPLACE_MASK:-1}"                   # 1 => use external replace mask (fast-fail if missing/mismatch)
OFFLOAD_TYPE="${OFFLOAD_TYPE:-model}"                      # none | model | sequential

CHUNK_SIZE="${CHUNK_SIZE:-22}"
ENABLE_DYNAMIC_CHUNK="${ENABLE_DYNAMIC_CHUNK:-1}"
TILE_MODE="${TILE_MODE:-1 and 2}"
TILE1_MAX_SIZE="${TILE1_MAX_SIZE:-22}"
TILE2_MAX_SIZE="${TILE2_MAX_SIZE:-55}"
OVERLAP="${OVERLAP:-2}"
TAIL_PAD="${TAIL_PAD:-1}"
ORIGINAL_INPUT_BLEND_STRENGTH="${ORIGINAL_INPUT_BLEND_STRENGTH:-0}"
OUTPUT_CODEC="${OUTPUT_CODEC:-libx264}"                # optional override (e.g. libx264, h264_nvenc, hevc_nvenc)
OUTPUT_ENCODING_MODE="${OUTPUT_ENCODING_MODE:-lossless}"       # shared encoder mode for color outputs
OUTPUT_EXTRA_ARGS="${OUTPUT_EXTRA_ARGS:-}"             # optional extra ffmpeg args
PROCESS_LENGTH="${PROCESS_LENGTH:--1}"

# Steps control:
# - If you want dynamic steps from sharpness.csv, set NO_SHARPNESS_CSV=0 and (optionally) SHARPNESS_BASE.
# - If you want fixed steps for all files, set NO_SHARPNESS_CSV=1 and FIXED_STEPS.
NO_SHARPNESS_CSV="${NO_SHARPNESS_CSV:-0}"
SHARPNESS_BASE="${SHARPNESS_BASE:-./work/}"                    # folder containing sharpness.csv; empty => defaults to input folder
SHARPNESS_CSV_PATH="${SHARPNESS_CSV_PATH:-}"                    # optional explicit sharpness CSV path
FIXED_STEPS="${FIXED_STEPS:-8}"

# Mask settings
MASK_INITIAL_THRESHOLD="${MASK_INITIAL_THRESHOLD:-0.3}"
MASK_MORPH_KERNEL_SIZE="${MASK_MORPH_KERNEL_SIZE:-0.0}"
MASK_DILATE_KERNEL_SIZE="${MASK_DILATE_KERNEL_SIZE:-5}"
MASK_BLUR_KERNEL_SIZE="${MASK_BLUR_KERNEL_SIZE:-10}"

ENABLE_POST_INPAINTING_BLEND="${ENABLE_POST_INPAINTING_BLEND:-0}"        # 1 to enable
DISABLE_COLOR_TRANSFER="${DISABLE_COLOR_TRANSFER:-1}"                     # 1 to disable

SKIP_EXISTING="${SKIP_EXISTING:-1}"
MOVE_FINISHED="${MOVE_FINISHED:-0}"
FINISHED_SUBDIR="${FINISHED_SUBDIR:-finished}"

RETRY_POLICY_JSON="${RETRY_POLICY_JSON:-}"

DEBUG="${DEBUG:-0}"
STOP_MARKER="${STOP_MARKER:-$OUTPUT_DIR/.stop_after_current}"

# --- runner command (edit freely) ---
CMD=("$PYTHON" "$RUNNER")

if [[ -n "$INPUT_VIDEO" ]]; then
  CMD+=(--input_video "$INPUT_VIDEO")
else
  CMD+=(--input_dir "$INPUT_DIR" --glob "$GLOB")
fi

CMD+=(--output_dir "$OUTPUT_DIR"
     --stop_marker "$STOP_MARKER"
     --chunk_size "$CHUNK_SIZE"
     --tile_mode "$TILE_MODE"
     --tile1_max_size "$TILE1_MAX_SIZE"
     --tile2_max_size "$TILE2_MAX_SIZE"
     --overlap "$OVERLAP"
     --tail_pad "$TAIL_PAD"
     --original_input_blend_strength "$ORIGINAL_INPUT_BLEND_STRENGTH"
     --process_length "$PROCESS_LENGTH"
     --offload_type "$OFFLOAD_TYPE"
     --hires_blend_folder "$HIRES_BLEND_FOLDER"
     --replace_mask_folder "$REPLACE_MASK_FOLDER"
     --mask_initial_threshold "$MASK_INITIAL_THRESHOLD"
     --mask_morph_kernel_size "$MASK_MORPH_KERNEL_SIZE"
     --mask_dilate_kernel_size "$MASK_DILATE_KERNEL_SIZE"
     --mask_blur_kernel_size "$MASK_BLUR_KERNEL_SIZE"
     --fixed_steps "$FIXED_STEPS"
     --finished_subdir "$FINISHED_SUBDIR"
)

if [[ "$NO_SHARPNESS_CSV" == "1" ]]; then
  CMD+=(--no_sharpness_csv)
else
  if [[ -n "$SHARPNESS_CSV_PATH" ]]; then
    CMD+=(--sharpness_csv_path "$SHARPNESS_CSV_PATH")
  elif [[ -n "$SHARPNESS_BASE" ]]; then
    CMD+=(--sharpness_base "$SHARPNESS_BASE")
  fi
fi

if [[ "$ENABLE_POST_INPAINTING_BLEND" == "1" ]]; then CMD+=(--enable_post_inpainting_blend); fi
if [[ "$USE_REPLACE_MASK" == "1" ]]; then CMD+=(--use_replace_mask); fi
if [[ "$DISABLE_COLOR_TRANSFER" == "1" ]]; then CMD+=(--disable_color_transfer); fi
if [[ -n "$OUTPUT_CODEC" ]]; then CMD+=(--output_codec "$OUTPUT_CODEC"); fi
if [[ -n "$OUTPUT_ENCODING_MODE" ]]; then CMD+=(--output_encoding_mode "$OUTPUT_ENCODING_MODE"); fi
if [[ -n "$OUTPUT_EXTRA_ARGS" ]]; then CMD+=(--output_extra_args "$OUTPUT_EXTRA_ARGS"); fi
if [[ "$SKIP_EXISTING" == "1" ]]; then CMD+=(--skip_existing); fi
if [[ "$MOVE_FINISHED" == "1" ]]; then CMD+=(--move_finished); fi
if [[ "$DEBUG" == "1" ]]; then CMD+=(--debug); fi
if [[ "$ENABLE_DYNAMIC_CHUNK" == "1" ]]; then
  CMD+=(--enable_dynamic_chunk)
else
  CMD+=(--disable_dynamic_chunk)
fi
if [[ -n "$RETRY_POLICY_JSON" ]]; then CMD+=(--retry_policy_json "$RETRY_POLICY_JSON"); fi

echo "[CMD] ${CMD[*]}"

# Retry/Watchdog policy.
# Set MAX_RETRIES=0 for infinite restarts.
MAX_RETRIES="${MAX_RETRIES:-0}"
RETRY_SLEEP_SEC="${RETRY_SLEEP_SEC:-2}"
WATCHDOG_ENABLED="${WATCHDOG_ENABLED:-False}"
WATCHDOG_POLL_SEC="${WATCHDOG_POLL_SEC:-20}"
WATCHDOG_IDLE_SEC="${WATCHDOG_IDLE_SEC:-600}"
WATCHDOG_TERM_GRACE_SEC="${WATCHDOG_TERM_GRACE_SEC:-15}"
STOP_REQUESTED=0
FORCE_STOP=0
CURRENT_CHILD_PID=""
CURRENT_CHILD_PGID=""

if [[ -f "$STOP_MARKER" ]]; then
  echo "[INFO] removing stale stop marker: $STOP_MARKER"
  rm -f -- "$STOP_MARKER" || true
fi

_latest_mp4_in_output() {
  find "$OUTPUT_DIR" -type f -name "*.mp4" -printf '%T@|%p\n' 2>/dev/null \
    | awk -F'|' '
      BEGIN { max_ts = -1; latest = "" }
      {
        ts = $1 + 0
        if (ts > max_ts) {
          max_ts = ts
          latest = $2
        }
      }
      END {
        if (latest != "") print latest
      }
    ' || true
  return 0
}

_ffprobe_quick_ok() {
  local f="$1"
  [[ -n "$f" && -f "$f" ]] || return 1
  ffprobe -v error \
    -select_streams v:0 \
    -show_entries stream=codec_name,width,height,avg_frame_rate,nb_frames \
    -show_entries format=duration \
    -of default=nw=1:nk=1 \
    "$f" >/dev/null 2>&1
}

_cleanup_unreadable_latest_output() {
  local last_mp4
  last_mp4="$(_latest_mp4_in_output)"
  if [[ -z "$last_mp4" ]]; then
    echo "[CHECK] no output mp4 found to validate."
    return 0
  fi

  if _ffprobe_quick_ok "$last_mp4"; then
    echo "[CHECK] last output readable: $last_mp4"
    return 0
  fi

  echo "[CHECK] last output unreadable, removing before restart: $last_mp4"
  rm -f -- "$last_mp4" || true
}

_latest_output_token() {
  find "$OUTPUT_DIR" -type f -printf '%T@|%s|%p\n' 2>/dev/null \
    | awk -F'|' '
      BEGIN { max_ts = -1; latest = "" }
      {
        ts = $1 + 0
        if (ts > max_ts) {
          max_ts = ts
          latest = $0
        }
      }
      END {
        if (latest != "") print latest
      }
    ' || true
  return 0
}

_pgid_has_members() {
  local pgid="$1"
  [[ -n "$pgid" ]] || return 1
  ps -o pid= -g "$pgid" 2>/dev/null | awk 'NF{found=1; exit} END{exit found?0:1}'
}

_kill_child_group() {
  local pid="$1"
  local pgid="$2"

  if [[ -n "$pgid" ]]; then
    kill -TERM -- "-$pgid" 2>/dev/null || true
  fi
  kill -TERM "$pid" 2>/dev/null || true

  sleep "$WATCHDOG_TERM_GRACE_SEC"

  if [[ -n "$pgid" ]] && _pgid_has_members "$pgid"; then
    kill -KILL -- "-$pgid" 2>/dev/null || true
  fi
  if kill -0 "$pid" 2>/dev/null; then
    kill -KILL "$pid" 2>/dev/null || true
  fi
}

_request_stop_signal() {
  if [[ "$STOP_REQUESTED" -eq 0 ]]; then
    STOP_REQUESTED=1
    local marker_dir
    marker_dir="$(dirname "$STOP_MARKER")"
    mkdir -p "$marker_dir" 2>/dev/null || true
    : > "$STOP_MARKER"
    echo "[STOP] graceful stop requested. Finishing current file before exiting."
    return 0
  fi

  FORCE_STOP=1
  echo "[STOP] force stop requested. Killing runner immediately."
  if [[ -n "$CURRENT_CHILD_PID" ]] && kill -0 "$CURRENT_CHILD_PID" 2>/dev/null; then
    _kill_child_group "$CURRENT_CHILD_PID" "$CURRENT_CHILD_PGID"
  fi
}

trap _request_stop_signal INT TERM

_is_true() {
  local v="${1:-}"
  case "${v,,}" in
    1|true|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

_run_once_with_watchdog() {
  local child_pid child_pgid self_pgid
  local last_token current_token
  local last_activity_ts now idle_sec

  if command -v setsid >/dev/null 2>&1; then
    setsid "${CMD[@]}" &
  else
    "${CMD[@]}" &
  fi
  child_pid=$!
  child_pgid="$(ps -o pgid= -p "$child_pid" 2>/dev/null | tr -d '[:space:]')"
  self_pgid="$(ps -o pgid= -p "$$" 2>/dev/null | tr -d '[:space:]')"
  if [[ -n "$child_pgid" && -n "$self_pgid" && "$child_pgid" == "$self_pgid" ]]; then
    # If setsid is unavailable and child shares our PGID, avoid group-kill on our own shell group.
    child_pgid=""
  fi
  CURRENT_CHILD_PID="$child_pid"
  CURRENT_CHILD_PGID="$child_pgid"

  if ! _is_true "$WATCHDOG_ENABLED"; then
    local rc=0
    if wait "$child_pid"; then
      rc=0
    else
      rc=$?
    fi
    CURRENT_CHILD_PID=""
    CURRENT_CHILD_PGID=""
    return "$rc"
  fi

  last_token="$(_latest_output_token)"
  last_activity_ts=$(date +%s)

  while kill -0 "$child_pid" 2>/dev/null; do
    sleep "$WATCHDOG_POLL_SEC"

    current_token="$(_latest_output_token)"
    if [[ -n "$current_token" && "$current_token" != "$last_token" ]]; then
      last_token="$current_token"
      last_activity_ts=$(date +%s)
      continue
    fi

    now=$(date +%s)
    idle_sec=$((now - last_activity_ts))
    if (( idle_sec >= WATCHDOG_IDLE_SEC )); then
      echo "[WATCHDOG] no output activity for ${idle_sec}s. Killing runner pid=$child_pid pgid=${child_pgid:-n/a}"
      _kill_child_group "$child_pid" "$child_pgid"
      wait "$child_pid" 2>/dev/null || true
      CURRENT_CHILD_PID=""
      CURRENT_CHILD_PGID=""
      return 124
    fi
  done

  local rc=0
  if wait "$child_pid"; then
    rc=0
  else
    rc=$?
  fi
  CURRENT_CHILD_PID=""
  CURRENT_CHILD_PGID=""
  return "$rc"
}

attempt=1
while true; do
  set +e
  _run_once_with_watchdog
  rc=$?
  set -e

  if [[ "$FORCE_STOP" -eq 1 ]]; then
    rm -f -- "$STOP_MARKER" 2>/dev/null || true
    echo "[STOP] forced stop completed."
    exit 130
  fi

  if [[ "$STOP_REQUESTED" -eq 1 ]]; then
    rm -f -- "$STOP_MARKER" 2>/dev/null || true
    echo "[STOP] graceful stop completed (last rc=$rc)."
    exit 0
  fi

  if [[ $rc -eq 0 ]]; then
    exit 0
  fi

  echo "[WARN] runner exited with rc=$rc (attempt $attempt). Restarting..."
  _cleanup_unreadable_latest_output

  if [[ $MAX_RETRIES -ne 0 && $attempt -ge $MAX_RETRIES ]]; then
    echo "[ERR] reached MAX_RETRIES=$MAX_RETRIES, giving up (last rc=$rc)"
    exit $rc
  fi

  attempt=$((attempt+1))
  sleep "$RETRY_SLEEP_SEC"
done
