#!/usr/bin/env bash
set -euo pipefail

# --------------------------------------------
# User-editable parameters (via env override)
# --------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-python}"
RUNNER="${RUNNER:-$SCRIPT_DIR/batch_depthcrafter_runner.py}"
WORKER_SCRIPT="${WORKER_SCRIPT:-$SCRIPT_DIR/depthcrafter_nogui_batch.py}"

INPUT_DIR="${INPUT_DIR:-./work/seg/}"
OUTPUT_DIR="${OUTPUT_DIR:-./work/depthmap/}"
GLOB="${GLOB:-*.mp4}"

WINDOW_SIZE="${WINDOW_SIZE:-70}"
OVERLAP="${OVERLAP:-20}"
INFERENCE_STEPS="${INFERENCE_STEPS:-5}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-1.0}"
SEED="${SEED:-42}"
CPU_OFFLOAD_MODE="${CPU_OFFLOAD_MODE:-model}"
DECODE_CHUNK_SIZE="${DECODE_CHUNK_SIZE:-2}"
DEBUG_MEM="${DEBUG_MEM:-True}"
FINAL_UPSCALE="${FINAL_UPSCALE:-False}"
SCALE_FACTOR="${SCALE_FACTOR:-0.5}"
RESTART_EVERY="${RESTART_EVERY:-100}"
PAD_ALIGN_BOTTOM="${PAD_ALIGN_BOTTOM:-True}"
SCENE_STRIP_PAD_TOP="${SCENE_STRIP_PAD_TOP:-0}"
SCENE_STRIP_PAD_BOTTOM="${SCENE_STRIP_PAD_BOTTOM:-0}"

# Depth preprocess codec override. Temporary/final grayscale outputs are fixed.
FFMPEG_CODEC="${FFMPEG_CODEC:-}"
RETRY_POLICY_JSON="${RETRY_POLICY_JSON:-}"
RETRY_PROCESS_RESTART_ALLOC_MB="${RETRY_PROCESS_RESTART_ALLOC_MB:-1024}"

# ---------------------------------
# Crash/kill retry policy (process)
# ---------------------------------
MAX_RETRIES="${MAX_RETRIES:-100}"      # set 0 for infinite retries
RETRY_SLEEP_SEC="${RETRY_SLEEP_SEC:-2}"
RETRY_CODES_DEFAULT="124 133 135 136 137 139 132 134"
RETRY_CODES="${RETRY_CODES:-$RETRY_CODES_DEFAULT}"

# Watchdog policy (output activity)
# Disabled by default: long clips can legitimately take > idle timeout between output updates.
WATCHDOG_ENABLED="${WATCHDOG_ENABLED:-False}"
WATCHDOG_POLL_SEC="${WATCHDOG_POLL_SEC:-20}"
WATCHDOG_IDLE_SEC="${WATCHDOG_IDLE_SEC:-600}"
WATCHDOG_TERM_GRACE_SEC="${WATCHDOG_TERM_GRACE_SEC:-10}"

STOP_MARKER="${STOP_MARKER:-$OUTPUT_DIR/.stop_after_current}"
STOP_REQUEST_FILE="${TMPDIR:-/tmp}/depth_stop_request_${$}.flag"
STOP_REQUESTED=0
FORCE_STOP=0
CURRENT_CHILD_PID=""
CURRENT_CHILD_PGID=""

mkdir -p "$OUTPUT_DIR"
if [[ -f "$STOP_MARKER" ]]; then
  echo "[INFO] removing stale stop marker: $STOP_MARKER"
  rm -f -- "$STOP_MARKER" || true
fi

CMD=(
  "$PYTHON" "$RUNNER"
  --worker_script "$WORKER_SCRIPT"
  --input_dir "$INPUT_DIR"
  --output_dir "$OUTPUT_DIR"
  --glob "$GLOB"
  --window_size "$WINDOW_SIZE"
  --overlap "$OVERLAP"
  --inference_steps "$INFERENCE_STEPS"
  --guidance_scale "$GUIDANCE_SCALE"
  --seed "$SEED"
  --cpu_offload_mode "$CPU_OFFLOAD_MODE"
  --decode_chunk_size "$DECODE_CHUNK_SIZE"
  --debug_mem "$DEBUG_MEM"
  --final_upscale "$FINAL_UPSCALE"
  --scale_factor "$SCALE_FACTOR"
  --restart_every "$RESTART_EVERY"
  --pad_align_bottom "$PAD_ALIGN_BOTTOM"
  --scene_strip_pad_top "$SCENE_STRIP_PAD_TOP"
  --scene_strip_pad_bottom "$SCENE_STRIP_PAD_BOTTOM"
)

if [[ -n "${FFMPEG_CODEC// }" ]]; then
  CMD+=(--ffmpeg_codec "$FFMPEG_CODEC")
fi
if [[ -n "${RETRY_POLICY_JSON// }" ]]; then
  CMD+=(--retry_policy_json "$RETRY_POLICY_JSON")
fi
if [[ "$RETRY_PROCESS_RESTART_ALLOC_MB" =~ ^[0-9]+$ ]] && [[ "$RETRY_PROCESS_RESTART_ALLOC_MB" -ge 0 ]]; then
  CMD+=(--retry_process_restart_alloc_mb "$RETRY_PROCESS_RESTART_ALLOC_MB")
fi

echo "[CMD] ${CMD[*]}"

cleanup_runtime() {
  rm -f "$STOP_REQUEST_FILE" 2>/dev/null || true
  rm -f "$STOP_MARKER" 2>/dev/null || true
}

_latest_mp4_in_output() {
  find "$OUTPUT_DIR" -maxdepth 1 -type f -name "*.mp4" -printf '%T@|%p\n' 2>/dev/null \
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
    return 0
  fi
  if _ffprobe_quick_ok "$last_mp4"; then
    return 0
  fi
  echo "[CLEANUP] removing unreadable output: $last_mp4"
  rm -f -- "$last_mp4" || true
}

_cleanup_depth_tmp() {
  local tmp_dir="$OUTPUT_DIR/.tmp_depthcrafter"
  if [[ -d "$tmp_dir" ]]; then
    echo "[CLEANUP] removing temp dir: $tmp_dir"
    rm -rf -- "$tmp_dir" || true
  fi
}

_latest_output_token() {
  find "$OUTPUT_DIR" -type f ! -name ".stop_after_current" -printf '%T@|%s|%p\n' 2>/dev/null \
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

_kill_child_group() {
  local pid="$1"
  local pgid="$2"
  if [[ -n "$pgid" ]]; then
    kill -TERM -- "-$pgid" 2>/dev/null || true
  else
    kill -TERM "$pid" 2>/dev/null || true
  fi

  sleep "$WATCHDOG_TERM_GRACE_SEC"

  if kill -0 "$pid" 2>/dev/null; then
    if [[ -n "$pgid" ]]; then
      kill -KILL -- "-$pgid" 2>/dev/null || true
    fi
    kill -KILL "$pid" 2>/dev/null || true
  fi
}

_send_child_interrupt() {
  local pid="$1"
  local pgid="$2"
  if [[ -n "$pgid" ]]; then
    kill -INT -- "-$pgid" 2>/dev/null || true
  else
    kill -INT "$pid" 2>/dev/null || true
  fi
}

_is_true() {
  local v="${1:-}"
  case "${v,,}" in
    1|true|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

_request_stop_signal() {
  if [[ "$STOP_REQUESTED" -eq 0 ]]; then
    STOP_REQUESTED=1
    local marker_dir
    marker_dir="$(dirname "$STOP_MARKER")"
    mkdir -p "$marker_dir" 2>/dev/null || true
    : > "$STOP_MARKER"
    : > "$STOP_REQUEST_FILE"
    echo "[STOP] graceful stop requested (Ctrl+C style)."
    if [[ -n "$CURRENT_CHILD_PID" ]] && kill -0 "$CURRENT_CHILD_PID" 2>/dev/null; then
      _send_child_interrupt "$CURRENT_CHILD_PID" "$CURRENT_CHILD_PGID"
    fi
    return 0
  fi

  FORCE_STOP=1
  echo "[STOP] force stop requested. Killing now."
  if [[ -n "$CURRENT_CHILD_PID" ]] && kill -0 "$CURRENT_CHILD_PID" 2>/dev/null; then
    _kill_child_group "$CURRENT_CHILD_PID" "$CURRENT_CHILD_PGID"
  fi
  _cleanup_unreadable_latest_output
  _cleanup_depth_tmp
}

should_retry() {
  local _code="${1:-1}"
  # Retry on any non-zero exit code; MAX_RETRIES still applies.
  return 0
}

_run_once_with_watchdog() {
  local child_pid child_pgid
  local last_token current_token
  local last_activity_ts now idle_sec

  if command -v setsid >/dev/null 2>&1; then
    setsid "${CMD[@]}" &
  else
    "${CMD[@]}" &
  fi
  child_pid=$!
  child_pgid="$(ps -o pgid= -p "$child_pid" 2>/dev/null | tr -d '[:space:]')"
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

trap _request_stop_signal INT TERM
trap cleanup_runtime EXIT

attempt=1
while true; do
  set +e
  _run_once_with_watchdog
  rc=$?
  set -e

  if [[ "$FORCE_STOP" -eq 1 ]]; then
    echo "[STOP] forced stop completed."
    exit 130
  fi

  if [[ "$STOP_REQUESTED" -eq 1 ]]; then
    _cleanup_unreadable_latest_output
    _cleanup_depth_tmp
    echo "[STOP] graceful stop completed (last rc=$rc)."
    exit 0
  fi

  if [[ "$rc" -eq 0 ]]; then
    echo "[OK] success"
    exit 0
  fi

  echo "[WARN] runner exited with rc=$rc (attempt $attempt). Restarting..."
  _cleanup_unreadable_latest_output
  _cleanup_depth_tmp

  if [[ "$MAX_RETRIES" -ne 0 && "$attempt" -ge "$MAX_RETRIES" ]]; then
    echo "[FAIL] reached MAX_RETRIES=$MAX_RETRIES (last rc=$rc)"
    exit "$rc"
  fi
  if ! should_retry "$rc"; then
    echo "[FAIL] exit_code=$rc is not retryable"
    exit "$rc"
  fi

  for ((s=0; s<RETRY_SLEEP_SEC; s++)); do
    if [[ "$STOP_REQUESTED" -eq 1 ]]; then
      _cleanup_unreadable_latest_output
      _cleanup_depth_tmp
      echo "[STOP] graceful stop completed."
      exit 0
    fi
    sleep 1
  done
  attempt=$((attempt + 1))
done
