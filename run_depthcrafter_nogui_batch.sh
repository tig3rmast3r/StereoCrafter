#!/usr/bin/env bash
set -euo pipefail

# --------------------------------------------
# User-editable parameters (via env override)
# --------------------------------------------
PYTHON="${PYTHON:-python}"
RUNNER="${RUNNER:-batch_depthcrafter_runner.py}"
WORKER_SCRIPT="${WORKER_SCRIPT:-./depthcrafter_nogui_batch.py}"

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

# Optional ffmpeg overrides (leave empty/default to keep legacy behavior)
FFMPEG_CODEC="${FFMPEG_CODEC:-}"
FFMPEG_CRF="${FFMPEG_CRF:--1}"
FFMPEG_PRESET="${FFMPEG_PRESET:-}"
FFMPEG_PIX_FMT="${FFMPEG_PIX_FMT:-}"
FFMPEG_EXTRA_ARGS="${FFMPEG_EXTRA_ARGS:-}"

# Optional RealESRGAN upscale stage (used by pipeline GUI auto mode)
USE_REALESRGAN_UPSCALE="${USE_REALESRGAN_UPSCALE:-False}"
REALESRGAN_UPSCALE_SCRIPT="${REALESRGAN_UPSCALE_SCRIPT:-Utilities/upscale_esrgan_x264.sh}"
REALESRGAN_SCALE="${REALESRGAN_SCALE:-2}"
REALESRGAN_MODEL="${REALESRGAN_MODEL:-realesr-animevideov3-x2}"
REALESRGAN_TILE="${REALESRGAN_TILE:-auto}"
REALESRGAN_DEST="${REALESRGAN_DEST:-}"
REALESRGAN_JOBS="${REALESRGAN_JOBS:-$(nproc)}"
REALESRGAN_RETRIES="${REALESRGAN_RETRIES:-3}"
# Optional runtime overrides, typically set by GUI when Bundled runtime is selected.
REALESRGAN_BIN="${REALESRGAN_BIN:-}"
REALESRGAN_MODEL_DIR="${REALESRGAN_MODEL_DIR:-}"

# ---------------------------------
# Crash/kill retry policy (process)
# ---------------------------------
MAX_RETRIES="${MAX_RETRIES:-100}"      # set 0 for infinite retries
RETRY_SLEEP_SEC="${RETRY_SLEEP_SEC:-2}"
RETRY_CODES_DEFAULT="124 133 135 136 137 139 132 134"
RETRY_CODES="${RETRY_CODES:-$RETRY_CODES_DEFAULT}"

# Watchdog policy (output activity)
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
if [[ "$FFMPEG_CRF" =~ ^-?[0-9]+$ ]] && [[ "$FFMPEG_CRF" -ge 0 ]]; then
  CMD+=(--ffmpeg_crf "$FFMPEG_CRF")
fi
if [[ -n "${FFMPEG_PRESET// }" ]]; then
  CMD+=(--ffmpeg_preset "$FFMPEG_PRESET")
fi
if [[ -n "${FFMPEG_PIX_FMT// }" ]]; then
  CMD+=(--ffmpeg_pix_fmt "$FFMPEG_PIX_FMT")
fi
if [[ -n "${FFMPEG_EXTRA_ARGS// }" ]]; then
  CMD+=(--ffmpeg_extra_args "$FFMPEG_EXTRA_ARGS")
fi

echo "[CMD] ${CMD[*]}"

cleanup_runtime() {
  rm -f "$STOP_REQUEST_FILE" 2>/dev/null || true
  rm -f "$STOP_MARKER" 2>/dev/null || true
}

_latest_mp4_in_output() {
  find "$OUTPUT_DIR" -maxdepth 1 -type f -name "*.mp4" -printf '%T@|%p\n' 2>/dev/null \
    | sort -t'|' -nr -k1,1 \
    | head -n1 \
    | cut -d'|' -f2-
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
    | sort -t'|' -nr -k1,1 \
    | head -n1
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

_run_realesrgan_upscale() {
  local script_path="$REALESRGAN_UPSCALE_SCRIPT"
  local tmp_out="$OUTPUT_DIR/.tmp_esrgan_upscaled"

  if [[ -z "$script_path" || ! -f "$script_path" ]]; then
    echo "[ESRGAN] upscale script not found: $script_path"
    return 2
  fi

  rm -rf -- "$tmp_out" 2>/dev/null || true
  mkdir -p -- "$tmp_out"

  echo "[ESRGAN] running: $script_path"

  if [[ -n "${REALESRGAN_BIN// }" ]]; then
    export REALESRGAN_BIN
  fi
  if [[ -n "${REALESRGAN_MODEL_DIR// }" ]]; then
    export REALESRGAN_MODEL_DIR
  fi
  # Keep only encoder family alignment (NVENC vs x264).
  # Other encode knobs stay depthmap-optimized/hardcoded in ESRGAN scripts.
  if [[ -n "${FFMPEG_CODEC// }" ]]; then
    export REALESRGAN_OUT_CODEC="$FFMPEG_CODEC"
  fi

  if ! bash "$script_path" \
    "$OUTPUT_DIR" \
    "$tmp_out" \
    "$REALESRGAN_SCALE" \
    "$REALESRGAN_MODEL" \
    "$REALESRGAN_TILE" \
    "$REALESRGAN_DEST" \
    "$REALESRGAN_JOBS" \
    "$REALESRGAN_RETRIES"; then
    echo "[ESRGAN] upscale failed"
    rm -rf -- "$tmp_out" 2>/dev/null || true
    return 3
  fi

  shopt -s nullglob
  local up_files=("$tmp_out"/*.mp4)
  shopt -u nullglob
  if (( ${#up_files[@]} == 0 )); then
    echo "[ESRGAN] no upscaled outputs found in $tmp_out"
    rm -rf -- "$tmp_out" 2>/dev/null || true
    return 4
  fi

  local f base
  for f in "${up_files[@]}"; do
    base="$(basename "$f")"
    mv -f -- "$f" "$OUTPUT_DIR/$base"
  done
  rm -rf -- "$tmp_out" 2>/dev/null || true
  echo "[ESRGAN] upscale completed"
  return 0
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
  local code="$1"
  for c in $RETRY_CODES; do
    if [[ "$code" -eq "$c" ]]; then
      return 0
    fi
  done
  return 1
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
    if _is_true "$USE_REALESRGAN_UPSCALE"; then
      if ! _run_realesrgan_upscale; then
        echo "[FAIL] RealESRGAN upscale stage failed."
        exit 1
      fi
    fi
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
