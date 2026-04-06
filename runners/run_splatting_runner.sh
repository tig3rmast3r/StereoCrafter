#!/usr/bin/env bash
set -euo pipefail

# --------------------------------------------
# User-editable parameters (env-overridable)
# --------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-python3}"
RUNNER="${RUNNER:-$SCRIPT_DIR/batch_splatting_runner.py}"

INPUT_SOURCE_CLIPS="${INPUT_SOURCE_CLIPS:-./work/seg/}"
INPUT_DEPTH_MAPS="${INPUT_DEPTH_MAPS:-./work/depthmap/}"
OUTPUT_SPLATTED="${OUTPUT_SPLATTED:-./work/splat/}"
MASK_OUTPUT="${MASK_OUTPUT:-./work/mask/}"

FULL_RES_BATCH_SIZE="${FULL_RES_BATCH_SIZE:-50}"
DISPARITY="${DISPARITY:-20}"
OUTPUT_LAYOUT="${OUTPUT_LAYOUT:-single_warp}"
AUTO_CONVERGENCE_MODE="${AUTO_CONVERGENCE_MODE:-MinBorders}"
CONVERGENCE="${CONVERGENCE:-50}"

DILATE_X="${DILATE_X:-1}"
DILATE_Y="${DILATE_Y:-1}"
BLUR_X="${BLUR_X:-0}"
BLUR_Y="${BLUR_Y:-0}"
DILATE_LEFT="${DILATE_LEFT:-2}"
BLUR_BALANCE="${BLUR_BALANCE:-0.5}"
GAMMA="${GAMMA:-1}"

STAIR_SMOOTH="${STAIR_SMOOTH:-1}"
STAIR_SMOOTH_KERNEL="${STAIR_SMOOTH_KERNEL:-3}"
STAIR_SMOOTH_X_OFF="${STAIR_SMOOTH_X_OFF:-2}"
STAIR_SMOOTH_STRIP="${STAIR_SMOOTH_STRIP:-3}"
STAIR_SMOOTH_STRENGTH="${STAIR_SMOOTH_STRENGTH:-1}"

USE_REPLACE_MASK="${USE_REPLACE_MASK:-1}"
REPLACE_MASK_SCALE="${REPLACE_MASK_SCALE:-1}"
REPLACE_MASK_MIN="${REPLACE_MASK_MIN:-1}"
REPLACE_MASK_MAX="${REPLACE_MASK_MAX:-32}"
REPLACE_MASK_GAP="${REPLACE_MASK_GAP:-0}"
REPLACE_MASK_EDGE="${REPLACE_MASK_EDGE:-0}"
REPLACE_MASK_CODEC="${REPLACE_MASK_CODEC:-ffv1}"

ENABLE_FULL_RES="${ENABLE_FULL_RES:-True}"
ENABLE_LOW_RES="${ENABLE_LOW_RES:-False}"
PROCESS_LENGTH="${PROCESS_LENGTH:--1}"

FFMPEG_CODEC="${FFMPEG_CODEC:-}"
ENCODER_MODE="${ENCODER_MODE:-}"
FFMPEG_EXTRA_ARGS="${FFMPEG_EXTRA_ARGS:-}"

# Retry/stop policy
MAX_RETRIES="${MAX_RETRIES:-100}"          # 0 => infinite
RETRY_SLEEP_SEC="${RETRY_SLEEP_SEC:-2}"
RETRY_CODES_DEFAULT="133 135 136 137 139 132 134"
RETRY_CODES="${RETRY_CODES:-$RETRY_CODES_DEFAULT}"

STOP_MARKER="${STOP_MARKER:-$OUTPUT_SPLATTED/.stop_after_current}"
STOP_REQUEST_FILE="${TMPDIR:-/tmp}/splat_stop_request_${$}.flag"

INTERRUPT_COUNT=0
STOP_REQUESTED=0
FORCE_STOP=0
CURRENT_PID=""

CMD=(
  "$PYTHON" "$RUNNER"
  --input_source_clips "$INPUT_SOURCE_CLIPS"
  --input_depth_maps "$INPUT_DEPTH_MAPS"
  --output_splatted "$OUTPUT_SPLATTED"
  --mask_output "$MASK_OUTPUT"
  --full_res_batch_size "$FULL_RES_BATCH_SIZE"
  --disparity "$DISPARITY"
  --process_length "$PROCESS_LENGTH"
  --enable_full_res "$ENABLE_FULL_RES"
  --enable_low_res "$ENABLE_LOW_RES"
  --output_layout "$OUTPUT_LAYOUT"
  --auto_convergence_mode "$AUTO_CONVERGENCE_MODE"
  --convergence "$CONVERGENCE"
  --dilate_x "$DILATE_X"
  --dilate_y "$DILATE_Y"
  --blur_x "$BLUR_X"
  --blur_y "$BLUR_Y"
  --dilate_left "$DILATE_LEFT"
  --blur_balance "$BLUR_BALANCE"
  --gamma "$GAMMA"
  --stair_smooth "$STAIR_SMOOTH"
  --stair_smooth_kernel "$STAIR_SMOOTH_KERNEL"
  --stair_smooth_x_off "$STAIR_SMOOTH_X_OFF"
  --stair_smooth_strip "$STAIR_SMOOTH_STRIP"
  --stair_smooth_strength "$STAIR_SMOOTH_STRENGTH"
  --use_replace_mask "$USE_REPLACE_MASK"
  --replace_mask_scale "$REPLACE_MASK_SCALE"
  --replace_mask_min "$REPLACE_MASK_MIN"
  --replace_mask_max "$REPLACE_MASK_MAX"
  --replace_mask_gap "$REPLACE_MASK_GAP"
  --replace_mask_edge "$REPLACE_MASK_EDGE"
  --replace_mask_codec "$REPLACE_MASK_CODEC"
  --stop_marker "$STOP_MARKER"
)

if [[ -n "${FFMPEG_CODEC// }" ]]; then
  CMD+=(--ffmpeg_codec "$FFMPEG_CODEC")
fi
if [[ -n "${ENCODER_MODE// }" ]]; then
  CMD+=(--encoder_mode "$ENCODER_MODE")
fi
if [[ -n "${FFMPEG_EXTRA_ARGS// }" ]]; then
  CMD+=(--ffmpeg_extra_args "$FFMPEG_EXTRA_ARGS")
fi

echo "[CMD] ${CMD[*]}"

cleanup_runtime() {
  rm -f -- "$STOP_REQUEST_FILE" 2>/dev/null || true
  rm -f -- "$STOP_MARKER" 2>/dev/null || true
}

if [[ -f "$STOP_MARKER" ]]; then
  echo "[INFO] removing stale stop marker: $STOP_MARKER"
  rm -f -- "$STOP_MARKER" || true
fi

signal_descendants() {
  local sig="$1"
  local parent="$2"
  if ! command -v pgrep >/dev/null 2>&1; then
    return
  fi
  local kids
  kids="$(pgrep -P "$parent" 2>/dev/null || true)"
  for kid in $kids; do
    signal_descendants "$sig" "$kid"
    kill "-$sig" "$kid" 2>/dev/null || true
  done
}

signal_tree() {
  local sig="$1"
  local pid="$2"
  if [[ -z "${pid:-}" ]]; then
    return
  fi
  signal_descendants "$sig" "$pid"
  kill "-$sig" "$pid" 2>/dev/null || true
}

wait_for_pid() {
  local pid="$1"
  local code=0
  while true; do
    if wait "$pid"; then
      code=0
    else
      code=$?
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
      break
    fi
    sleep 0.1
  done
  return "$code"
}

request_graceful_stop() {
  if [[ "$STOP_REQUESTED" -eq 1 ]]; then
    return
  fi
  STOP_REQUESTED=1
  local marker_dir
  marker_dir="$(dirname "$STOP_MARKER")"
  mkdir -p "$marker_dir" 2>/dev/null || true
  : > "$STOP_MARKER"
  : > "$STOP_REQUEST_FILE"
  echo "[STOP] graceful stop requested. Finishing current clip before exiting."
}

request_forced_stop() {
  FORCE_STOP=1
  echo "[STOP] force stop requested. Killing runner immediately."
  if [[ -n "${CURRENT_PID:-}" ]]; then
    signal_tree KILL "$CURRENT_PID"
  fi
  exit 130
}

on_interrupt() {
  INTERRUPT_COUNT=$((INTERRUPT_COUNT + 1))
  if [[ "$INTERRUPT_COUNT" -eq 1 ]]; then
    request_graceful_stop
  else
    request_forced_stop
  fi
}

trap on_interrupt INT TERM
trap cleanup_runtime EXIT

run_once() {
  if [[ -f "$STOP_REQUEST_FILE" ]]; then
    return 0
  fi

  (trap '' INT TERM; "${CMD[@]}") &

  CURRENT_PID="$!"
  wait_for_pid "$CURRENT_PID"
  local code=$?
  CURRENT_PID=""
  return "$code"
}

should_retry() {
  local _code="${1:-1}"
  # Retry on any non-zero exit code; MAX_RETRIES still applies.
  return 0
}

attempt=1
while true; do
  if [[ -f "$STOP_REQUEST_FILE" ]]; then
    echo "[STOP] graceful stop completed"
    exit 0
  fi

  echo "[RUN ] attempt ${attempt}/${MAX_RETRIES}"
  set +e
  run_once
  code=$?
  set -e

  if [[ "$FORCE_STOP" -eq 1 ]]; then
    exit 130
  fi

  if [[ "$STOP_REQUESTED" -eq 1 ]] || [[ -f "$STOP_REQUEST_FILE" ]]; then
    echo "[STOP] graceful stop completed (last rc=$code)"
    exit 0
  fi

  if [[ "$code" -eq 0 ]]; then
    echo "[OK  ] success"
    exit 0
  fi

  if [[ "$code" -eq 130 ]]; then
    echo "[STOP] interrupted by user"
    exit 130
  fi

  if [[ "$MAX_RETRIES" -ne 0 && "$attempt" -ge "$MAX_RETRIES" ]]; then
    echo "[FAIL] reached MAX_RETRIES=$MAX_RETRIES (last rc=$code)"
    exit "$code"
  fi

  if ! should_retry "$code"; then
    echo "[FAIL] exit_code=$code is not retryable"
    exit "$code"
  fi

  echo "[RETRY] exit_code=$code -> retrying in ${RETRY_SLEEP_SEC}s"
  for ((s=0; s<RETRY_SLEEP_SEC; s++)); do
    if [[ "$STOP_REQUESTED" -eq 1 ]] || [[ -f "$STOP_REQUEST_FILE" ]]; then
      echo "[STOP] graceful stop completed"
      exit 0
    fi
    sleep 1
  done
  attempt=$((attempt + 1))
done
