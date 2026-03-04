#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python3}"
RUNNER="${RUNNER:-merging_nogui_batch_parallel.py}"

INPAINTED_FOLDER="${INPAINTED_FOLDER:-./work/output/}"
SPLATTED_FOLDER="${SPLATTED_FOLDER:-./work/splat/hires/}"
ORIGINAL_FOLDER="${ORIGINAL_FOLDER:-./work/seg/}"
OUTPUT_FOLDER="${OUTPUT_FOLDER:-./work/sbs/}"
REPLACE_MASK_FOLDER="${REPLACE_MASK_FOLDER:-./work/mask/}"

CT_PRESET="${CT_PRESET:-1}"
CT_AUTO_MODE="${CT_AUTO_MODE:-CSV Blend}"
CT_CSV_BLEND_PATH="${CT_CSV_BLEND_PATH:-./autoct.csv}"
ENABLE_COLOR_TRANSFER="${ENABLE_COLOR_TRANSFER:-1}"
ADD_BORDERS="${ADD_BORDERS:-0}"
PAD_TO_16_9="${PAD_TO_16_9:-0}"
MERGE_DEBUG="${MERGE_DEBUG:-0}"
PREPROCESSED_MASK_FOLDER="${PREPROCESSED_MASK_FOLDER:-./work/mask_for_merge/}"
OUTPUT_FORMAT="${OUTPUT_FORMAT:-Full SBS (Left-Right)}"
CHUNK_SIZE="${CHUNK_SIZE:-20}"
READER_RELOAD_EVERY_CHUNKS="${READER_RELOAD_EVERY_CHUNKS:-1}"
USE_GPU="${USE_GPU:-0}"
CT_STRENGTH="${CT_STRENGTH:-1}"
CT_BLACK_THRESH="${CT_BLACK_THRESH:-0}"
CT_MIN_VALID_RATIO="${CT_MIN_VALID_RATIO:-0}"
CT_MIN_VALID="${CT_MIN_VALID:-0}"
CT_RING_WIDTH="${CT_RING_WIDTH:-20}"
CT_CLAMP_L_MIN="${CT_CLAMP_L_MIN:-0.1}"
CT_CLAMP_L_MAX="${CT_CLAMP_L_MAX:-2}"
CT_CLAMP_AB_MIN="${CT_CLAMP_AB_MIN:-0.1}"
CT_CLAMP_AB_MAX="${CT_CLAMP_AB_MAX:-3}"
CT_EXCLUDE_BLACK_IN_TARGET="${CT_EXCLUDE_BLACK_IN_TARGET:-1}"
FFMPEG_CODEC="${FFMPEG_CODEC:-}"
FFMPEG_CRF="${FFMPEG_CRF:-}"
FFMPEG_PRESET="${FFMPEG_PRESET:-}"
FFMPEG_PIX_FMT="${FFMPEG_PIX_FMT:-}"
FFMPEG_EXTRA_ARGS="${FFMPEG_EXTRA_ARGS:-}"

MAX_RETRIES="${MAX_RETRIES:-100}"
RETRY_SLEEP_SEC="${RETRY_SLEEP_SEC:-2}"
RETRY_CODES_DEFAULT="132 133 135 136 137 139 132 134"
RETRY_CODES="${RETRY_CODES:-$RETRY_CODES_DEFAULT}"

WORKERS="${WORKERS:-2}"
STOP_MARKER="${STOP_MARKER:-$OUTPUT_FOLDER/.stop_after_current}"
STOP_REQUEST_FILE="${TMPDIR:-/tmp}/merge_stop_request_${$}.flag"
INTERRUPT_COUNT=0
STOP_REQUESTED=0
FORCE_STOP=0
pids=()
wids=()

CMD=(
  "$PYTHON" "$RUNNER"
  --inpainted-folder "$INPAINTED_FOLDER"
  --splatted-folder "$SPLATTED_FOLDER"
  --original-folder "$ORIGINAL_FOLDER"
  --output-folder "$OUTPUT_FOLDER"
  --stop-marker "$STOP_MARKER"
  --output-format "$OUTPUT_FORMAT"
  --chunk-size "$CHUNK_SIZE"
  --reader-reload-every-chunks "$READER_RELOAD_EVERY_CHUNKS"
  --ct-preset "$CT_PRESET"
  --ct-auto-mode "$CT_AUTO_MODE"
  --ct-strength "$CT_STRENGTH"
  --ct-black-thresh "$CT_BLACK_THRESH"
  --ct-min-valid-ratio "$CT_MIN_VALID_RATIO"
  --ct-min-valid "$CT_MIN_VALID"
  --ct-clamp-L-min "$CT_CLAMP_L_MIN"
  --ct-clamp-L-max "$CT_CLAMP_L_MAX"
  --ct-clamp-ab-min "$CT_CLAMP_AB_MIN"
  --ct-clamp-ab-max "$CT_CLAMP_AB_MAX"
  --ct-ring-width "$CT_RING_WIDTH"
)

if [ -z "${REPLACE_MASK_FOLDER// }" ] || [ ! -d "$REPLACE_MASK_FOLDER" ]; then
  echo "[ERR ] replace-mask folder missing: ${REPLACE_MASK_FOLDER:-<empty>}"
  exit 2
fi
if [ -z "${PREPROCESSED_MASK_FOLDER// }" ] || [ ! -d "$PREPROCESSED_MASK_FOLDER" ]; then
  echo "[ERR ] preprocessed mask_for_merge folder missing: ${PREPROCESSED_MASK_FOLDER:-<empty>}"
  exit 2
fi
CMD+=(--replace-mask-folder "$REPLACE_MASK_FOLDER")
CMD+=(--preprocessed-mask-folder "$PREPROCESSED_MASK_FOLDER")
if [ "${CT_AUTO_MODE}" = "CSV Blend" ] && [ -n "${CT_CSV_BLEND_PATH// }" ]; then
  CMD+=(--ct-csv-blend-path "$CT_CSV_BLEND_PATH")
fi
if [ "${ENABLE_COLOR_TRANSFER}" != "1" ]; then
  CMD+=(--no-color-transfer)
fi
if [ "${ADD_BORDERS}" = "1" ]; then
  CMD+=(--add-borders)
else
  CMD+=(--no-add-borders)
fi
if [ "${PAD_TO_16_9}" = "1" ]; then
  CMD+=(--pad-to-16-9)
else
  CMD+=(--no-pad-to-16-9)
fi
if [ "${USE_GPU}" = "1" ]; then
  CMD+=(--use-gpu)
else
  CMD+=(--no-use-gpu)
fi
if [ "${CT_EXCLUDE_BLACK_IN_TARGET}" = "1" ]; then
  CMD+=(--ct-exclude-black-in-target)
else
  CMD+=(--no-ct-exclude-black-in-target)
fi
if [ -n "${FFMPEG_CODEC// }" ]; then
  CMD+=(--ffmpeg-codec "$FFMPEG_CODEC")
fi
if [ -n "${FFMPEG_CRF// }" ]; then
  CMD+=(--ffmpeg-crf "$FFMPEG_CRF")
fi
if [ -n "${FFMPEG_PRESET// }" ]; then
  CMD+=(--ffmpeg-preset "$FFMPEG_PRESET")
fi
if [ -n "${FFMPEG_PIX_FMT// }" ]; then
  CMD+=(--ffmpeg-pix-fmt "$FFMPEG_PIX_FMT")
fi
if [ -n "${FFMPEG_EXTRA_ARGS// }" ]; then
  CMD+=(--ffmpeg-extra-args "$FFMPEG_EXTRA_ARGS")
fi

export MERGE_DEBUG

echo "[BASE CMD] ${CMD[*]}"
echo "[PAR] WORKERS=$WORKERS"
echo "[DBG] MERGE_DEBUG=$MERGE_DEBUG"

cleanup_runtime() {
  rm -f "$STOP_MARKER" 2>/dev/null || true
  rm -f "$STOP_REQUEST_FILE" 2>/dev/null || true
}

if [ -f "$STOP_MARKER" ]; then
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
  if [ -z "${pid:-}" ]; then
    return
  fi
  signal_descendants "$sig" "$pid"
  kill "-$sig" "$pid" 2>/dev/null || true
}

wait_for_pid() {
  local pid="$1"
  local code=0
  while true; do
    set +e
    wait "$pid"
    code=$?
    set -e
    if ! kill -0 "$pid" 2>/dev/null; then
      break
    fi
    sleep 0.1
  done
  return "$code"
}

run_worker_once() {
  local wid="$1"
  if [ -f "$STOP_REQUEST_FILE" ]; then
    return 0
  fi
  local cmdw=("${CMD[@]}" --num-workers "$WORKERS" --worker-id "$wid")
  echo "[CMD w$wid] ${cmdw[*]}"
  if [ -z "${DISPLAY:-}" ] && command -v xvfb-run >/dev/null 2>&1; then
    xvfb-run -a "${cmdw[@]}"
  else
    "${cmdw[@]}"
  fi
}

should_retry() {
  local code="$1"
  for c in $RETRY_CODES; do
    if [ "$code" -eq "$c" ]; then
      return 0
    fi
  done
  return 1
}

run_worker_with_retries() {
  local wid="$1"
  local attempt=1
  trap '' INT TERM
  while true; do
    if [ -f "$STOP_REQUEST_FILE" ]; then
      echo "[STOP w$wid] stop marker detected before next attempt"
      return 0
    fi

    echo "[RUN w$wid] attempt ${attempt}/${MAX_RETRIES}"
    set +e
    run_worker_once "$wid"
    local code=$?
    set -e

    if [ "$code" -eq 0 ] && [ ! -f "$STOP_REQUEST_FILE" ]; then
      echo "[OK  w$wid] success"
      return 0
    fi
    if [ -f "$STOP_REQUEST_FILE" ]; then
      echo "[STOP w$wid] graceful stop completed (last rc=$code)"
      return 0
    fi
    if [ "$code" -eq 130 ]; then
      echo "[STOP w$wid] interrupted by user"
      return 130
    fi
    if [ "$attempt" -ge "$MAX_RETRIES" ] || ! should_retry "$code"; then
      echo "[FAIL w$wid] exit_code=$code (no more retries)"
      return "$code"
    fi

    echo "[RETRY w$wid] exit_code=$code -> retrying in ${RETRY_SLEEP_SEC}s"
    for ((s=0; s<RETRY_SLEEP_SEC; s++)); do
      if [ -f "$STOP_REQUEST_FILE" ]; then
        echo "[STOP w$wid] graceful stop completed"
        return 0
      fi
      sleep 1
    done
    attempt=$((attempt + 1))
  done
}

request_graceful_stop() {
  if [ "$STOP_REQUESTED" -eq 1 ]; then
    return
  fi
  STOP_REQUESTED=1
  local marker_dir
  marker_dir="$(dirname "$STOP_MARKER")"
  mkdir -p "$marker_dir" 2>/dev/null || true
  : > "$STOP_MARKER"
  : > "$STOP_REQUEST_FILE"
  echo "[STOP] graceful stop requested. Finishing current file before exiting."
}

request_forced_stop() {
  FORCE_STOP=1
  echo "[STOP] force stop requested. Killing runners immediately."
  for pid in "${pids[@]:-}"; do
    signal_tree KILL "$pid"
  done
  exit 130
}

on_interrupt() {
  INTERRUPT_COUNT=$((INTERRUPT_COUNT + 1))
  if [ "$INTERRUPT_COUNT" -eq 1 ]; then
    request_graceful_stop
  else
    request_forced_stop
  fi
}

trap on_interrupt INT TERM
trap cleanup_runtime EXIT

for ((wid=0; wid<WORKERS; wid++)); do
  wids+=("$wid")
  run_worker_with_retries "$wid" > "merge_worker_${wid}.log" 2>&1 &
  pids+=("$!")
  echo "[START] worker $wid pid=${pids[-1]} log=merge_worker_${wid}.log"
done

fail=0
fail_code=0
for i in "${!pids[@]}"; do
  pid="${pids[$i]}"
  wid="${wids[$i]}"
  wait_for_pid "$pid"
  code=$?
  if [ "$code" -ne 0 ]; then
    echo "[DONE] worker $wid FAILED exit_code=$code (see merge_worker_${wid}.log)"
    fail=1
    if [ "$fail_code" -eq 0 ]; then
      fail_code="$code"
    fi
  else
    echo "[DONE] worker $wid OK"
  fi
done

if [ "$fail" -ne 0 ]; then
  if [ "$STOP_REQUESTED" -eq 1 ] || [ -f "$STOP_REQUEST_FILE" ]; then
    echo "[STOP] graceful stop completed"
    exit 0
  fi
  exit "$fail_code"
fi

if [ "$STOP_REQUESTED" -eq 1 ] || [ -f "$STOP_REQUEST_FILE" ]; then
  echo "[STOP] graceful stop completed"
  exit 0
fi

echo "[OK] all workers finished"
exit 0
