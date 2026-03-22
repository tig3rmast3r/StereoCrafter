#!/usr/bin/env bash
set -euo pipefail

# --------------------------------------------
# User-editable parameters (PATHS ONLY)
# --------------------------------------------

PYTHON="${PYTHON:-python3}"

# Headless merging runner (the CT version also includes replace-mask streaming)
RUNNER="${RUNNER:-merging_nogui_batch.py}"

# Folder containing the inpainted outputs (e.g. *_inpainted_right_eye.mp4 or *_inpainted_sbs.mp4)
INPAINTED_FOLDER="${INPAINTED_FOLDER:-./work/output/}"
PREFERRED_INPAINTED_FOLDER="${PREFERRED_INPAINTED_FOLDER:-}"

# Folder containing splatted inputs (e.g. *_splatted1.mp4 / *_splatted2.mp4 / *_splatted4.mp4)
SPLATTED_FOLDER="${SPLATTED_FOLDER:-./work/splat/hires/}"

# Folder containing original/source clips (used for the left eye in QUAD or for ref)
ORIGINAL_FOLDER="${ORIGINAL_FOLDER:-./work/seg/}"

# Output folder for merged results
OUTPUT_FOLDER="${OUTPUT_FOLDER:-./work/sbs/}"

# Folder containing replace-mask videos (e.g. *_splatted1_replace_mask.mkv / *_splatted2_replace_mask.mkv)
# Leave empty to let the runner search next to each splatted file.
REPLACE_MASK_FOLDER="${REPLACE_MASK_FOLDER:-./work/mask/}"

# Behavior toggles (non-path)
CT_PRESET="${CT_PRESET:-1}"               # 1..8 or full preset label
CT_AUTO_MODE="${CT_AUTO_MODE:-On}"        # Off | On | CSV Blend
CT_CSV_BLEND_PATH="${CT_CSV_BLEND_PATH:-./autoct.csv}"
ENABLE_COLOR_TRANSFER="${ENABLE_COLOR_TRANSFER:-1}"  # 1=on,0=off
ADD_BORDERS="${ADD_BORDERS:-0}"           # 1=apply sidecar borders, 0=disable
PAD_TO_16_9="${PAD_TO_16_9:-0}"           # 1=pad to 16:9, 0=disable
MERGE_DEBUG="${MERGE_DEBUG:-0}"           # 1=enable Python debug mode/logging
PREPROCESSED_MASK_FOLDER="${PREPROCESSED_MASK_FOLDER:-./work/mask_for_merge/}"
OUTPUT_FORMAT="${OUTPUT_FORMAT:-Full SBS (Left-Right)}"
CHUNK_SIZE="${CHUNK_SIZE:-20}"
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
RESTART_EVERY="${RESTART_EVERY:-1}"
PLANNED_RESTART_CODE="${PLANNED_RESTART_CODE:-99}"

# ---------------------------------
# Crash/kill retry policy (process)
# ---------------------------------
MAX_RETRIES="${MAX_RETRIES:-100}"       # total attempts for the whole run (python process)
RETRY_SLEEP_SEC="${RETRY_SLEEP_SEC:-2}"

# Exit codes commonly seen when the process dies outside Python:
# 137 = killed (SIGKILL, often OOM)
# 139 = segfault
# 132 = illegal instruction
# 134 = abort
RETRY_CODES_DEFAULT="132  133 135 136 137 139 132 134"
RETRY_CODES="${RETRY_CODES:-$RETRY_CODES_DEFAULT}"
STOP_MARKER="${STOP_MARKER:-$OUTPUT_FOLDER/.stop_after_current}"
STOP_REQUEST_FILE="${TMPDIR:-/tmp}/merge_stop_request_${$}.flag"
INTERRUPT_COUNT=0
CURRENT_PID=""
STOP_REQUESTED=0
FORCE_STOP=0

# ------------------------------------------------
# Build command (keep the rest hardcoded in runner)
# ------------------------------------------------

CMD=(
  "$PYTHON" "$RUNNER"
  --inpainted-folder "$INPAINTED_FOLDER"
  --splatted-folder "$SPLATTED_FOLDER"
  --original-folder "$ORIGINAL_FOLDER"
  --output-folder "$OUTPUT_FOLDER"
  --stop-marker "$STOP_MARKER"
  --restart-every "$RESTART_EVERY"
  --output-format "$OUTPUT_FORMAT"
  --chunk-size "$CHUNK_SIZE"
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

if [ -n "${PREFERRED_INPAINTED_FOLDER// }" ] && [ -d "$PREFERRED_INPAINTED_FOLDER" ]; then
  CMD+=(--preferred-inpainted-folder "$PREFERRED_INPAINTED_FOLDER")
fi

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

echo "[CMD] ${CMD[*]}"
echo "[DBG] MERGE_DEBUG=$MERGE_DEBUG"

cleanup_runtime() {
  if [ -n "${CURRENT_PID:-}" ] && kill -0 "$CURRENT_PID" 2>/dev/null; then
    signal_tree KILL "$CURRENT_PID"
  fi
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
  echo "[STOP] force stop requested. Killing runner immediately."
  if [ -n "${CURRENT_PID:-}" ]; then
    signal_tree KILL "$CURRENT_PID"
  fi
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

run_once() {
  if [ -f "$STOP_REQUEST_FILE" ]; then
    return 0
  fi

  # Runner is headless (no Tk). Keep xvfb fallback just in case.
  if [ -z "${DISPLAY:-}" ] && command -v xvfb-run >/dev/null 2>&1; then
    xvfb-run -a "${CMD[@]}" &
  else
    "${CMD[@]}" &
  fi
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
  if [ -f "$STOP_REQUEST_FILE" ]; then
    echo "[STOP] graceful stop completed"
    exit 0
  fi

  echo "[RUN ] attempt ${attempt}/${MAX_RETRIES}"
  set +e
  run_once
  code=$?
  set -e

  if [ "$FORCE_STOP" -eq 1 ]; then
    exit 130
  fi

  if [ "$STOP_REQUESTED" -eq 1 ] || [ -f "$STOP_REQUEST_FILE" ]; then
    echo "[STOP] graceful stop completed (last rc=$code)"
    exit 0
  fi

  if [ "$code" -eq "$PLANNED_RESTART_CODE" ]; then
    echo "[RESTART] planned process restart requested -> relaunching immediately"
    continue
  fi

  if [ "$code" -eq 0 ] && [ ! -f "$STOP_MARKER" ]; then
    echo "[OK  ] success"
    exit 0
  fi

  if [ "$code" -eq 130 ]; then
    echo "[STOP] interrupted by user"
    exit 130
  fi

  if [ "$attempt" -ge "$MAX_RETRIES" ] || ! should_retry "$code"; then
    echo "[FAIL] exit_code=$code (no more retries)"
    exit "$code"
  fi

  echo "[RETRY] exit_code=$code -> retrying in ${RETRY_SLEEP_SEC}s"
  for ((s=0; s<RETRY_SLEEP_SEC; s++)); do
    if [ "$STOP_REQUESTED" -eq 1 ] || [ -f "$STOP_REQUEST_FILE" ]; then
      echo "[STOP] graceful stop completed"
      exit 0
    fi
    sleep 1
  done
  attempt=$((attempt + 1))
done
