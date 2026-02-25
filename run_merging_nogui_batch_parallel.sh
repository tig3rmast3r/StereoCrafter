#!/usr/bin/env bash
set -euo pipefail

# --------------------------------------------
# User-editable parameters (PATHS ONLY)
# --------------------------------------------

PYTHON="${PYTHON:-python3}"

# Headless merging runner (the CT version also includes replace-mask streaming)
RUNNER="merging_nogui_batch_parallel.py"

# Folder containing the inpainted outputs (e.g. *_inpainted_right_eye.mp4 or *_inpainted_sbs.mp4)
INPAINTED_FOLDER="./work/output/"

# Folder containing splatted inputs (e.g. *_splatted2.mp4 / *_splatted4.mp4)
SPLATTED_FOLDER="./work/splat/hires/"

# Folder containing original/source clips (used for the left eye in QUAD or for ref)
ORIGINAL_FOLDER="./work/seg/"

# Output folder for merged results
OUTPUT_FOLDER="./work/sbs/"

# Folder containing replace-mask videos (e.g. *_splatted2_replace_mask.mkv)
# Leave empty to let the runner search next to each splatted file.
REPLACE_MASK_FOLDER="./work/mask/fixed/"

# Behavior toggles (non-path)
CT_PRESET="${CT_PRESET:-1}"               # 1..8 or full preset label
AUTO_CT_EVAL="${AUTO_CT_EVAL:-1}"         # 1=enable per-frame auto preset evaluation
ENABLE_COLOR_TRANSFER="${ENABLE_COLOR_TRANSFER:-1}"  # 1=on,0=off
ADD_BORDERS="${ADD_BORDERS:-0}"           # 1=apply sidecar borders, 0=disable
MERGE_DEBUG="${MERGE_DEBUG:-0}"           # 1=enable Python debug mode/logging

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
RETRY_CODES_DEFAULT="137 139 132 134"
RETRY_CODES="${RETRY_CODES:-$RETRY_CODES_DEFAULT}"

# Number of parallel worker processes (each handles a deterministic slice of files)
WORKERS="${WORKERS:-2}"
STOP_MARKER="${STOP_MARKER:-$OUTPUT_FOLDER/.stop_after_current}"
STOP_REQUEST_FILE="${TMPDIR:-/tmp}/merge_stop_request_${$}.flag"
INTERRUPT_COUNT=0
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
  --ct-preset "$CT_PRESET"
)

# Replace-mask is optional; enable only if a non-empty folder is provided
if [ -n "${REPLACE_MASK_FOLDER// }" ]; then
  CMD+=(--use-replace-mask --replace-mask-folder "$REPLACE_MASK_FOLDER")
fi
if [ "${AUTO_CT_EVAL}" = "1" ]; then
  CMD+=(--auto-ct-eval)
else
  CMD+=(--no-auto-ct-eval)
fi
if [ "${ENABLE_COLOR_TRANSFER}" != "1" ]; then
  CMD+=(--no-color-transfer)
fi
if [ "${ADD_BORDERS}" = "1" ]; then
  CMD+=(--add-borders)
else
  CMD+=(--no-add-borders)
fi

export MERGE_DEBUG

echo "[BASE CMD] ${CMD[*]}"
echo "[PAR] WORKERS=$WORKERS  (override with WORKERS=N env var)"
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

  # Worker inherits ignored INT/TERM from run_worker_with_retries trap.
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
  # Parent handles Ctrl+C and writes stop marker; workers must not handle TTY INT directly.
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
  # Per-worker logs (optional). Comment out if you don't want them.
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
