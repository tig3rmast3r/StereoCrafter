#!/usr/bin/env bash
set -euo pipefail

# --------------------------------------------
# User-editable parameters (paths + batch size)
# --------------------------------------------

PYTHON="${PYTHON:-python3}"

RUNNER="batch_splatting_bm_runner.py"
GUI_SCRIPT="splatting_bm_gui.py"

INPUT_SOURCE_CLIPS="./work/seg/"
INPUT_DEPTH_MAPS="./work/depthmap/upscaled/"
OUTPUT_SPLATTED="./work/splat/"
MASK_OUTPUT="./work/mask/"

FULL_RES_BATCH_SIZE=50

# -----------------------
# Crash/kill retry policy
# -----------------------
MAX_RETRIES="${MAX_RETRIES:-3}"     # total attempts = MAX_RETRIES
RETRY_SLEEP_SEC="${RETRY_SLEEP_SEC:-2}"

# Exit codes commonly seen when the process dies outside Python:
# 137 = killed (SIGKILL, often OOM)
# 139 = segfault
# 132 = illegal instruction
# 134 = abort
RETRY_CODES_DEFAULT="137 139 132 134"
RETRY_CODES="${RETRY_CODES:-$RETRY_CODES_DEFAULT}"

# ------------------------------------------------
# Build command (keep the rest hardcoded in runner)
# ------------------------------------------------

CMD=(
  "$PYTHON" "$RUNNER"
  --gui_script "$GUI_SCRIPT"
  --input_source_clips "$INPUT_SOURCE_CLIPS"
  --input_depth_maps "$INPUT_DEPTH_MAPS"
  --output_splatted "$OUTPUT_SPLATTED"
  --mask_output "$MASK_OUTPUT"
  --full_res_batch_size "$FULL_RES_BATCH_SIZE"
)

echo "[CMD] ${CMD[*]}"

run_once() {
  # Tk requires a display. If DISPLAY is missing (headless), run under xvfb when available.
  if [ -z "${DISPLAY:-}" ] && command -v xvfb-run >/dev/null 2>&1; then
    xvfb-run -a "${CMD[@]}"
  else
    "${CMD[@]}"
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

attempt=1
while true; do
  echo "[RUN ] attempt ${attempt}/${MAX_RETRIES}"
  set +e
  run_once
  code=$?
  set -e

  if [ "$code" -eq 0 ]; then
    echo "[OK  ] success"
    exit 0
  fi

  if [ "$attempt" -ge "$MAX_RETRIES" ] || ! should_retry "$code"; then
    echo "[FAIL] exit_code=$code (no more retries)"
    exit "$code"
  fi

  echo "[RETRY] exit_code=$code -> retrying in ${RETRY_SLEEP_SEC}s"
  sleep "$RETRY_SLEEP_SEC"
  attempt=$((attempt + 1))
done
