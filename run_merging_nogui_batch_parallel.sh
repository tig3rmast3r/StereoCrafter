#!/usr/bin/env bash
set -euo pipefail

# --------------------------------------------
# User-editable parameters (PATHS ONLY)
# --------------------------------------------

PYTHON="${PYTHON:-python3}"

# Headless merging runner (the CT version also includes replace-mask streaming)
RUNNER="merging_nogui_batch_sharded.py"

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

# ---------------------------------
# Crash/kill retry policy (process)
# ---------------------------------
MAX_RETRIES="${MAX_RETRIES:-3}"       # total attempts for the whole run (python process)
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

# ------------------------------------------------
# Build command (keep the rest hardcoded in runner)
# ------------------------------------------------

CMD=(
  "$PYTHON" "$RUNNER"
  --inpainted-folder "$INPAINTED_FOLDER"
  --splatted-folder "$SPLATTED_FOLDER"
  --original-folder "$ORIGINAL_FOLDER"
  --output-folder "$OUTPUT_FOLDER"
)

# Replace-mask is optional; enable only if a non-empty folder is provided
if [ -n "${REPLACE_MASK_FOLDER// }" ]; then
  CMD+=(--use-replace-mask --replace-mask-folder "$REPLACE_MASK_FOLDER")
fi

echo "[BASE CMD] ${CMD[*]}"
echo "[PAR] WORKERS=$WORKERS  (override with WORKERS=N env var)"

run_worker_once() {
  local wid="$1"
  local cmdw=("${CMD[@]}" --num-workers "$WORKERS" --worker-id "$wid")

  echo "[CMD w$wid] ${cmdw[*]}"

  # Runner is headless (no Tk). Keep xvfb fallback just in case.
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
  while true; do
    echo "[RUN w$wid] attempt ${attempt}/${MAX_RETRIES}"
    set +e
    run_worker_once "$wid"
    local code=$?
    set -e

    if [ "$code" -eq 0 ]; then
      echo "[OK  w$wid] success"
      return 0
    fi

    if [ "$attempt" -ge "$MAX_RETRIES" ] || ! should_retry "$code"; then
      echo "[FAIL w$wid] exit_code=$code (no more retries)"
      return "$code"
    fi

    echo "[RETRY w$wid] exit_code=$code -> retrying in ${RETRY_SLEEP_SEC}s"
    sleep "$RETRY_SLEEP_SEC"
    attempt=$((attempt + 1))
  done
}

pids=()
wids=()

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
  set +e
  wait "$pid"
  code=$?
  set -e
  if [ "$code" -ne 0 ]; then
    echo "[DONE] worker $wid FAILED exit_code=$code (see merge_worker_${wid}.log)"
    fail=1
    fail_code="$code"
  else
    echo "[DONE] worker $wid OK"
  fi
done

if [ "$fail" -ne 0 ]; then
  exit "$fail_code"
fi

echo "[OK] all workers finished"
exit 0

