#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python3}"
RUNNER="${RUNNER:-mask_formerge_nogui.py}"

# Input replace-mask folder from merge defaults
REPLACE_MASK_FOLDER="${REPLACE_MASK_FOLDER:-./work/mask/}"
OUTPUT_FOLDER="${OUTPUT_FOLDER:-./work/mask_for_merge/}"
INPUT_GLOB="${INPUT_GLOB:-*_replace_mask.*}"

# Parallelism: one python process at a time per worker, deterministic file sharding.
WORKERS="${WORKERS:-8}"

# Defaults aligned with run_merging_nogui_batch.sh
CHUNK_SIZE="${CHUNK_SIZE:-20}"
USE_GPU="${USE_GPU:-0}"
MASK_BINARIZE_THRESHOLD="${MASK_BINARIZE_THRESHOLD:-0.5}"
MASK_DILATE_KERNEL_SIZE="${MASK_DILATE_KERNEL_SIZE:-2}"
MASK_BLUR_KERNEL_SIZE="${MASK_BLUR_KERNEL_SIZE:-4}"
SHADOW_LENGTH_PX="${SHADOW_LENGTH_PX:-25}"
SHADOW_CURVE="${SHADOW_CURVE:-0}"
SHADOW_MOTION_GAIN="${SHADOW_MOTION_GAIN:-1}"
SHADOW_MOTION_DEADZONE_PX="${SHADOW_MOTION_DEADZONE_PX:-20}"
SHADOW_MOTION_MAX_PX="${SHADOW_MOTION_MAX_PX:-40}"
SHADOW_MOTION_CHAIN_ENABLED="${SHADOW_MOTION_CHAIN_ENABLED:-1}"
SHADOW_AREA_MIN_PX="${SHADOW_AREA_MIN_PX:-0}"
SHADOW_AREA_MAX_PX="${SHADOW_AREA_MAX_PX:-0}"
SHADOW_AREA_RESET_RATIO="${SHADOW_AREA_RESET_RATIO:-1.8}"
SHADOW_AREA_RESET_ABS_PX="${SHADOW_AREA_RESET_ABS_PX:-0}"
SHADOW_COMPONENT_MERGE_Y_TOL_PX="${SHADOW_COMPONENT_MERGE_Y_TOL_PX:-0}"
SHADOW_ALPHA_DOWN="${SHADOW_ALPHA_DOWN:-0.45}"
SHADOW_WIDTH_ADAPTIVE="${SHADOW_WIDTH_ADAPTIVE:-1}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
VERBOSE="${VERBOSE:-0}"

# Crash/kill retry policy
MAX_RETRIES="${MAX_RETRIES:-100}"
RETRY_SLEEP_SEC="${RETRY_SLEEP_SEC:-2}"
RETRY_CODES_DEFAULT="132 133 135 136 137 139 134"
RETRY_CODES="${RETRY_CODES:-$RETRY_CODES_DEFAULT}"
CLEANUP_PARTIALS="${CLEANUP_PARTIALS:-1}"

STOP_MARKER="${STOP_MARKER:-$OUTPUT_FOLDER/.stop_after_current}"
STOP_REQUEST_FILE="${TMPDIR:-/tmp}/mask_formerge_stop_request_${$}.flag"
RUN_STATE_DIR="${TMPDIR:-/tmp}/mask_formerge_state_${$}"
INTERRUPT_COUNT=0
STOP_REQUESTED=0
FORCE_STOP=0
pids=()
wids=()

if ! [[ "$WORKERS" =~ ^[0-9]+$ ]] || [ "$WORKERS" -le 0 ]; then
  echo "[ERR ] invalid WORKERS='$WORKERS' (must be integer > 0)"
  exit 2
fi

if [ ! -d "$REPLACE_MASK_FOLDER" ]; then
  echo "[ERR ] input folder not found: $REPLACE_MASK_FOLDER"
  exit 2
fi

mkdir -p "$OUTPUT_FOLDER"
mkdir -p "$RUN_STATE_DIR"

FILES=()
while IFS= read -r -d '' f; do
  FILES+=("$f")
done < <(find "$REPLACE_MASK_FOLDER" -maxdepth 1 -type f -name "$INPUT_GLOB" -print0 | sort -z)

if [ "${#FILES[@]}" -eq 0 ]; then
  echo "[WARN] no files found in $REPLACE_MASK_FOLDER matching '$INPUT_GLOB'"
  exit 0
fi

echo "[CFG ] workers=$WORKERS files=${#FILES[@]} use_gpu=$USE_GPU chunk=$CHUNK_SIZE"
echo "[CFG ] input=$REPLACE_MASK_FOLDER"
echo "[CFG ] output=$OUTPUT_FOLDER"
echo "[CFG ] motion_chain=$SHADOW_MOTION_CHAIN_ENABLED motion_deadzone=$SHADOW_MOTION_DEADZONE_PX motion_max=$SHADOW_MOTION_MAX_PX area_min=$SHADOW_AREA_MIN_PX area_max=$SHADOW_AREA_MAX_PX area_reset_ratio=$SHADOW_AREA_RESET_RATIO area_reset_abs=$SHADOW_AREA_RESET_ABS_PX y_tol=$SHADOW_COMPONENT_MERGE_Y_TOL_PX alpha_down=$SHADOW_ALPHA_DOWN"

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

should_retry() {
  local code="$1"
  for c in $RETRY_CODES; do
    if [ "$code" -eq "$c" ]; then
      return 0
    fi
  done
  return 1
}

output_path_for_input() {
  local f="$1"
  local stem
  stem="$(basename "$f")"
  stem="${stem%.*}"
  printf '%s/%s.mkv' "$OUTPUT_FOLDER" "$stem"
}

cleanup_current_outputs() {
  if [ "$CLEANUP_PARTIALS" != "1" ]; then
    return
  fi
  if [ ! -d "$RUN_STATE_DIR" ]; then
    return
  fi
  local st out
  for st in "$RUN_STATE_DIR"/w*.current; do
    [ -f "$st" ] || continue
    out="$(cat "$st" 2>/dev/null || true)"
    if [ -n "$out" ] && [ -f "$out" ]; then
      rm -f -- "$out" 2>/dev/null || true
      echo "[CLEAN] removed partial output: $out"
    fi
  done
}

cleanup_runtime() {
  rm -f "$STOP_MARKER" 2>/dev/null || true
  rm -f "$STOP_REQUEST_FILE" 2>/dev/null || true
  rm -rf "$RUN_STATE_DIR" 2>/dev/null || true
}

if [ -f "$STOP_MARKER" ]; then
  echo "[INFO] removing stale stop marker: $STOP_MARKER"
  rm -f -- "$STOP_MARKER" || true
fi

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
  echo "[STOP] force stop requested. Killing workers immediately."
  for pid in "${pids[@]:-}"; do
    signal_tree KILL "$pid"
  done
  cleanup_current_outputs
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

run_single_file_once() {
  local wid="$1"
  local fpath="$2"
  local base out_path
  base="$(basename "$fpath")"
  out_path="$(output_path_for_input "$fpath")"
  echo "$out_path" > "$RUN_STATE_DIR/w${wid}.current"

  local cmd=(
    "$PYTHON" "$RUNNER"
    --input-dir "$REPLACE_MASK_FOLDER"
    --output-dir "$OUTPUT_FOLDER"
    --glob "$base"
    --chunk-size "$CHUNK_SIZE"
    --mask-binarize-threshold "$MASK_BINARIZE_THRESHOLD"
    --mask-dilate-kernel-size "$MASK_DILATE_KERNEL_SIZE"
    --mask-blur-kernel-size "$MASK_BLUR_KERNEL_SIZE"
    --shadow-length-px "$SHADOW_LENGTH_PX"
    --shadow-curve "$SHADOW_CURVE"
    --shadow-motion-gain "$SHADOW_MOTION_GAIN"
    --shadow-motion-deadzone-px "$SHADOW_MOTION_DEADZONE_PX"
    --shadow-motion-max-px "$SHADOW_MOTION_MAX_PX"
    --shadow-area-min-px "$SHADOW_AREA_MIN_PX"
    --shadow-area-max-px "$SHADOW_AREA_MAX_PX"
    --shadow-area-reset-ratio "$SHADOW_AREA_RESET_RATIO"
    --shadow-area-reset-abs-px "$SHADOW_AREA_RESET_ABS_PX"
    --shadow-component-merge-y-tol-px "$SHADOW_COMPONENT_MERGE_Y_TOL_PX"
    --shadow-alpha-down "$SHADOW_ALPHA_DOWN"
  )

  if [ "$USE_GPU" = "1" ]; then
    cmd+=(--use-gpu-mask-ops)
  else
    cmd+=(--no-use-gpu-mask-ops)
  fi
  if [ "$SHADOW_WIDTH_ADAPTIVE" = "1" ]; then
    cmd+=(--shadow-width-adaptive)
  else
    cmd+=(--no-shadow-width-adaptive)
  fi
  if [ "$SHADOW_MOTION_CHAIN_ENABLED" = "1" ]; then
    cmd+=(--shadow-motion-chain-enabled)
  else
    cmd+=(--no-shadow-motion-chain-enabled)
  fi
  if [ "$SKIP_EXISTING" = "1" ]; then
    cmd+=(--skip-existing)
  else
    cmd+=(--no-skip-existing)
  fi
  if [ "$VERBOSE" = "1" ]; then
    cmd+=(--verbose)
  fi

  echo "[RUN w$wid] $base"
  if [ -z "${DISPLAY:-}" ] && command -v xvfb-run >/dev/null 2>&1; then
    xvfb-run -a "${cmd[@]}"
  else
    "${cmd[@]}"
  fi
}

run_single_file_with_retries() {
  local wid="$1"
  local fpath="$2"
  local base out_path
  base="$(basename "$fpath")"
  out_path="$(output_path_for_input "$fpath")"
  local attempt=1

  while true; do
    if [ -f "$STOP_REQUEST_FILE" ]; then
      echo "[STOP w$wid] stop marker detected before $base"
      return 0
    fi

    echo "[TRY w$wid] ${base} attempt ${attempt}/${MAX_RETRIES}"
    set +e
    run_single_file_once "$wid" "$fpath"
    local code=$?
    set -e

    rm -f "$RUN_STATE_DIR/w${wid}.current" 2>/dev/null || true

    if [ "$code" -eq 0 ]; then
      return 0
    fi

    if [ "$CLEANUP_PARTIALS" = "1" ] && [ -f "$out_path" ]; then
      rm -f -- "$out_path" 2>/dev/null || true
      echo "[CLEAN w$wid] removed partial output: $out_path"
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
      echo "[FAIL w$wid] $base exit_code=$code (no more retries)"
      return "$code"
    fi

    echo "[RETRY w$wid] $base exit_code=$code -> retrying in ${RETRY_SLEEP_SEC}s"
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

run_worker_with_retries() {
  local wid="$1"
  trap '' INT TERM

  local idx fpath rc had_fail first_fail_code
  had_fail=0
  first_fail_code=0
  for idx in "${!FILES[@]}"; do
    if [ $((idx % WORKERS)) -ne "$wid" ]; then
      continue
    fi
    fpath="${FILES[$idx]}"
    set +e
    run_single_file_with_retries "$wid" "$fpath"
    rc=$?
    set -e
    if [ "$rc" -ne 0 ]; then
      if [ "$rc" -eq 130 ]; then
        return 130
      fi
      if [ "$first_fail_code" -eq 0 ]; then
        first_fail_code="$rc"
      fi
      had_fail=1
      echo "[ERR w$wid] file failed rc=$rc: $(basename "$fpath") (continuing with remaining shard files)"
      continue
    fi
    if [ -f "$STOP_REQUEST_FILE" ]; then
      return 0
    fi
  done
  if [ "$had_fail" -ne 0 ]; then
    return "$first_fail_code"
  fi
  return 0
}

for ((wid=0; wid<WORKERS; wid++)); do
  wids+=("$wid")
  run_worker_with_retries "$wid" > "mask_formerge_worker_${wid}.log" 2>&1 &
  pids+=("$!")
  echo "[START] worker $wid pid=${pids[-1]} log=mask_formerge_worker_${wid}.log"
done

fail=0
fail_code=0
for i in "${!pids[@]}"; do
  pid="${pids[$i]}"
  wid="${wids[$i]}"
  set +e
  wait_for_pid "$pid"
  code=$?
  set -e
  if [ "$code" -ne 0 ]; then
    echo "[DONE] worker $wid FAILED exit_code=$code (see mask_formerge_worker_${wid}.log)"
    fail=1
    if [ "$fail_code" -eq 0 ]; then
      fail_code="$code"
    fi
  else
    echo "[DONE] worker $wid OK"
  fi
done

if [ "$FORCE_STOP" -eq 1 ]; then
  exit 130
fi

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

echo "[DONE] parallel mask_formerge completed"
