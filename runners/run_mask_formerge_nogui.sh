#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="$REPO_ROOT/logs"
mkdir -p "$LOG_DIR"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-python3}"
RUNNER="${RUNNER:-$SCRIPT_DIR/mask_formerge_nogui.py}"

# Input replace-mask folder from merge defaults
REPLACE_MASK_FOLDER="${REPLACE_MASK_FOLDER:-./work/mask/}"
OUTPUT_FOLDER="${OUTPUT_FOLDER:-./work/mask_for_merge/}"
INPUT_GLOB="${INPUT_GLOB:-*_replace_mask.*}"

# Parallelism: one python process at a time per worker, deterministic file sharding.
WORKERS="${WORKERS:-19}"

# Defaults aligned with run_merging_nogui_batch.sh
CHUNK_SIZE="${CHUNK_SIZE:-20}"
USE_GPU="${USE_GPU:-0}"
MASK_BINARIZE_THRESHOLD="${MASK_BINARIZE_THRESHOLD:-0.5}"
MASK_DILATE_KERNEL_SIZE="${MASK_DILATE_KERNEL_SIZE:-2}"
MASK_BLUR_KERNEL_SIZE="${MASK_BLUR_KERNEL_SIZE:-2}"
SHADOW_LENGTH_PX="${SHADOW_LENGTH_PX:-15}"
SHADOW_CURVE="${SHADOW_CURVE:-0}"
SHADOW_WIDTH_ADAPTIVE="${SHADOW_WIDTH_ADAPTIVE:-1}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
VERBOSE="${VERBOSE:-0}"
OMP_NUM_THREADS_MASK="${OMP_NUM_THREADS_MASK:-1}"
MKL_NUM_THREADS_MASK="${MKL_NUM_THREADS_MASK:-1}"
OPENBLAS_NUM_THREADS_MASK="${OPENBLAS_NUM_THREADS_MASK:-1}"
NUMEXPR_NUM_THREADS_MASK="${NUMEXPR_NUM_THREADS_MASK:-1}"

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
worker_logs=()
worker_jobs=()
worker_line_offsets=()
worker_done_counts=()
worker_total_counts=()
worker_exit_reported=()
worker_exit_codes=()

last_progress_done=-1
last_progress_total=-1

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
  if [ ! -f "$f" ]; then
    echo "[WARN] skipping unreadable input (broken link?): $f"
    continue
  fi
  FILES+=("$f")
done < <(find "$REPLACE_MASK_FOLDER" -maxdepth 1 \( -type f -o -type l \) -name "$INPUT_GLOB" -print0 | sort -z)

if [ "${#FILES[@]}" -eq 0 ]; then
  echo "[WARN] no files found in $REPLACE_MASK_FOLDER matching '$INPUT_GLOB'"
  exit 0
fi

for ((wid=0; wid<WORKERS; wid++)); do
  worker_jobs[$wid]=0
done
for idx in "${!FILES[@]}"; do
  wid=$((idx % WORKERS))
  worker_jobs[$wid]=$((worker_jobs[$wid] + 1))
done

echo "[CFG ] workers=$WORKERS files=${#FILES[@]} use_gpu=$USE_GPU chunk=$CHUNK_SIZE"
echo "[CFG ] input=$REPLACE_MASK_FOLDER"
echo "[CFG ] output=$OUTPUT_FOLDER"

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

is_uint() {
  [[ "${1:-}" =~ ^[0-9]+$ ]]
}

emit_progress_snapshot() {
  local sum_done=0
  local sum_total=0
  local wid d t
  for ((wid=0; wid<WORKERS; wid++)); do
    d="${worker_done_counts[$wid]:-0}"
    t="${worker_total_counts[$wid]:-0}"
    if is_uint "$d"; then
      sum_done=$((sum_done + d))
    fi
    if is_uint "$t"; then
      sum_total=$((sum_total + t))
    fi
  done
  if [ "$sum_total" -le 0 ]; then
    return
  fi
  if [ "$sum_done" -gt "$sum_total" ]; then
    sum_done="$sum_total"
  fi
  if [ "$sum_done" -ne "$last_progress_done" ] || [ "$sum_total" -ne "$last_progress_total" ]; then
    last_progress_done="$sum_done"
    last_progress_total="$sum_total"
    echo "[RUN ] ${sum_done}/${sum_total}"
  fi
}

increment_worker_done() {
  local wid="$1"
  local cur="${worker_done_counts[$wid]:-0}"
  if ! is_uint "$cur"; then
    cur=0
  fi
  cur=$((cur + 1))
  local total="${worker_total_counts[$wid]:-0}"
  if is_uint "$total" && [ "$total" -gt 0 ] && [ "$cur" -gt "$total" ]; then
    cur="$total"
  fi
  worker_done_counts[$wid]="$cur"
}

parse_worker_log_line() {
  local wid="$1"
  local raw_line="$2"
  local line="${raw_line//$'\r'/}"
  line="${line#"${line%%[![:space:]]*}"}"

  if [[ "$line" == *"[DONE]"* ]]; then
    increment_worker_done "$wid"
    return
  fi
  if [[ "$line" == *"[SKIP]"* ]]; then
    increment_worker_done "$wid"
    return
  fi
  if [[ "$line" == *"[ERR w${wid}] file failed rc="* ]]; then
    increment_worker_done "$wid"
    echo "[ERR ] worker $wid ${line}"
    return
  fi

  if [[ "$line" == *"[RETRY w${wid}]"* ]]; then
    echo "[RETRY] worker $wid ${line#*] }"
    return
  fi
  if [[ "$line" == *"[FAIL w${wid}]"* ]]; then
    echo "[ERR ] worker $wid ${line}"
    return
  fi
  if [[ "$line" == *"[STOP w${wid}]"* ]]; then
    echo "[STOP] worker $wid ${line#*] }"
    return
  fi

  local lc="${line,,}"
  if [[ "$lc" == *"cuda out of memory"* || "$lc" == *"out of memory"* || "$lc" == *"cudnn_status_alloc_failed"* || "$lc" == *"std::bad_alloc"* ]]; then
    echo "[ERR ][OOM] worker $wid ${line}"
    return
  fi
  if [[ "$lc" == *"cuda error"* || "$lc" == *"torch.acceleratorerror"* || "$lc" == *"invalid configuration argument"* ]]; then
    echo "[ERR ][CUDA] worker $wid ${line}"
    return
  fi
}

poll_worker_log() {
  local wid="$1"
  local log_file="$2"
  if [ ! -f "$log_file" ]; then
    return
  fi
  local offset="${worker_line_offsets[$wid]:-0}"
  if ! is_uint "$offset"; then
    offset=0
  fi
  local -a new_lines=()
  mapfile -s "$offset" -t new_lines <"$log_file" || true
  local new_count="${#new_lines[@]}"
  if [ "$new_count" -le 0 ]; then
    return
  fi
  worker_line_offsets[$wid]=$((offset + new_count))
  local line
  for line in "${new_lines[@]}"; do
    parse_worker_log_line "$wid" "$line"
  done
}

poll_all_worker_logs() {
  local i wid log_file
  for i in "${!worker_logs[@]}"; do
    wid="${wids[$i]}"
    log_file="${worker_logs[$i]}"
    poll_worker_log "$wid" "$log_file"
  done
}

should_retry() {
  local _code="${1:-1}"
  local retry_code
  if ! [[ "$_code" =~ ^[0-9]+$ ]] || [ "$_code" -eq 0 ]; then
    return 1
  fi
  for retry_code in $RETRY_CODES; do
    if [ "$retry_code" = "$_code" ]; then
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
    env
    OMP_NUM_THREADS="$OMP_NUM_THREADS_MASK"
    MKL_NUM_THREADS="$MKL_NUM_THREADS_MASK"
    OPENBLAS_NUM_THREADS="$OPENBLAS_NUM_THREADS_MASK"
    NUMEXPR_NUM_THREADS="$NUMEXPR_NUM_THREADS_MASK"
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
    if [ "$STOP_REQUESTED" -eq 0 ] && [ -f "$STOP_MARKER" ]; then
      request_graceful_stop
    fi
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
      if [ "$STOP_REQUESTED" -eq 0 ] && [ -f "$STOP_MARKER" ]; then
        request_graceful_stop
      fi
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
  log_file="$LOG_DIR/mask_formerge_worker_${wid}.log"
  worker_logs+=("$log_file")
  worker_line_offsets[$wid]=0
  worker_done_counts[$wid]=0
  worker_total_counts[$wid]="${worker_jobs[$wid]:-0}"
  run_worker_with_retries "$wid" > "$log_file" 2>&1 &
  pids+=("$!")
  echo "[START] worker $wid jobs=${worker_jobs[$wid]:-0} pid=${pids[-1]} log=$log_file"
done

fail=0
fail_code=0
for i in "${!pids[@]}"; do
  worker_exit_reported[$i]=0
  worker_exit_codes[$i]=-1
done

remaining="${#pids[@]}"
while [ "$remaining" -gt 0 ]; do
  if [ "$STOP_REQUESTED" -eq 0 ] && [ -f "$STOP_MARKER" ]; then
    request_graceful_stop
  fi
  poll_all_worker_logs
  emit_progress_snapshot

  for i in "${!pids[@]}"; do
    if [ "${worker_exit_reported[$i]:-0}" -eq 1 ]; then
      continue
    fi
    pid="${pids[$i]}"
    if kill -0 "$pid" 2>/dev/null; then
      continue
    fi

    set +e
    wait "$pid"
    code=$?
    set -e

    worker_exit_reported[$i]=1
    worker_exit_codes[$i]="$code"
    remaining=$((remaining - 1))
    wid="${wids[$i]}"

    if [ "$code" -ne 0 ]; then
      echo "[ERR ][CRASH] worker $wid exit_code=$code (log: $LOG_DIR/mask_formerge_worker_${wid}.log)"
      fail=1
      if [ "$fail_code" -eq 0 ]; then
        fail_code="$code"
      fi
    else
      echo "[WORKER] worker $wid OK"
    fi
  done

  if [ "$remaining" -gt 0 ]; then
    sleep 0.4
  fi
done

poll_all_worker_logs
if [ "$fail" -eq 0 ]; then
  for i in "${!pids[@]}"; do
    if [ "${worker_exit_codes[$i]:-1}" -ne 0 ]; then
      continue
    fi
    wid="${wids[$i]}"
    total_for_wid="${worker_total_counts[$wid]:-0}"
    if is_uint "$total_for_wid" && [ "$total_for_wid" -gt 0 ]; then
      worker_done_counts[$wid]="$total_for_wid"
    fi
  done
fi
emit_progress_snapshot

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

echo "[OK] parallel mask_formerge completed"
