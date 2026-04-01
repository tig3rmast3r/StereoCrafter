#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python3}"
RUNNER="${RUNNER:-merging_nogui_batch_parallel.py}"

INPAINTED_FOLDER="${INPAINTED_FOLDER:-./work/output/}"
PREFERRED_INPAINTED_FOLDER="${PREFERRED_INPAINTED_FOLDER:-}"
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
ENCODING_MODE="${ENCODING_MODE:-}"
FFMPEG_EXTRA_ARGS="${FFMPEG_EXTRA_ARGS:-}"
RESTART_EVERY="${RESTART_EVERY:-1}"
PLANNED_RESTART_CODE="${PLANNED_RESTART_CODE:-99}"

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
worker_logs=()
worker_line_offsets=()
worker_done_counts=()
worker_total_counts=()
worker_exit_reported=()
worker_exit_codes=()
declare -A worker_seen_jobs=()

last_progress_done=-1
last_progress_total=-1

if ! [[ "$WORKERS" =~ ^[0-9]+$ ]] || [ "$WORKERS" -lt 1 ]; then
  echo "[ERR ] invalid WORKERS='$WORKERS' (must be >=1 integer)"
  exit 2
fi

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
if [ -n "${ENCODING_MODE// }" ]; then
  CMD+=(--encoding-mode "$ENCODING_MODE")
fi
if [ -n "${FFMPEG_EXTRA_ARGS// }" ]; then
  CMD+=(--ffmpeg-extra-args "$FFMPEG_EXTRA_ARGS")
fi

export MERGE_DEBUG

echo "[BASE CMD] ${CMD[*]}"
echo "[PAR] WORKERS=$WORKERS"
echo "[DBG] MERGE_DEBUG=$MERGE_DEBUG"

if [ "$WORKERS" -eq 1 ]; then
  echo "[INFO] WORKERS=1 -> switching to single-run launcher: run_merging_nogui_batch.sh"
  export RUNNER="merging_nogui_batch.py"
  exec bash run_merging_nogui_batch.sh
fi

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

set_worker_total_max() {
  local wid="$1"
  local total="$2"
  local prev="${worker_total_counts[$wid]:-0}"
  if ! is_uint "$prev"; then
    prev=0
  fi
  if is_uint "$total" && [ "$total" -gt "$prev" ]; then
    worker_total_counts[$wid]="$total"
  fi
}

increment_worker_done() {
  local wid="$1"
  local cur="${worker_done_counts[$wid]:-0}"
  if ! is_uint "$cur"; then
    cur=0
  fi
  cur=$((cur + 1))
  local max_total="${worker_total_counts[$wid]:-0}"
  if is_uint "$max_total" && [ "$max_total" -gt 0 ] && [ "$cur" -gt "$max_total" ]; then
    cur="$max_total"
  fi
  worker_done_counts[$wid]="$cur"
}

extract_worker_progress_key() {
  local line="$1"
  if [[ "$line" == *"DONE:"* ]]; then
    printf '%s' "${line##*: }"
    return
  fi
  if [[ "$line" == *"SKIP (exists"*":"* ]]; then
    printf '%s' "${line##*: }"
    return
  fi
  if [[ "$line" == *"GIVING UP:"* ]]; then
    printf '%s' "${line##*: }"
    return
  fi
  printf ''
}

mark_worker_job_done_once() {
  local wid="$1"
  local line="$2"
  local key
  key="$(extract_worker_progress_key "$line")"
  if [ -z "$key" ]; then
    increment_worker_done "$wid"
    return
  fi
  local seen_key="${wid}::${key}"
  if [ -n "${worker_seen_jobs[$seen_key]:-}" ]; then
    return
  fi
  worker_seen_jobs[$seen_key]=1
  increment_worker_done "$wid"
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

parse_worker_log_line() {
  local wid="$1"
  local raw_line="$2"
  local line="${raw_line//$'\r'/}"
  line="${line#"${line%%[![:space:]]*}"}"

  if [[ "$line" =~ \[SHARD\][[:space:]]worker[[:space:]]${wid}/[0-9]+[[:space:]]will[[:space:]]process[[:space:]]([0-9]+)[[:space:]]jobs ]]; then
    set_worker_total_max "$wid" "${BASH_REMATCH[1]}"
    return
  fi

  if [[ "$line" == *"DONE:"* ]]; then
    mark_worker_job_done_once "$wid" "$line"
    return
  fi
  if [[ "$line" == *"SKIP (exists"* ]]; then
    mark_worker_job_done_once "$wid" "$line"
    return
  fi
  if [[ "$line" == *"GIVING UP:"* ]]; then
    mark_worker_job_done_once "$wid" "$line"
    echo "[ERR ] worker $wid ${line}"
    return
  fi

  if [[ "$line" =~ ^\[RETRY[[:space:]]w${wid}\] ]]; then
    echo "[RETRY] worker $wid ${line#*] }"
    return
  fi
  if [[ "$line" =~ ^\[PLANNED[[:space:]]RESTART\] ]]; then
    echo "[INFO] worker $wid ${line}"
    return
  fi
  if [[ "$line" =~ ^\[RESUME\] ]]; then
    echo "[INFO] worker $wid ${line}"
    return
  fi
  if [[ "$line" =~ ^\[FAIL[[:space:]]w${wid}\] ]]; then
    echo "[ERR ] worker $wid ${line}"
    return
  fi
  if [[ "$line" =~ ^\[STOP ]]; then
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
  local _code="${1:-1}"
  # Retry on any non-zero exit code; MAX_RETRIES still applies.
  return 0
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
    if [ "$code" -eq "$PLANNED_RESTART_CODE" ]; then
      echo "[RESTART w$wid] planned process restart requested -> relaunching immediately"
      continue
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
  log_file="merge_worker_${wid}.log"
  worker_logs+=("$log_file")
  worker_line_offsets[$wid]=0
  worker_done_counts[$wid]=0
  worker_total_counts[$wid]=0
  run_worker_with_retries "$wid" > "$log_file" 2>&1 &
  pids+=("$!")
  echo "[START] worker $wid pid=${pids[-1]} log=$log_file"
done

fail=0
fail_code=0
for i in "${!pids[@]}"; do
  worker_exit_reported[$i]=0
  worker_exit_codes[$i]=-1
done

remaining="${#pids[@]}"
while [ "$remaining" -gt 0 ]; do
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
      echo "[ERR ][CRASH] worker $wid exit_code=$code (log: merge_worker_${wid}.log)"
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
