#!/usr/bin/env bash
set -euo pipefail

# Parallel launcher for run_splatting_runner.sh without changing Python runner.
# Strategy:
# - Build source/depth pairs
# - Shard pairs round-robin into per-worker temp folders (symlinks)
# - Launch one run_splatting_runner.sh per worker with isolated INPUT_* and STOP_MARKER
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="$REPO_ROOT/logs"
mkdir -p "$LOG_DIR"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO_ROOT"

RUN_SCRIPT="${RUN_SCRIPT:-$SCRIPT_DIR/run_splatting_runner.sh}"
WORKERS="${WORKERS:-2}"

INPUT_SOURCE_CLIPS="${INPUT_SOURCE_CLIPS:-./work/seg/}"
INPUT_DEPTH_MAPS="${INPUT_DEPTH_MAPS:-./work/depthmap/}"
OUTPUT_SPLATTED="${OUTPUT_SPLATTED:-./work/splat/}"

SHARD_ROOT="${SHARD_ROOT:-${TMPDIR:-/tmp}/splat_parallel_${USER:-user}_$$}"
KEEP_SHARDS="${KEEP_SHARDS:-0}"   # 1 keeps shard folders for debugging
LOG_PREFIX="${LOG_PREFIX:-$LOG_DIR/splat_worker}"

STOP_REQUEST_FILE="${TMPDIR:-/tmp}/splat_parallel_stop_${$}.flag"
INTERRUPT_COUNT=0
STOP_REQUESTED=0
FORCE_STOP=0

declare -a pids=()
declare -a wids=()
declare -a worker_markers=()
declare -a worker_logs=()
declare -a worker_jobs=()
declare -a worker_line_offsets=()
declare -a worker_done_counts=()
declare -a worker_total_counts=()
declare -a worker_exit_reported=()
declare -a worker_exit_codes=()

matched=0
last_progress_done=-1
last_progress_total=-1

if [[ ! "$WORKERS" =~ ^[0-9]+$ ]] || [[ "$WORKERS" -lt 1 ]]; then
  echo "[ERR ] invalid WORKERS='$WORKERS' (must be >=1 integer)"
  exit 2
fi

if [[ ! -f "$RUN_SCRIPT" ]]; then
  echo "[ERR ] run script not found: $RUN_SCRIPT"
  exit 2
fi

if [[ -f "$INPUT_SOURCE_CLIPS" && -f "$INPUT_DEPTH_MAPS" ]]; then
  echo "[INFO] single-file mode detected; WORKERS ignored."
  exec bash "$RUN_SCRIPT"
fi

if [[ "$WORKERS" -eq 1 ]]; then
  echo "[INFO] WORKERS=1 -> switching to single-run launcher: $RUN_SCRIPT"
  exec bash "$RUN_SCRIPT"
fi

if [[ ! -d "$INPUT_SOURCE_CLIPS" ]]; then
  echo "[ERR ] source folder missing: $INPUT_SOURCE_CLIPS"
  exit 2
fi
if [[ ! -d "$INPUT_DEPTH_MAPS" ]]; then
  echo "[ERR ] depth folder missing: $INPUT_DEPTH_MAPS"
  exit 2
fi

mkdir -p "$SHARD_ROOT"
mkdir -p "$OUTPUT_SPLATTED"

for ((wid=0; wid<WORKERS; wid++)); do
  mkdir -p "$SHARD_ROOT/w${wid}/src" "$SHARD_ROOT/w${wid}/depth"
  worker_jobs[$wid]=0
done

find_depth_for_source() {
  local src_path="$1"
  local src_base src_stem
  src_base="$(basename "$src_path")"
  src_stem="${src_base%.*}"
  local exts=(mp4 mkv mov avi npz)
  local ext cand
  for ext in "${exts[@]}"; do
    cand="$INPUT_DEPTH_MAPS/${src_stem}_depth.${ext}"
    [[ -f "$cand" ]] && { printf '%s\n' "$cand"; return 0; }
  done
  for ext in "${exts[@]}"; do
    cand="$INPUT_DEPTH_MAPS/${src_stem}.${ext}"
    [[ -f "$cand" ]] && { printf '%s\n' "$cand"; return 0; }
  done
  return 1
}

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

is_uint() {
  [[ "${1:-}" =~ ^[0-9]+$ ]]
}

update_worker_progress() {
  local wid="$1"
  local done="$2"
  local total="$3"
  local prev_done="${worker_done_counts[$wid]:-0}"
  local prev_total="${worker_total_counts[$wid]:-0}"
  if ! is_uint "$prev_done"; then prev_done=0; fi
  if ! is_uint "$prev_total"; then prev_total=0; fi

  if is_uint "$done" && (( done > prev_done )); then
    worker_done_counts[$wid]="$done"
  fi
  if is_uint "$total" && (( total > prev_total )); then
    worker_total_counts[$wid]="$total"
  fi

  local cur_done="${worker_done_counts[$wid]:-0}"
  local cur_total="${worker_total_counts[$wid]:-0}"
  if is_uint "$cur_total" && is_uint "$cur_done" && (( cur_total > 0 && cur_done > cur_total )); then
    worker_done_counts[$wid]="$cur_total"
  fi
}

emit_progress_snapshot() {
  local sum_done=0
  local sum_total=0
  local wid
  for ((wid=0; wid<WORKERS; wid++)); do
    local d="${worker_done_counts[$wid]:-0}"
    local t="${worker_total_counts[$wid]:-0}"
    if is_uint "$d"; then
      sum_done=$((sum_done + d))
    fi
    if is_uint "$t" && (( t > 0 )); then
      sum_total=$((sum_total + t))
    fi
  done

  if (( sum_total <= 0 )); then
    sum_total="$matched"
  fi
  if (( sum_total <= 0 )); then
    return
  fi
  if (( sum_done > sum_total )); then
    sum_done="$sum_total"
  fi

  if (( sum_done != last_progress_done || sum_total != last_progress_total )); then
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

  if [[ "$line" =~ ^\[RUN[[:space:]]*\][[:space:]]*([0-9]+)[[:space:]]*/[[:space:]]*([0-9]+) ]]; then
    update_worker_progress "$wid" "${BASH_REMATCH[1]}" "${BASH_REMATCH[2]}"
    return
  fi

  if [[ "$line" =~ ^\[TOTAL\][[:space:]]*([0-9]+) ]]; then
    update_worker_progress "$wid" "${worker_done_counts[$wid]:-0}" "${BASH_REMATCH[1]}"
    return
  fi

  if [[ "$line" =~ ^\[RETRY\] ]]; then
    echo "[RETRY] worker $wid ${line#\[RETRY\] }"
    return
  fi

  if [[ "$line" =~ ^\[FAIL\] ]]; then
    echo "[ERR ] worker $wid ${line}"
    return
  fi

  if [[ "$line" =~ ^\[STOP\] ]]; then
    echo "[STOP] worker $wid ${line#\[STOP\] }"
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

  if [[ "$line" =~ \[ERR[[:space:]]*\] ]]; then
    echo "[ERR ] worker $wid ${line}"
    return
  fi
}

poll_worker_log() {
  local wid="$1"
  local log_file="$2"
  if [[ ! -f "$log_file" ]]; then
    return
  fi
  local offset="${worker_line_offsets[$wid]:-0}"
  if ! is_uint "$offset"; then
    offset=0
  fi
  local -a new_lines=()
  mapfile -s "$offset" -t new_lines <"$log_file" || true
  local new_count="${#new_lines[@]}"
  if (( new_count <= 0 )); then
    return
  fi
  worker_line_offsets[$wid]=$((offset + new_count))
  local line
  for line in "${new_lines[@]}"; do
    parse_worker_log_line "$wid" "$line"
  done
}

poll_all_worker_logs() {
  local i
  for i in "${!worker_logs[@]}"; do
    local wid="${wids[$i]}"
    local log_file="${worker_logs[$i]}"
    poll_worker_log "$wid" "$log_file"
  done
}

request_graceful_stop() {
  if [[ "$STOP_REQUESTED" -eq 1 ]]; then
    return
  fi
  STOP_REQUESTED=1
  : > "$STOP_REQUEST_FILE"
  echo "[STOP] graceful stop requested. Waiting current clips..."
  local m
  for m in "${worker_markers[@]:-}"; do
    [[ -n "$m" ]] && { mkdir -p "$(dirname "$m")" 2>/dev/null || true; : > "$m"; }
  done
}

request_forced_stop() {
  FORCE_STOP=1
  echo "[STOP] force stop requested. Killing all workers."
  local pid
  for pid in "${pids[@]:-}"; do
    signal_tree KILL "$pid"
  done
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

cleanup_runtime() {
  rm -f -- "$STOP_REQUEST_FILE" 2>/dev/null || true
  local m
  for m in "${worker_markers[@]:-}"; do
    rm -f -- "$m" 2>/dev/null || true
  done
  if [[ "$KEEP_SHARDS" != "1" ]]; then
    rm -rf -- "$SHARD_ROOT" 2>/dev/null || true
  fi
}

trap on_interrupt INT TERM
trap cleanup_runtime EXIT

mapfile -t source_files < <(
  find "$INPUT_SOURCE_CLIPS" -maxdepth 1 \( -type f -o -type l \) \
    \( -iname '*.mp4' -o -iname '*.mkv' -o -iname '*.mov' -o -iname '*.avi' \) \
    | sort
)

if [[ "${#source_files[@]}" -eq 0 ]]; then
  echo "[ERR ] no source clips found in: $INPUT_SOURCE_CLIPS"
  exit 2
fi

missing=0
broken_sources=0
for src in "${source_files[@]}"; do
  if [[ ! -f "$src" ]]; then
    echo "[WARN] source clip unreadable or broken link, skipping: $src"
    broken_sources=$((broken_sources + 1))
    continue
  fi

  depth_path="$(find_depth_for_source "$src" || true)"
  if [[ -z "${depth_path:-}" ]]; then
    echo "[WARN] depth match missing for $(basename "$src"), skipping"
    missing=$((missing + 1))
    continue
  fi

  wid=$((matched % WORKERS))
  ln -sfn "$src" "$SHARD_ROOT/w${wid}/src/$(basename "$src")"
  ln -sfn "$depth_path" "$SHARD_ROOT/w${wid}/depth/$(basename "$depth_path")"
  worker_jobs[$wid]=$((worker_jobs[$wid] + 1))
  matched=$((matched + 1))
done

if [[ "$matched" -eq 0 ]]; then
  echo "[ERR ] no source/depth pairs resolved. Nothing to process."
  exit 2
fi

echo "[PAIR] matched=$matched missing_depth=$missing broken_source=$broken_sources workers=$WORKERS shard_root=$SHARD_ROOT"

active_workers=0
for ((wid=0; wid<WORKERS; wid++)); do
  jobs="${worker_jobs[$wid]}"
  if [[ "$jobs" -le 0 ]]; then
    echo "[SKIP] worker $wid has 0 jobs"
    continue
  fi

  src_dir="$SHARD_ROOT/w${wid}/src"
  depth_dir="$SHARD_ROOT/w${wid}/depth"
  stop_marker="$OUTPUT_SPLATTED/.stop_after_current_w${wid}"
  log_file="${LOG_PREFIX}_${wid}.log"

  worker_markers+=("$stop_marker")
  worker_logs+=("$log_file")
  worker_line_offsets[$wid]=0
  worker_done_counts[$wid]=0
  worker_total_counts[$wid]="$jobs"

  (
    trap '' INT TERM
    INPUT_SOURCE_CLIPS="$src_dir" \
    INPUT_DEPTH_MAPS="$depth_dir" \
    STOP_MARKER="$stop_marker" \
    bash "$RUN_SCRIPT"
  ) >"$log_file" 2>&1 &

  pids+=("$!")
  wids+=("$wid")
  active_workers=$((active_workers + 1))
  echo "[START] worker $wid jobs=$jobs pid=${pids[-1]} log=$log_file"
done

if [[ "$active_workers" -eq 0 ]]; then
  echo "[ERR ] all workers empty."
  exit 2
fi

fail=0
fail_code=0
for i in "${!pids[@]}"; do
  worker_exit_reported[$i]=0
  worker_exit_codes[$i]=-1
done

remaining="${#pids[@]}"
emit_progress_snapshot
while (( remaining > 0 )); do
  poll_all_worker_logs
  emit_progress_snapshot

  for i in "${!pids[@]}"; do
    if [[ "${worker_exit_reported[$i]:-0}" -eq 1 ]]; then
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

    if [[ "$code" -ne 0 ]]; then
      echo "[ERR ][CRASH] worker $wid exit_code=$code (log: ${LOG_PREFIX}_${wid}.log)"
      fail=1
      if [[ "$fail_code" -eq 0 ]]; then
        fail_code="$code"
        request_graceful_stop
      fi
    else
      echo "[WORKER] worker $wid OK"
    fi
  done

  if (( remaining > 0 )); then
    sleep 0.4
  fi
done

poll_all_worker_logs
if [[ "$fail" -eq 0 ]]; then
  for i in "${!pids[@]}"; do
    if [[ "${worker_exit_codes[$i]:-1}" -ne 0 ]]; then
      continue
    fi
    wid="${wids[$i]}"
    total_for_wid="${worker_total_counts[$wid]:-${worker_jobs[$wid]:-0}}"
    if ! is_uint "$total_for_wid"; then
      total_for_wid=0
    fi
    if (( total_for_wid > 0 )); then
      update_worker_progress "$wid" "$total_for_wid" "$total_for_wid"
    fi
  done
fi
emit_progress_snapshot

if [[ "$FORCE_STOP" -eq 1 ]]; then
  exit 130
fi

if [[ "$STOP_REQUESTED" -eq 1 ]] || [[ -f "$STOP_REQUEST_FILE" ]]; then
  if [[ "$fail" -eq 0 ]]; then
    echo "[STOP] graceful stop completed"
    exit 0
  fi
fi

if [[ "$fail" -ne 0 ]]; then
  exit "$fail_code"
fi

echo "[OK] all workers finished"
exit 0
