#!/usr/bin/env bash
set -euo pipefail

# Parallel launcher for run_splatting_runner.sh without changing Python runner.
# Strategy:
# - Build source/depth pairs
# - Shard pairs round-robin into per-worker temp folders (symlinks)
# - Launch one run_splatting_runner.sh per worker with isolated INPUT_* and STOP_MARKER

RUN_SCRIPT="${RUN_SCRIPT:-run_splatting_runner.sh}"
WORKERS="${WORKERS:-2}"

INPUT_SOURCE_CLIPS="${INPUT_SOURCE_CLIPS:-./work/seg/}"
INPUT_DEPTH_MAPS="${INPUT_DEPTH_MAPS:-./work/depthmap/upscaled/}"
OUTPUT_SPLATTED="${OUTPUT_SPLATTED:-./work/splat/}"

SHARD_ROOT="${SHARD_ROOT:-${TMPDIR:-/tmp}/splat_parallel_${USER:-user}_$$}"
KEEP_SHARDS="${KEEP_SHARDS:-0}"   # 1 keeps shard folders for debugging
LOG_PREFIX="${LOG_PREFIX:-splat_worker}"

STOP_REQUEST_FILE="${TMPDIR:-/tmp}/splat_parallel_stop_${$}.flag"
INTERRUPT_COUNT=0
STOP_REQUESTED=0
FORCE_STOP=0

declare -a pids=()
declare -a wids=()
declare -a worker_markers=()
declare -a worker_logs=()
declare -a worker_jobs=()

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
  find "$INPUT_SOURCE_CLIPS" -maxdepth 1 -type f \
    \( -iname '*.mp4' -o -iname '*.mkv' -o -iname '*.mov' -o -iname '*.avi' \) \
    | sort
)

if [[ "${#source_files[@]}" -eq 0 ]]; then
  echo "[ERR ] no source clips found in: $INPUT_SOURCE_CLIPS"
  exit 2
fi

matched=0
missing=0
for src in "${source_files[@]}"; do
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

echo "[PAIR] matched=$matched missing_depth=$missing workers=$WORKERS shard_root=$SHARD_ROOT"

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
  pid="${pids[$i]}"
  wid="${wids[$i]}"
  wait_for_pid "$pid"
  code=$?
  if [[ "$code" -ne 0 ]]; then
    echo "[DONE] worker $wid FAILED exit_code=$code (log: ${LOG_PREFIX}_${wid}.log)"
    fail=1
    if [[ "$fail_code" -eq 0 ]]; then
      fail_code="$code"
      request_graceful_stop
    fi
  else
    echo "[DONE] worker $wid OK"
  fi
done

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
