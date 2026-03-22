#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python3}"
RUNNER="${RUNNER:-batch_inpaint_sharpen_runner.py}"
INPUT_DIR="${INPUT_DIR:-./work/output/}"
MASK_DIR="${MASK_DIR:-./work/mask/}"
OUTPUT_DIR="${OUTPUT_DIR:-./work/output-sharpen/}"
SHARPNESS_CSV_PATH="${SHARPNESS_CSV_PATH:-./work/sharpness.csv}"
GLOB="${GLOB:-*.mp4}"
WORKERS="${WORKERS:-8}"
ONLY="${ONLY:-}"
STOP_MARKER="${STOP_MARKER:-$OUTPUT_DIR/.stop_after_current}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
OUTPUT_CODEC="${OUTPUT_CODEC:-libx264}"
OUTPUT_PRESET="${OUTPUT_PRESET:-slow}"
OUTPUT_PIX_FMT="${OUTPUT_PIX_FMT:-yuv444p}"
OUTPUT_CRF="${OUTPUT_CRF:-0}"
OUTPUT_EXTRA_ARGS="${OUTPUT_EXTRA_ARGS:-}"

CMD=(
  "$PYTHON" "$RUNNER"
  --input_dir "$INPUT_DIR"
  --mask_dir "$MASK_DIR"
  --output_dir "$OUTPUT_DIR"
  --sharpness_csv_path "$SHARPNESS_CSV_PATH"
  --glob "$GLOB"
  --workers "$WORKERS"
  --stop_marker "$STOP_MARKER"
  --codec "$OUTPUT_CODEC"
  --preset "$OUTPUT_PRESET"
  --pix_fmt "$OUTPUT_PIX_FMT"
  --crf "$OUTPUT_CRF"
)

if [[ -n "${ONLY// }" ]]; then
  CMD+=(--only "$ONLY")
fi
if [[ "$SKIP_EXISTING" == "1" ]]; then
  CMD+=(--skip_existing)
fi
if [[ -n "${OUTPUT_EXTRA_ARGS// }" ]]; then
  CMD+=(--output_extra_args "$OUTPUT_EXTRA_ARGS")
fi

echo "[CMD] ${CMD[*]}"

mkdir -p "$OUTPUT_DIR"
if [[ -f "$STOP_MARKER" ]]; then
  rm -f -- "$STOP_MARKER" || true
fi

MAX_RETRIES="${MAX_RETRIES:-0}"
RETRY_SLEEP_SEC="${RETRY_SLEEP_SEC:-2}"
WATCHDOG_ENABLED="${WATCHDOG_ENABLED:-False}"
WATCHDOG_POLL_SEC="${WATCHDOG_POLL_SEC:-20}"
WATCHDOG_IDLE_SEC="${WATCHDOG_IDLE_SEC:-600}"
WATCHDOG_TERM_GRACE_SEC="${WATCHDOG_TERM_GRACE_SEC:-15}"

STOP_REQUESTED=0
FORCE_STOP=0
CURRENT_CHILD_PID=""
CURRENT_CHILD_PGID=""

_pgid_has_members() {
  local pgid="$1"
  [[ -n "$pgid" ]] || return 1
  ps -o pid= -g "$pgid" 2>/dev/null | awk 'NF{found=1; exit} END{exit found?0:1}'
}

_kill_child_group() {
  local pid="$1"
  local pgid="$2"
  if [[ -n "$pgid" ]]; then
    kill -TERM -- "-$pgid" 2>/dev/null || true
  fi
  kill -TERM "$pid" 2>/dev/null || true
  sleep 2
  if [[ -n "$pgid" ]] && _pgid_has_members "$pgid"; then
    kill -KILL -- "-$pgid" 2>/dev/null || true
  fi
  if kill -0 "$pid" 2>/dev/null; then
    kill -KILL "$pid" 2>/dev/null || true
  fi
}

_request_stop_signal() {
  if [[ "$STOP_REQUESTED" -eq 0 ]]; then
    STOP_REQUESTED=1
    mkdir -p "$(dirname "$STOP_MARKER")" 2>/dev/null || true
    : > "$STOP_MARKER"
    echo "[STOP] graceful stop requested. Finishing current sharpen file(s) before exit."
    return 0
  fi
  FORCE_STOP=1
  echo "[STOP] force stop requested. Killing sharpen runner immediately."
  if [[ -n "$CURRENT_CHILD_PID" ]] && kill -0 "$CURRENT_CHILD_PID" 2>/dev/null; then
    _kill_child_group "$CURRENT_CHILD_PID" "$CURRENT_CHILD_PGID"
  fi
}

trap _request_stop_signal INT TERM

_is_true() {
  local v="${1:-}"
  case "${v,,}" in
    1|true|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

_latest_mp4_in_output() {
  find "$OUTPUT_DIR" -type f -name "*.mp4" -printf '%T@|%p\n' 2>/dev/null \
    | awk -F'|' '
      BEGIN { max_ts = -1; latest = "" }
      {
        ts = $1 + 0
        if (ts > max_ts) {
          max_ts = ts
          latest = $2
        }
      }
      END {
        if (latest != "") print latest
      }
    ' || true
  return 0
}

_ffprobe_quick_ok() {
  local f="$1"
  [[ -n "$f" && -f "$f" ]] || return 1
  ffprobe -v error \
    -select_streams v:0 \
    -show_entries stream=codec_name,width,height,avg_frame_rate,nb_frames \
    -show_entries format=duration \
    -of default=nw=1:nk=1 \
    "$f" >/dev/null 2>&1
}

_cleanup_unreadable_latest_output() {
  local last_mp4
  last_mp4="$(_latest_mp4_in_output)"
  if [[ -z "$last_mp4" ]]; then
    echo "[CHECK] no output mp4 found to validate."
    return 0
  fi

  if _ffprobe_quick_ok "$last_mp4"; then
    echo "[CHECK] last output readable: $last_mp4"
    return 0
  fi

  echo "[CHECK] last output unreadable, removing before restart: $last_mp4"
  rm -f -- "$last_mp4" || true
}

_cleanup_partial_outputs() {
  find "$OUTPUT_DIR" -type f \( -name "*.part.mp4" -o -name "*.part.mkv" -o -name "*.part.mov" -o -name "*.part.webm" -o -name "*.part.avi" \) -delete 2>/dev/null || true
}

_latest_output_token() {
  find "$OUTPUT_DIR" -type f -printf '%T@|%s|%p\n' 2>/dev/null \
    | awk -F'|' '
      BEGIN { max_ts = -1; latest = "" }
      {
        ts = $1 + 0
        if (ts > max_ts) {
          max_ts = ts
          latest = $0
        }
      }
      END {
        if (latest != "") print latest
      }
    ' || true
  return 0
}

_run_once_with_watchdog() {
  local child_pid child_pgid self_pgid
  local last_token current_token
  local last_activity_ts now idle_sec

  if command -v setsid >/dev/null 2>&1; then
    setsid "${CMD[@]}" &
  else
    "${CMD[@]}" &
  fi
  child_pid=$!
  child_pgid="$(ps -o pgid= -p "$child_pid" 2>/dev/null | tr -d '[:space:]')"
  self_pgid="$(ps -o pgid= -p "$$" 2>/dev/null | tr -d '[:space:]')"
  if [[ -n "$child_pgid" && -n "$self_pgid" && "$child_pgid" == "$self_pgid" ]]; then
    child_pgid=""
  fi
  CURRENT_CHILD_PID="$child_pid"
  CURRENT_CHILD_PGID="$child_pgid"

  if ! _is_true "$WATCHDOG_ENABLED"; then
    local rc=0
    if wait "$child_pid"; then
      rc=0
    else
      rc=$?
    fi
    CURRENT_CHILD_PID=""
    CURRENT_CHILD_PGID=""
    return "$rc"
  fi

  last_token="$(_latest_output_token)"
  last_activity_ts=$(date +%s)

  while kill -0 "$child_pid" 2>/dev/null; do
    sleep "$WATCHDOG_POLL_SEC"

    current_token="$(_latest_output_token)"
    if [[ -n "$current_token" && "$current_token" != "$last_token" ]]; then
      last_token="$current_token"
      last_activity_ts=$(date +%s)
      continue
    fi

    now=$(date +%s)
    idle_sec=$((now - last_activity_ts))
    if (( idle_sec >= WATCHDOG_IDLE_SEC )); then
      echo "[WATCHDOG] no output activity for ${idle_sec}s. Killing runner pid=$child_pid pgid=${child_pgid:-n/a}"
      _kill_child_group "$child_pid" "$child_pgid"
      wait "$child_pid" 2>/dev/null || true
      CURRENT_CHILD_PID=""
      CURRENT_CHILD_PGID=""
      return 124
    fi
  done

  local rc=0
  if wait "$child_pid"; then
    rc=0
  else
    rc=$?
  fi
  CURRENT_CHILD_PID=""
  CURRENT_CHILD_PGID=""
  return "$rc"
}

attempt=1
while true; do
  set +e
  _run_once_with_watchdog
  rc=$?
  set -e

  if [[ "$FORCE_STOP" -eq 1 ]]; then
    rm -f -- "$STOP_MARKER" 2>/dev/null || true
    echo "[STOP] forced stop completed."
    exit 130
  fi

  if [[ "$STOP_REQUESTED" -eq 1 ]]; then
    rm -f -- "$STOP_MARKER" 2>/dev/null || true
    echo "[STOP] graceful stop completed (last rc=$rc)."
    exit 0
  fi

  if [[ $rc -eq 0 ]]; then
    exit 0
  fi

  echo "[WARN] sharpen runner exited with rc=$rc (attempt $attempt). Restarting..."
  _cleanup_unreadable_latest_output
  _cleanup_partial_outputs

  if [[ $MAX_RETRIES -ne 0 && $attempt -ge $MAX_RETRIES ]]; then
    echo "[ERR] reached MAX_RETRIES=$MAX_RETRIES, giving up (last rc=$rc)"
    exit $rc
  fi

  attempt=$((attempt+1))
  sleep "$RETRY_SLEEP_SEC"
done
