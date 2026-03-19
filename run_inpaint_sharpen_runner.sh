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

STOP_REQUESTED=0
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
  echo "[STOP] force stop requested. Killing sharpen runner immediately."
  if [[ -n "$CURRENT_CHILD_PID" ]] && kill -0 "$CURRENT_CHILD_PID" 2>/dev/null; then
    _kill_child_group "$CURRENT_CHILD_PID" "$CURRENT_CHILD_PGID"
  fi
}

trap _request_stop_signal INT TERM

set +e
setsid "${CMD[@]}" &
CURRENT_CHILD_PID=$!
CURRENT_CHILD_PGID="$(ps -o pgid= "$CURRENT_CHILD_PID" 2>/dev/null | tr -d ' ' || true)"
wait "$CURRENT_CHILD_PID"
rc=$?
set -e

CURRENT_CHILD_PID=""
CURRENT_CHILD_PGID=""

if [[ "$STOP_REQUESTED" -eq 1 ]] && [[ -f "$STOP_MARKER" ]]; then
  rm -f -- "$STOP_MARKER" || true
fi

exit "$rc"
