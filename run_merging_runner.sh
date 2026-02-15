#!/usr/bin/env bash
set -u  # don't use -e because we want to handle non-zero exit codes ourselves

# -----------------------------
# User knobs (edit these)
# -----------------------------
INPAINTED_DIR="./work/output"
SPLAT_DIR="./work/splat/hires"          # NOTE: this is the hires folder (runner expects splatted2/4 inside here)
ORIGINAL_DIR="./work/seg"
OUT_DIR="./work/sbs"

# Optional: folder where *_replace_mask.mkv/.mp4 live (leave empty to disable / fallback to SPLAT_DIR)
REPLACE_MASK_DIR="./work/mask"
USE_REPLACE_MASK=1   # 1 = enable (will use REPLACE_MASK_DIR if set, else fallback to SPLAT_DIR)

CHUNK_SIZE=20
RETRY_PER_CLIP=1

# Restart policy for hard crashes (segfault, OOM-kill, etc.)
MAX_RESTARTS=10
SLEEP_BETWEEN_RESTARTS=5

# Optional: activate venv/conda here
# source /path/to/venv/bin/activate
# conda activate stereocrafter

PYTHON_BIN="${PYTHON_BIN:-python3}"
RUNNER="${RUNNER:-./batch_merging_runner.py}"

LOG_DIR="${LOG_DIR:-./logs}"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/merging_$(date +%Y%m%d_%H%M%S).log"

# -----------------------------
# Helpers
# -----------------------------
is_hard_crash_code () {
  local code="$1"
  # 137 = killed (often OOM), 139 = segfault, 143 = SIGTERM, 134 = abort
  case "$code" in
    134|137|139|143) return 0 ;;
    *) return 1 ;;
  esac
}

build_cmd () {
  local cmd=("$PYTHON_BIN" "$RUNNER"
    --inpainted "$INPAINTED_DIR"
    --splat "$SPLAT_DIR"
    --original "$ORIGINAL_DIR"
    --out "$OUT_DIR"
    --chunk-size "$CHUNK_SIZE"
    --retry "$RETRY_PER_CLIP"
  )

  if [[ -n "${REPLACE_MASK_DIR}" && "${REPLACE_MASK_DIR}" != "/path/to/replace_masks" ]]; then
    cmd+=( --replace-mask "$REPLACE_MASK_DIR" )
  fi
  if [[ "$USE_REPLACE_MASK" == "1" ]]; then
    cmd+=( --use-replace-mask )
  fi

  printf '%q ' "${cmd[@]}"
}

# -----------------------------
# Main restart loop
# -----------------------------
restarts=0
while true; do
  echo "=== [$(date '+%F %T')] START merging (restart ${restarts}/${MAX_RESTARTS}) ===" | tee -a "$LOG_FILE"
  echo "CMD: $(build_cmd)" | tee -a "$LOG_FILE"

  # Run and capture exit code
  set +e
  eval "$(build_cmd)" 2>&1 | tee -a "$LOG_FILE"
  code=${PIPESTATUS[0]}
  set -e

  echo "=== [$(date '+%F %T')] EXIT code=${code} ===" | tee -a "$LOG_FILE"

  if [[ "$code" == "0" ]]; then
    echo "Done." | tee -a "$LOG_FILE"
    exit 0
  fi

  # If runner completed but had failed clips, you may want to stop here (exit 1).
  # If you prefer automatic restarts even on "failed clips", change this behavior.
  if [[ "$code" == "1" ]]; then
    echo "Runner finished but some clips failed (exit 1). Not restarting by default." | tee -a "$LOG_FILE"
    exit 1
  fi

  if is_hard_crash_code "$code"; then
    restarts=$((restarts + 1))
    if (( restarts > MAX_RESTARTS )); then
      echo "Too many restarts (${restarts}). Giving up." | tee -a "$LOG_FILE"
      exit "$code"
    fi
    echo "Hard crash (code ${code}). Restarting in ${SLEEP_BETWEEN_RESTARTS}s..." | tee -a "$LOG_FILE"
    sleep "$SLEEP_BETWEEN_RESTARTS"
    # Runner will resume thanks to SKIP_EXISTING/VALIDATE_EXISTING
    continue
  fi

  echo "Non-zero exit code ${code}. Not a recognized hard crash. Exiting." | tee -a "$LOG_FILE"
  exit "$code"
done
