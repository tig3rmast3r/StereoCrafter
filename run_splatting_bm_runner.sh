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

# Tk requires a display. If DISPLAY is missing (headless), run under xvfb when available.
if [ -z "${DISPLAY:-}" ] && command -v xvfb-run >/dev/null 2>&1; then
  exec xvfb-run -a "${CMD[@]}"
else
  exec "${CMD[@]}"
fi
