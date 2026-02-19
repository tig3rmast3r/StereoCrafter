#!/usr/bin/env bash
set -euo pipefail

# Edit these paths/values as needed.
INPUT_DIR="./work/splat/"
INPUT_VIDEO=""                       # if set (non-empty), overrides INPUT_DIR
OUTPUT_DIR="./work/output/"
GLOB="*.mp4"

HIRES_BLEND_FOLDER=""                # optional
OFFLOAD_TYPE="model"                 # none | model | sequential

TILE_NUM=2
FRAMES_CHUNK=50
OVERLAP=4
ORIGINAL_INPUT_BLEND_STRENGTH=0
OUTPUT_CRF=1
PROCESS_LENGTH=-1

# Steps control:
# - If you want dynamic steps from sharpness.csv, set NO_SHARPNESS_CSV=0 and (optionally) SHARPNESS_BASE.
# - If you want fixed steps for all files, set NO_SHARPNESS_CSV=1 and FIXED_STEPS.
NO_SHARPNESS_CSV=0
SHARPNESS_BASE="./work/"                    # folder containing sharpness.csv; empty => defaults to input folder
FIXED_STEPS=8

# Mask settings
MASK_INITIAL_THRESHOLD=0.3
MASK_MORPH_KERNEL_SIZE=0.0
MASK_DILATE_KERNEL_SIZE=5
MASK_BLUR_KERNEL_SIZE=10

ENABLE_POST_INPAINTING_BLEND=0        # 1 to enable
DISABLE_COLOR_TRANSFER=1              # 1 to disable

SKIP_EXISTING=1
MOVE_FAILED=1
MOVE_FINISHED=0
FAILED_SUBDIR="failed"
FINISHED_SUBDIR="finished"

DISABLE_DYNAMIC_CHUNK=1

DEBUG=0

# --- runner command (edit freely) ---
CMD=(python3 batch_inpainting_runner.py)

if [[ -n "$INPUT_VIDEO" ]]; then
  CMD+=(--input_video "$INPUT_VIDEO")
else
  CMD+=(--input_dir "$INPUT_DIR" --glob "$GLOB")
fi

CMD+=(--output_dir "$OUTPUT_DIR"
     --tile_num "$TILE_NUM"
     --frames_chunk "$FRAMES_CHUNK"
     --overlap "$OVERLAP"
     --original_input_blend_strength "$ORIGINAL_INPUT_BLEND_STRENGTH"
     --output_crf "$OUTPUT_CRF"
     --process_length "$PROCESS_LENGTH"
     --offload_type "$OFFLOAD_TYPE"
     --hires_blend_folder "$HIRES_BLEND_FOLDER"
     --mask_initial_threshold "$MASK_INITIAL_THRESHOLD"
     --mask_morph_kernel_size "$MASK_MORPH_KERNEL_SIZE"
     --mask_dilate_kernel_size "$MASK_DILATE_KERNEL_SIZE"
     --mask_blur_kernel_size "$MASK_BLUR_KERNEL_SIZE"
     --fixed_steps "$FIXED_STEPS"
     --failed_subdir "$FAILED_SUBDIR"
     --finished_subdir "$FINISHED_SUBDIR"
)

if [[ "$NO_SHARPNESS_CSV" == "1" ]]; then
  CMD+=(--no_sharpness_csv)
else
  if [[ -n "$SHARPNESS_BASE" ]]; then
    CMD+=(--sharpness_base "$SHARPNESS_BASE")
  fi
fi

if [[ "$ENABLE_POST_INPAINTING_BLEND" == "1" ]]; then CMD+=(--enable_post_inpainting_blend); fi
if [[ "$DISABLE_COLOR_TRANSFER" == "1" ]]; then CMD+=(--disable_color_transfer); fi
if [[ "$SKIP_EXISTING" == "1" ]]; then CMD+=(--skip_existing); fi
if [[ "$MOVE_FAILED" == "1" ]]; then CMD+=(--move_failed); fi
if [[ "$MOVE_FINISHED" == "1" ]]; then CMD+=(--move_finished); fi
if [[ "$DEBUG" == "1" ]]; then CMD+=(--debug); fi
if [[ "$DISABLE_DYNAMIC_CHUNK" == "1" ]]; then CMD+=(--no_dynamic_chunk); fi

echo "[CMD] ${CMD[*]}"

# Retry on ANY non-zero exit (ignore specific crash codes).
# Set MAX_RETRIES=0 for infinite restarts.
MAX_RETRIES=0
attempt=1
while true; do
  set +e
  "${CMD[@]}"
  rc=$?
  set -e

  if [[ $rc -eq 0 ]]; then
    exit 0
  fi

  echo "[WARN] runner exited with rc=$rc (attempt $attempt). Restarting..."

  if [[ $MAX_RETRIES -ne 0 && $attempt -ge $MAX_RETRIES ]]; then
    echo "[ERR] reached MAX_RETRIES=$MAX_RETRIES, giving up (last rc=$rc)"
    exit $rc
  fi

  attempt=$((attempt+1))
  sleep 2
done
