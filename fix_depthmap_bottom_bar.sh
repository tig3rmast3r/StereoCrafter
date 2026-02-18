#!/usr/bin/env bash
set -euo pipefail

# Fix bottom bright/white bar in depthmap videos by overwriting the last PAD rows
# with a copy of the last "good" row (replicated downward).
#
# Usage:
#   chmod +x fix_depthmap_bottom_bar.sh
#   ./fix_depthmap_bottom_bar.sh /path/in_dir /path/out_dir
#
# Optional:
#   PAD=14 ./fix_depthmap_bottom_bar.sh /path/in_dir /path/out_dir

IN_DIR="${1:?uso: $0 /folder_depth_in /folder_depth_out}"
OUT_DIR="${2:?uso: $0 /folder_depth_in /folder_depth_out}"
PAD="${PAD:-16}"

mkdir -p "$OUT_DIR"
shopt -s nullglob

for inpath in "$IN_DIR"/*.mp4 "$IN_DIR"/*.mkv "$IN_DIR"/*.mov; do
  [ -f "$inpath" ] || continue

  base="$(basename "$inpath")"
  outpath="$OUT_DIR/$base"

  if [ -s "$outpath" ]; then
    echo "[SKIP] $base"
    continue
  fi

  h="$(ffprobe -v error -select_streams v:0 -show_entries stream=height \
      -of default=nw=1:nk=1 "$inpath" | head -n1 || true)"

  if [[ -z "${h:-}" ]] || ! [[ "$h" =~ ^[0-9]+$ ]]; then
    echo "[WARN] $base: ffprobe didn't read height. Skip."
    continue
  fi
  if (( h <= PAD + 1 )); then
    echo "[WARN] $base: height=$h too small for PAD=$PAD. Skip."
    continue
  fi

  top_h=$(( h - PAD ))
  y_good=$(( h - PAD - 1 ))

  echo "[FIX] $base (h=$h, pad_bottom=$PAD, top_h=$top_h, y_good=$y_good)"

  # IMPORTANT:
  # We convert to format=gray before doing a 1-pixel crop.
  # With yuv420p (4:2:0), cropping to height=1 is invalid (chroma subsampling).
  ffmpeg -hide_banner -loglevel error -stats -y \
    -i "$inpath" \
    -filter_complex "\
      [0:v]format=gray,split=2[v0][v1]; \
      [v0]crop=iw:${top_h}:0:0[top]; \
      [v1]crop=iw:1:0:${y_good},scale=iw:${PAD}:flags=neighbor[pad]; \
      [top][pad]vstack=inputs=2[out]" \
    -map "[out]" \
    -c:v libx264 -preset veryfast -crf 1 -pix_fmt yuv420p \
    -movflags +faststart -an \
    "$outpath"

done

echo "Done. Output in: $OUT_DIR"

