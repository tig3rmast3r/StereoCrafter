#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   chmod +x fix_mask_top_bottom_black.sh
#   ./fix_mask_top_bottom_black.sh /path/in_dir /path/out_dir
#
# Optional:
#   PAD_TOP=14 PAD_BOTTOM=14 ./fix_mask_top_bottom_black.sh /in /out

IN_DIR="${1:?uso: $0 /folder_mask_in /folder_mask_out}"
OUT_DIR="${2:?uso: $0 /folder_mask_in /folder_mask_out}"

PAD_TOP="${PAD_TOP:-14}"
PAD_BOTTOM="${PAD_BOTTOM:-14}"

mkdir -p "$OUT_DIR"
shopt -s nullglob

for inpath in "$IN_DIR"/*.mkv; do
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

  if (( h <= PAD_TOP + PAD_BOTTOM )); then
    echo "[WARN] $base: height=$h too small for PAD_TOP=$PAD_TOP PAD_BOTTOM=$PAD_BOTTOM. Skip."
    continue
  fi

  y_bottom=$(( h - PAD_BOTTOM ))

  echo "[FIX] $base (h=$h, top=$PAD_TOP, bottom=$PAD_BOTTOM, y_bottom=$y_bottom)"

  ffmpeg -hide_banner -loglevel error -stats -y \
    -i "$inpath" \
    -vf "format=gray,\
drawbox=x=0:y=0:w=iw:h=${PAD_TOP}:color=black:t=fill,\
drawbox=x=0:y=${y_bottom}:w=iw:h=${PAD_BOTTOM}:color=black:t=fill" \
    -c:v libx264 -preset veryfast -qp 0 -pix_fmt gray \
    -an \
    "$outpath"
done

echo "Done. Output in: $OUT_DIR"

