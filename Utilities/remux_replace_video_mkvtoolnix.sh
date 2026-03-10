#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  remux_replace_video_mkvtoolnix.sh \
    --source /path/original_source.mkv \
    --video3d /path/final_3d_video.mp4 \
    [--output /path/final_remuxed.mkv] \
    [--mkvmerge-bin mkvmerge] \
    [--ffprobe-bin ffprobe] \
    [--overwrite]

Env fallback:
  SOURCE_FILE, VIDEO_3D_FILE, OUT_FILE, MKVMERGE_BIN, FFPROBE_BIN, OVERWRITE

Behavior:
  - Takes VIDEO stream from --video3d
  - Takes NON-VIDEO streams (audio/subtitles/chapters/attachments/tags) from --source
  - Writes a new MKV file
EOF
}

SOURCE_FILE="${SOURCE_FILE:-}"
VIDEO_3D_FILE="${VIDEO_3D_FILE:-}"
OUT_FILE="${OUT_FILE:-}"
MKVMERGE_BIN="${MKVMERGE_BIN:-mkvmerge}"
FFPROBE_BIN="${FFPROBE_BIN:-ffprobe}"
OVERWRITE="${OVERWRITE:-0}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source)
      SOURCE_FILE="${2:-}"
      shift 2
      ;;
    --video3d|--video-3d)
      VIDEO_3D_FILE="${2:-}"
      shift 2
      ;;
    --output|--out)
      OUT_FILE="${2:-}"
      shift 2
      ;;
    --mkvmerge-bin)
      MKVMERGE_BIN="${2:-}"
      shift 2
      ;;
    --ffprobe-bin)
      FFPROBE_BIN="${2:-}"
      shift 2
      ;;
    --overwrite|-y)
      OVERWRITE="1"
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "[ERR] Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$SOURCE_FILE" || -z "$VIDEO_3D_FILE" ]]; then
  echo "[ERR] --source and --video3d are required." >&2
  usage >&2
  exit 2
fi

if [[ ! -f "$SOURCE_FILE" ]]; then
  echo "[ERR] Source file not found: $SOURCE_FILE" >&2
  exit 2
fi
if [[ ! -f "$VIDEO_3D_FILE" ]]; then
  echo "[ERR] 3D video file not found: $VIDEO_3D_FILE" >&2
  exit 2
fi

if ! command -v "$MKVMERGE_BIN" >/dev/null 2>&1; then
  echo "[ERR] mkvmerge not found: $MKVMERGE_BIN" >&2
  exit 2
fi

if [[ -z "$OUT_FILE" ]]; then
  src_stem="$(basename "${SOURCE_FILE%.*}")"
  out_dir="$(cd -- "$(dirname -- "$VIDEO_3D_FILE")" && pwd -P)"
  OUT_FILE="${out_dir}/${src_stem}_3D_remux.mkv"
fi

mkdir -p "$(dirname -- "$OUT_FILE")"

if [[ -e "$OUT_FILE" ]]; then
  if [[ "$OVERWRITE" != "1" && "$OVERWRITE" != "yes" && "$OVERWRITE" != "true" ]]; then
    echo "[ERR] Output already exists (use --overwrite): $OUT_FILE" >&2
    exit 2
  fi
  rm -f -- "$OUT_FILE"
fi

if [[ "${OUT_FILE##*.}" != "mkv" ]]; then
  echo "[WARN] Output extension is not .mkv: $OUT_FILE"
fi

count_video_tracks() {
  local file="$1"
  "$FFPROBE_BIN" -v error -select_streams v \
    -show_entries stream=index -of csv=p=0 "$file" 2>/dev/null | wc -l
}

if command -v "$FFPROBE_BIN" >/dev/null 2>&1; then
  src_vc="$(count_video_tracks "$SOURCE_FILE" | tr -d '[:space:]')"
  out_vc="$(count_video_tracks "$VIDEO_3D_FILE" | tr -d '[:space:]')"
  [[ -n "$src_vc" ]] || src_vc="0"
  [[ -n "$out_vc" ]] || out_vc="0"
  if [[ "$out_vc" != "1" ]]; then
    echo "[WARN] 3D input has $out_vc video tracks (expected 1). First file's tracks policy will apply."
  fi
  if [[ "$src_vc" -gt 1 ]]; then
    echo "[WARN] Source has $src_vc video tracks. All source video tracks will be dropped."
  fi
fi

cmd=(
  "$MKVMERGE_BIN"
  -o "$OUT_FILE"
  --no-audio --no-subtitles --no-buttons --no-chapters --no-attachments --no-global-tags --no-track-tags "$VIDEO_3D_FILE"
  --no-video "$SOURCE_FILE"
)

echo "[INFO] Remux command:"
printf ' %q' "${cmd[@]}"
echo

"${cmd[@]}"

echo "[OK] Remux complete:"
echo "     $OUT_FILE"
