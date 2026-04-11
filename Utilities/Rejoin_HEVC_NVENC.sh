#!/usr/bin/env bash
set -euo pipefail

# --- CONFIG ---
DIR_SBS="${DIR_SBS:-./work/sbs}"
PATTERN="${PATTERN:-*_sbs.mp4}"
OUT="${OUT:-./work/final/final_sbs_1080_hevc_nvenc.mp4}"
FFMPEG_BIN="${FFMPEG_BIN:-ffmpeg}"
PRESET="${PRESET:-p7}"
ENCODER="${ENCODER:-hevc_nvenc}"
QUALITY_FLAG="${QUALITY_FLAG:-cq}"
QUALITY_VALUE="${QUALITY_VALUE:-}"
CQ="${CQ:-12}"
CRF="${CRF:-12}"
PIX_FMT="${PIX_FMT:-yuv420p}"
EXTRA_ARGS="${EXTRA_ARGS:--tune hq -rc vbr -b:v 0 -multipass fullres -spatial_aq 1 -temporal_aq 1 -aq-strength 12 -rc-lookahead 32 -bf 3}"
VF="${VF:-pad=iw:max(ih\,1080):0:(max(ih\,1080)-ih)/2:black,crop=iw:1080:0:(ih-1080)/2}"

if [[ -z "$QUALITY_VALUE" ]]; then
  if [[ "$QUALITY_FLAG" == "cq" ]]; then
    QUALITY_VALUE="$CQ"
  else
    QUALITY_VALUE="$CRF"
  fi
fi

if ! command -v "$FFMPEG_BIN" >/dev/null 2>&1; then
  echo "[ERR] ffmpeg not found: $FFMPEG_BIN" >&2
  exit 1
fi

if [[ ! -d "$DIR_SBS" ]]; then
  echo "[ERR] folder not found: $DIR_SBS" >&2
  exit 1
fi

mapfile -d '' -t file_names < <(
  find "$DIR_SBS" -maxdepth 1 \( -type f -o -type l \) -name "$PATTERN" -printf '%f\0' | sort -z
)

if (( ${#file_names[@]} == 0 )); then
  echo "[ERR] no files found in '$DIR_SBS' with pattern '$PATTERN'" >&2
  exit 1
fi

concat_list=""
valid_count=0
for file_name in "${file_names[@]}"; do
  file_path="${DIR_SBS%/}/$file_name"
  if [[ ! -f "$file_path" ]]; then
    echo "[WARN] skipping unreadable input (broken link?): $file_path" >&2
    continue
  fi
  escaped_path="$(printf "%s" "$file_path" | sed "s/'/'\\\\''/g")"
  concat_list+="file 'file:$escaped_path'"$'\n'
  valid_count=$((valid_count + 1))
done

if (( valid_count == 0 )); then
  echo "[ERR] no readable files found in '$DIR_SBS' with pattern '$PATTERN'" >&2
  exit 1
fi

ff_args=(
  "-hide_banner" "-y"
  "-f" "concat" "-safe" "0"
  "-protocol_whitelist" "file,pipe,crypto,data"
  "-i" "pipe:0"
)

if [[ -n "$VF" ]]; then
  ff_args+=("-vf" "$VF")
fi

ff_args+=(
  "-c:v" "$ENCODER"
)

if [[ -n "$PRESET" ]]; then
  ff_args+=("-preset" "$PRESET")
fi
if [[ -n "$QUALITY_FLAG" && -n "$QUALITY_VALUE" ]]; then
  ff_args+=("-${QUALITY_FLAG}" "$QUALITY_VALUE")
fi
if [[ -n "$PIX_FMT" ]]; then
  ff_args+=("-pix_fmt" "$PIX_FMT")
fi

ff_args+=(
  "-an"
)

if [[ -n "$EXTRA_ARGS" ]]; then
  read -r -a extra_arr <<<"$EXTRA_ARGS"
  ff_args+=("${extra_arr[@]}")
fi

ff_args+=("$OUT")

mkdir -p "$(dirname "$OUT")"

printf "%s" "$concat_list" | "$FFMPEG_BIN" "${ff_args[@]}"
