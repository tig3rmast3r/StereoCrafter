#!/usr/bin/env bash
set -euo pipefail

IN_DIR="${1:?uso: $0 IN_DIR OUT_DIR [SCALE] [MODEL] [TILE] [DEST_WxH] [JOBS] [RETRIES]}"
OUT_DIR="${2:?uso: $0 IN_DIR OUT_DIR [SCALE] [MODEL] [TILE] [DEST_WxH] [JOBS] [RETRIES]}"
SCALE="${3:-2}"
MODEL="${4:-realesr-animevideov3-x2}"
TILE="${5:-256}"
DEST="${6:-}"          # es: 1920x1152
MAX_JOBS="${7:-4}"     # default 4
MAX_RETRIES="${8:-3}"  # default 3 retry per file

REALESRGAN_BIN="${REALESRGAN_BIN:-$HOME/tools/realesrgan/realesrgan-ncnn-vulkan}"

mkdir -p "$OUT_DIR"

mapfile -d '' -t FILES < <(find "$IN_DIR" -maxdepth 1 -type f \( -iname '*.mp4' -o -iname '*.mkv' -o -iname '*.mov' -o -iname '*.avi' \) -print0 | sort -z -V)
(( ${#FILES[@]} > 0 )) || { echo "Nessun video in: $IN_DIR" >&2; exit 1; }

FIRST="${FILES[0]}"
FPS="$(ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate -of default=nw=1:nk=1 "$FIRST")"
[[ -n "$FPS" ]] || { echo "Impossibile leggere FPS da: $FIRST" >&2; exit 1; }

VF=""
if [[ -n "$DEST" ]]; then
  [[ "$DEST" =~ ^[0-9]+x[0-9]+$ ]] || { echo "DEST_WxH non valido: $DEST (es: 1920x1152)" >&2; exit 1; }
  W="${DEST%x*}"; H="${DEST#*x}"
  VF="scale=${W}:${H}:flags=lanczos"
fi

echo "FPS (dal primo file): $FPS"
echo "Scale: $SCALE  Model: $MODEL  Tile: $TILE"
[[ -n "$DEST" ]] && echo "Resize finale: $DEST (stretch libero)"
echo "Parallel jobs: $MAX_JOBS"
echo "Retries per file: $MAX_RETRIES"
echo

is_good_video() {
  local f="$1"
  [[ -s "$f" ]] || return 1
  local sz
  sz="$(stat -c '%s' "$f" 2>/dev/null || echo 0)"
  (( sz > 50000 )) || return 1

  ffprobe -v error \
    -select_streams v:0 \
    -show_entries stream=codec_name \
    -of csv=p=0 \
    "$f" >/dev/null 2>&1 || return 1

  local dur
  dur="$(ffprobe -v error -show_entries format=duration -of default=nw=1:nk=1 "$f" 2>/dev/null || true)"
  [[ -n "$dur" ]] || return 1
  awk -v d="$dur" 'BEGIN{exit !(d>0.05)}' || return 1
  return 0
}

tail_err() {
  # Print only the last lines if there's something
  local file="$1" label="${2:-ERR}"
  [[ -s "$file" ]] || return 0
  echo "[$label] tail:"
  tail -n 12 "$file" | sed "s/^/[$label] /"
}

process_one() {
  local IN="$1"
  local base stem OUT_VIDEO tmp_out TMP attempt ok log_esr log_extract log_encode rc i f n

  base="$(basename "$IN")"
  stem="${base%.*}"
  OUT_VIDEO="$OUT_DIR/${stem}.mp4"

  if [[ -s "$OUT_VIDEO" ]]; then
    echo "[SKIP] $base"
    return 0
  fi

  attempt=1
  ok=0

  while (( attempt <= MAX_RETRIES )); do
    echo "[RUN ] $base (try $attempt/$MAX_RETRIES)"

    TMP="$(mktemp -d)"
    cleanup() { rm -rf "$TMP"; }
    trap cleanup EXIT

    mkdir -p "$TMP/in" "$TMP/out" "$TMP/outseq"
    log_esr="$TMP/realesrgan.log"
    log_extract="$TMP/extract.log"
    log_encode="$TMP/encode.log"

    # Temporary output with .mp4 extension (ffmpeg otherwise can't mux here)
    tmp_out="${OUT_VIDEO}.part.$$.$attempt.mp4"
    rm -f "$tmp_out" "$OUT_VIDEO" 2>/dev/null || true

    # Disable -e to handle return codes and retries
    set +e

    # Extract frames
    ffmpeg -hide_banner -loglevel error -y -i "$IN" \
      -vf "fps=$FPS" \
      "$TMP/in/%08d.png" >"$log_extract" 2>&1
    rc=$?
    set -e
    if (( rc != 0 )); then
      echo "[WARN] extract failed rc=$rc"
      tail_err "$log_extract" "FFMPEG"
      rm -f "$tmp_out" "$OUT_VIDEO" 2>/dev/null || true
      rm -rf "$TMP"; trap - EXIT
      ((attempt++))
      continue
    fi

    # Upscale (silenzioso: log su file)
    set +e
    "$REALESRGAN_BIN" -i "$TMP/in" -o "$TMP/out" -n "$MODEL" -s "$SCALE" -t "$TILE" >"$log_esr" 2>&1
    rc=$?
    set -e
    if (( rc != 0 )); then
      echo "[WARN] realesrgan failed rc=$rc"
      tail_err "$log_esr" "ESRGAN"
      rm -f "$tmp_out" "$OUT_VIDEO" 2>/dev/null || true
      rm -rf "$TMP"; trap - EXIT
      ((attempt++))
      continue
    fi

    # Normalizza output in sequenza %08d.png
    i=1
    while IFS= read -r -d '' f; do
      printf -v n "%08d" "$i"
      cp -f "$f" "$TMP/outseq/$n.png"
      ((i++))
    done < <(find "$TMP/out" -maxdepth 1 -type f -iname '*.png' -print0 | sort -z -V)

    if (( i <= 1 )); then
      echo "[WARN] no output frames produced"
      tail_err "$log_esr" "ESRGAN"
      rm -f "$tmp_out" "$OUT_VIDEO" 2>/dev/null || true
      rm -rf "$TMP"; trap - EXIT
      ((attempt++))
      continue
    fi

    # Encode
    set +e
    if [[ -n "$VF" ]]; then
      ffmpeg -hide_banner -loglevel error -y \
        -framerate "$FPS" -i "$TMP/outseq/%08d.png" \
        -vf "$VF" \
        -c:v h264_nvenc -preset medium -rc constqp -qp 0 -profile:v main -pix_fmt yuv420p \
        "$tmp_out" >"$log_encode" 2>&1
    else
      ffmpeg -hide_banner -loglevel error -y \
        -framerate "$FPS" -i "$TMP/outseq/%08d.png" \
        -c:v h264_nvenc -preset medium -rc constqp -qp 0 -profile:v main -pix_fmt yuv420p \
        "$tmp_out" >"$log_encode" 2>&1
    fi
    rc=$?
    set -e
    if (( rc != 0 )); then
      echo "[WARN] encode failed rc=$rc"
      tail_err "$log_encode" "FFMPEG"
      rm -f "$tmp_out" "$OUT_VIDEO" 2>/dev/null || true
      rm -rf "$TMP"; trap - EXIT
      ((attempt++))
      continue
    fi

    # Validate output
    if ! is_good_video "$tmp_out"; then
      echo "[WARN] output invalid/small -> retry"
      # Keeping logs quiet; uncomment to see encode tail:
      # tail_err "$log_encode" "FFMPEG"
      rm -f "$tmp_out" "$OUT_VIDEO" 2>/dev/null || true
      rm -rf "$TMP"; trap - EXIT
      ((attempt++))
      continue
    fi

    mv -f "$tmp_out" "$OUT_VIDEO"
    ok=1

    rm -rf "$TMP"
    trap - EXIT

    echo "[OK  ] $base"
    break
  done

  if (( ok == 0 )); then
    echo "[FAIL] $base (after $MAX_RETRIES tries)"
    rm -f "$OUT_VIDEO" 2>/dev/null || true
    return 1
  fi

  return 0
}

# job control
pids=()
fail=0

trap 'echo "[ABORT] stopping..."; for pid in "${pids[@]:-}"; do kill "$pid" 2>/dev/null || true; done; exit 130' INT TERM

for IN in "${FILES[@]}"; do
  while (( ${#pids[@]} >= MAX_JOBS )); do
    if ! wait -n; then fail=1; fi
    alive=()
    for pid in "${pids[@]}"; do
      if kill -0 "$pid" 2>/dev/null; then alive+=("$pid"); fi
    done
    pids=("${alive[@]}")
  done

  ( process_one "$IN" ) &
  pids+=("$!")
done

for pid in "${pids[@]}"; do
  if ! wait "$pid"; then fail=1; fi
done

if (( fail != 0 )); then
  echo "[DONE] completato con errori."
  exit 1
fi

echo "Fatto. Output in: $OUT_DIR"

