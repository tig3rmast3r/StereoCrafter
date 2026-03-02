#!/usr/bin/env bash
set -euo pipefail

MPV_INPUT_CONF="${MPV_INPUT_CONF:-$HOME/.config/mpv/input.conf}"
MPV_CONFIG_DIR="$(dirname "$MPV_INPUT_CONF")"
MPV_CONF="${MPV_CONF:-$MPV_CONFIG_DIR/mpv.conf}"
STAMP="$(date +%Y%m%d_%H%M%S)"

read -r -d '' FALLBACK_BINDINGS <<'EOF' || true
# StereoCrafter navigation defaults
SPACE cycle pause
UP    playlist-prev
DOWN  playlist-next

LEFT  frame-back-step
RIGHT frame-step

Shift+LEFT  seek -1 exact
Shift+RIGHT seek 1 exact

# Optional duplicates for frame-step
Ctrl+LEFT  frame-back-step
Ctrl+RIGHT frame-step
EOF

if [[ -f "$MPV_INPUT_CONF" ]]; then
  BINDINGS_CONTENT="$(cat "$MPV_INPUT_CONF")"
  SOURCE_DESC="current file ($MPV_INPUT_CONF)"
else
  BINDINGS_CONTENT="$FALLBACK_BINDINGS"
  SOURCE_DESC="built-in fallback profile"
fi

if [[ -f "$MPV_CONF" ]]; then
  MPV_CONF_ORIG="$(cat "$MPV_CONF")"
else
  MPV_CONF_ORIG=""
fi

# Remove previous keep-open/idle keys and enforce the desired playback behavior.
MPV_CONF_CLEANED="$(printf '%s\n' "$MPV_CONF_ORIG" | awk '!/^[[:space:]]*(keep-open|idle)[[:space:]]*=/{print}')"
if [[ -n "${MPV_CONF_CLEANED//$'\n'/}" ]]; then
  UPDATED_MPV_CONF="${MPV_CONF_CLEANED}"$'\n'
else
  UPDATED_MPV_CONF=""
fi
UPDATED_MPV_CONF+="# StereoCrafter playback behavior"$'\n'
UPDATED_MPV_CONF+="keep-open=always"$'\n'
UPDATED_MPV_CONF+="idle=yes"$'\n'

echo "========================================"
echo " MPV Rebind Utility"
echo "========================================"
echo "[WARN] This will write key bindings to:"
echo "       $MPV_INPUT_CONF"
echo "[WARN] It will also update MPV playback behavior in:"
echo "       $MPV_CONF"
echo "       - keep-open=always"
echo "       - idle=yes"
echo
echo "[INFO] Source profile: $SOURCE_DESC"
echo "[INFO] Bindings that will be applied:"
echo
printf '%s\n' "$BINDINGS_CONTENT" | sed '/^[[:space:]]*$/d' | sed 's/^/  - /'
echo
echo "[NOTE] Left-click CSV logging is handled by:"
echo "       Utilities/lua/sbs_click_logger.lua"
echo "       (MBTN_LEFT forced binding inside Lua script)"
echo

read -r -p "Proceed with rebind? [y/N]: " ANSWER
ANSWER="${ANSWER:-N}"
case "${ANSWER,,}" in
  y|yes)
    ;;
  *)
    echo "[INFO] Aborted. No changes applied."
    exit 0
    ;;
esac

mkdir -p "$MPV_CONFIG_DIR"

if [[ -f "$MPV_INPUT_CONF" ]]; then
  BACKUP_PATH="${MPV_INPUT_CONF}.bak_${STAMP}"
  cp -a "$MPV_INPUT_CONF" "$BACKUP_PATH"
  echo "[INFO] Backup created: $BACKUP_PATH"
fi

printf '%s\n' "$BINDINGS_CONTENT" > "$MPV_INPUT_CONF"
echo "[OK] MPV key bindings written to: $MPV_INPUT_CONF"

if [[ -f "$MPV_CONF" ]]; then
  MPV_CONF_BACKUP="${MPV_CONF}.bak_${STAMP}"
  cp -a "$MPV_CONF" "$MPV_CONF_BACKUP"
  echo "[INFO] Backup created: $MPV_CONF_BACKUP"
fi

printf '%s' "$UPDATED_MPV_CONF" > "$MPV_CONF"
echo "[OK] MPV playback behavior updated in: $MPV_CONF"
