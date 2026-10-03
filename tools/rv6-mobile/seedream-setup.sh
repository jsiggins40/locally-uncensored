#!/usr/bin/env bash
# One-shot setup of the ☁️ Seedream (5.0 Pro Edit, Atlas Cloud) mode on the
# phone edit page: saves the Atlas API key on this VM, installs the latest
# page, restarts ComfyUI, checks the cloud mode is live and prints the link.
#   bash <(curl -fsSL https://raw.githubusercontent.com/jsiggins40/locally-uncensored/rv6-mobile-editor/tools/rv6-mobile/seedream-setup.sh)
set -euo pipefail

SRC="https://raw.githubusercontent.com/jsiggins40/locally-uncensored/rv6-mobile-editor/tools/rv6-mobile"
KEY_FILE="$HOME/.atlascloud_key"

echo "== Atlas Cloud API key"
if [ -s "$KEY_FILE" ]; then
  echo "A key is already saved. Paste a new one to replace it, or just press Enter to keep it."
else
  echo "Get it at atlascloud.ai -> Console -> API Keys, then paste it below and press Enter."
fi
read -rp "Atlas key: " key </dev/tty || key=""
# Drop spaces and the bracketed-paste markers ([200~ ... [201~) some terminals add
key="$(printf '%s' "$key" | sed 's/\x1b\[20[01]~//g; s/\[20[01]~//g' | tr -d '[:space:][:cntrl:]')"
clear 2>/dev/null || true  # keep the key off the screen
[ -n "$key" ] && echo "Got a key ending in ...${key: -4} (${#key} characters)"
if [ -n "$key" ]; then
  printf '%s' "$key" > "$KEY_FILE"; chmod 600 "$KEY_FILE"; echo "Saved to $KEY_FILE"
elif [ ! -s "$KEY_FILE" ]; then
  echo "No key given, stopping. Run this again when you have it."; exit 1
fi

echo "== Checking the key with Atlas (free, makes no image)"
code=$(curl -s -o /tmp/atlas_check.txt -w '%{http_code}' -m 20 \
  -H "Authorization: Bearer $(cat "$KEY_FILE")" https://api.atlascloud.ai/v1/models || echo 000)
case "$code" in
  200) echo "Atlas didn't reject the key." ;;
  401|403) echo "Atlas REJECTED the key ($code). Copy it again from the Atlas console and re-run this."; exit 1 ;;
  *) echo "Could not confirm the key (Atlas answered $code); continuing. A wrong key shows as an error on the page." ;;
esac
rm -f /tmp/atlas_check.txt

echo "== Installing the latest page and restarting ComfyUI"
RV6M_NO_PROMPT=1 bash <(curl -fsSL "$SRC/install.sh") || echo "(page installer reported a problem above; checking Seedream anyway)"

echo "== Checking the Seedream mode"
for i in $(seq 1 30); do
  st=$(curl -s -m 5 localhost:8188/rv6m/cloud/status || true)
  [ -n "$st" ] && break; sleep 2
done
if grep -q '"configured": true' <<<"$st"; then
  echo "SEEDREAM: ready ($(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["model"])' "$st"))"
else
  echo "SEEDREAM: not ready. ComfyUI said: ${st:-nothing}"
  tmux capture-pane -pt comfy 2>/dev/null | grep -i "rv6_mobile\|error" | tail -8 || true
  exit 1
fi

HOST=$(tailscale status --json 2>/dev/null | python3 -c 'import json,sys; print(json.load(sys.stdin)["Self"]["DNSName"].rstrip("."))' 2>/dev/null || true)
echo
echo "=============================================================="
echo " DONE. On your phone (Tailscale app connected) open:"
echo "   https://${HOST:-<your-vm>.ts.net}/extensions/rv6_mobile/edit.html"
echo " Reload the page, then tap ☁️ Seedream."
echo " Top up Atlas credit first: atlascloud.ai -> Billing (min \$25)."
echo "=============================================================="
