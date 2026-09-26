#!/usr/bin/env bash
#
# Search Civitai and pull LoRAs into ComfyUI (or Forge).
#
#   bash l.sh wan nsfw          search - prints names, ids and base models
#   bash l.sh 3071631 2609505   download those version ids
#   DEST=~/sd-webui-forge-neo/models/Lora bash l.sh 123   somewhere else
#
# Searching first matters because a LoRA is tied to one base model. A Klein
# adapter will not load on Qwen, a Wan 2.1 one will not load on Wan 2.2, and
# the failure looks like a broken pipeline rather than a wrong file. The
# search prints the base model next to every hit so the choice is made
# before the download rather than after.
#
# Needs an API key in ~/.civitai-token (civitai.com -> account -> API keys).

set -uo pipefail

TOKEN_FILE="$HOME/.civitai-token"
LOG="$HOME/lora-fetch.log"

if [ -n "${DEST:-}" ]; then
  LORA_DIR="$DEST"
elif [ -d "$HOME/ComfyUI/models/loras" ]; then
  LORA_DIR="$HOME/ComfyUI/models/loras"
elif [ -d "$HOME/sd-webui-forge-neo/models/Lora" ]; then
  LORA_DIR="$HOME/sd-webui-forge-neo/models/Lora"
else
  echo "no loras folder found; set DEST=..."; exit 2
fi

[ $# -gt 0 ] || { echo "usage: bash $0 <search words...|versionId...>"; exit 2; }

if [ ! -s "$TOKEN_FILE" ]; then
  cat <<EOF
No Civitai token at $TOKEN_FILE.

Run exactly this one line and paste ONLY the key at the prompt:

  read -p "key: " K && echo -n "\$K" > $TOKEN_FILE && chmod 600 $TOKEN_FILE && echo "stored \${#K} chars"
EOF
  exit 2
fi
TOKEN=$(tr -d '[:space:]' < "$TOKEN_FILE")
mkdir -p "$LORA_DIR"

# All-numeric arguments mean download; anything else is a search.
NUMERIC=1
for a in "$@"; do
  case "$a" in ''|*[!0-9]*) NUMERIC=0;; esac
done

if [ "$NUMERIC" = "0" ]; then
  Q=$(printf '%s ' "$@" | sed 's/ *$//')
  printf '\n=== searching Civitai for "%s" ===\n\n' "$Q"
  curl -fsSL -H "Authorization: Bearer $TOKEN" -G \
    --data-urlencode "query=$Q" --data-urlencode "types=LORA" \
    --data-urlencode "limit=15" --data-urlencode "nsfw=true" \
    "https://civitai.com/api/v1/models" \
  | python3 -c '
import sys, json
try:
    d = json.load(sys.stdin)
except Exception as e:
    sys.exit(f"could not read the reply: {e}")
items = d.get("items", [])
if not items:
    sys.exit("nothing found")
for m in items:
    name = m.get("name", "?")[:44]
    for v in (m.get("versions") or m.get("modelVersions") or [])[:2]:
        # Pulled out of the format call: a backslash inside an f-string
        # expression is a syntax error before Python 3.12, and this has to
        # run on whatever the box happens to ship.
        vid = v.get("id")
        vname = (v.get("name") or "")[:18]
        base = v.get("baseModel", "?")
        files = [f for f in v.get("files", []) if f.get("type") == "Model"]
        mb = (files[0].get("sizeKB", 0) / 1024) if files else 0
        print("%-9s %-24s %6.0fMB  %s / %s" % (vid, base, mb, name, vname))
' || echo "search failed"
  cat <<EOF

Pick the ids whose base model matches what you run:
  Klein images   Flux.2 Klein 9B
  Qwen editing   Qwen Image Edit / Qwen-Image
  Wan video      Wan Video 2.2 (2.1 adapters do not load on 2.2)

Then:  bash $0 <id> <id> ...
EOF
  exit 0
fi

exec > >(tee -a "$LOG") 2>&1
for VID in "$@"; do
  printf '\n=== version %s ===\n' "$VID"
  META=$(curl -fsSL -H "Authorization: Bearer $TOKEN" \
    "https://civitai.com/api/v1/model-versions/$VID" 2>/dev/null)
  if [ -z "$META" ]; then
    echo "  ! cannot read metadata (bad id, or the token lacks access)"
    continue
  fi

  read -r BASE NAME FILE <<EOF
$(printf '%s' "$META" | python3 -c '
import sys, json
d = json.load(sys.stdin)
fs = [f for f in d.get("files", []) if f.get("type") == "Model"] or d.get("files", [])
print(d.get("baseModel","?").replace(" ","_"),
      (d.get("model",{}).get("name","?"))[:40].replace(" ","_"),
      fs[0]["name"] if fs else "")
' 2>/dev/null)
EOF
  echo "  ${NAME//_/ }"
  echo "  base model: ${BASE//_/ }"
  [ -n "${FILE:-}" ] && echo "  file: $FILE"

  OUT="$LORA_DIR/${FILE:-civitai-$VID.safetensors}"
  if [ -s "$OUT" ]; then echo "  already have it"; continue; fi

  echo "  downloading..."
  if curl -fL --progress-bar -H "Authorization: Bearer $TOKEN" \
       "https://civitai.com/api/download/models/$VID" -o "$OUT.part"; then
    mv "$OUT.part" "$OUT"
    echo "  -> $OUT ($(du -h "$OUT" | cut -f1))"
  else
    rm -f "$OUT.part"
    echo "  ! download failed"
  fi
done

printf '\n=== in %s ===\n' "$LORA_DIR"
ls -la "$LORA_DIR" 2>/dev/null | grep -v '^total' || echo "(empty)"
cat <<'EOF'

ComfyUI rescans its folders per request, so the edit form picks these up
on its next restart:

  bash ~/e.sh
EOF
