#!/usr/bin/env bash
#
# Fetch Civitai LoRAs into Forge's models/Lora by model-version id.
#
#   bash lora.sh 2584502 2609505 ...
#
# The ids are VERSION ids, not model ids: on a Civitai model page they are the
# number in ?modelVersionId=... for the version you actually want. A model id
# alone is ambiguous once a LoRA has more than one version, and picking the
# wrong one is how you end up with a 4B adapter that will not load on 9B.
#
# Needs a Civitai API key in ~/.civitai-token (civitai.com -> account -> API
# keys). Downloads are refused without one.

set -uo pipefail

LORA_DIR="$HOME/sd-webui-forge-neo/models/Lora"
TOKEN_FILE="$HOME/.civitai-token"
LOG="$HOME/lora-fetch.log"
exec > >(tee -a "$LOG") 2>&1

[ $# -gt 0 ] || { echo "usage: bash $0 <versionId> [versionId ...]"; exit 2; }

if [ ! -s "$TOKEN_FILE" ]; then
  cat <<EOF
No Civitai token at $TOKEN_FILE.

Get one at https://civitai.com/user/account (API Keys), then run exactly:

  read -p "key: " K && echo -n "\$K" > $TOKEN_FILE && chmod 600 $TOKEN_FILE && echo "stored \${#K} chars"

and run this script again.
EOF
  exit 2
fi

TOKEN=$(tr -d '[:space:]' < "$TOKEN_FILE")
mkdir -p "$LORA_DIR"

for VID in "$@"; do
  printf '\n=== version %s ===\n' "$VID"

  # Ask the API what this version is before pulling it: the filename, and more
  # importantly the base model, so a 4B or FLUX.1 adapter is caught here rather
  # than by a loader error hours later.
  META=$(curl -fsSL -H "Authorization: Bearer $TOKEN" \
    "https://civitai.com/api/v1/model-versions/$VID" 2>/dev/null)
  if [ -z "$META" ]; then
    echo "  ! cannot read metadata (bad id, or token lacks access)"
    continue
  fi

  BASE=$(printf '%s' "$META" | python3 -c 'import sys,json;print(json.load(sys.stdin).get("baseModel","?"))' 2>/dev/null)
  NAME=$(printf '%s' "$META" | python3 -c 'import sys,json;d=json.load(sys.stdin);print(d.get("model",{}).get("name","?"),"/",d.get("name","?"))' 2>/dev/null)
  FILE=$(printf '%s' "$META" | python3 -c '
import sys, json
d = json.load(sys.stdin)
fs = [f for f in d.get("files", []) if f.get("type") == "Model"] or d.get("files", [])
print(fs[0]["name"] if fs else "")' 2>/dev/null)

  echo "  $NAME"
  echo "  base model: $BASE"
  [ -n "$FILE" ] && echo "  file: $FILE"

  case "$BASE" in
    *9B*|*9b*) ;;
    *Klein*|*klein*)
      echo "  !! says Klein but not 9B - a 4B adapter will not load on your 9B." ;;
    *)
      echo "  !! base model is not Klein 9B; Forge will refuse or produce noise." ;;
  esac

  OUT="$LORA_DIR/${FILE:-civitai-$VID.safetensors}"
  if [ -s "$OUT" ]; then
    echo "  already have $(basename "$OUT")"
    continue
  fi

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

Restart Forge so it rescans, or press the refresh arrow next to the Lora tab:
  tmux kill-session -t =forge
then re-run the setup script, which relaunches it.
EOF
