#!/usr/bin/env bash
# Finds checkpoints (or, with --lora, LoRAs) on Civitai by name, lets you
# confirm each one, and downloads it into ComfyUI. Re-run any time to add more.
#   bash <(curl -fsSL https://raw.githubusercontent.com/jsiggins40/locally-uncensored/rv6-mobile-editor/tools/rv6-mobile/add-models.sh)
#   bash <(curl -fsSL .../add-models.sh) --lora "qwen edit nsfw"
set -euo pipefail

COMFY="$HOME/ComfyUI"
TOKEN_FILE="$HOME/.civitai_token"

if [ ! -s "$TOKEN_FILE" ]; then
  echo "Civitai needs an API key to download these models (free):"
  echo "  civitai.com -> your avatar -> Settings -> API Keys -> Add API key"
  read -rp "Paste the key here and press Enter: " key </dev/tty
  [ -n "$key" ] || { echo "No key given."; exit 1; }
  printf '%s' "$key" > "$TOKEN_FILE"; chmod 600 "$TOKEN_FILE"
fi

cd "$COMFY"
CIVITAI_TOKEN="$(cat "$TOKEN_FILE")" venv/bin/python - "$@" <<'PY'
import json, os, subprocess, sys, urllib.parse, urllib.request, urllib.error

token = os.environ["CIVITAI_TOKEN"]
args = sys.argv[1:]
lora = "--lora" in args
args = [a for a in args if a != "--lora"]
kind = "LORA" if lora else "Checkpoint"
dest = os.path.join("models", "loras" if lora else "checkpoints")
queries = args or (["qwen edit nsfw"] if lora else ["Anteros XXXL", "bigASP"])
# The edit page reads trigger words and base model for LoRAs from here
meta_path = os.path.join("custom_nodes", "rv6_mobile", "web", "lora_info.json")
tty = open("/dev/tty")

def ask(prompt):
    print(prompt, end="", flush=True)
    return tty.readline().strip()

def api(path):
    req = urllib.request.Request("https://civitai.com/api/v1/" + path,
                                 headers={"Authorization": "Bearer " + token, "User-Agent": "rv6-mobile"})
    return json.load(urllib.request.urlopen(req, timeout=30))

for q in queries:
    print(f"\n== Searching Civitai for: {q}")
    try:
        items = api("models?" + urllib.parse.urlencode({"query": q, "types": kind, "nsfw": "true", "limit": 10}))["items"]
        if lora:  # Qwen LoRAs first: the Describe-edit mode can only use those
            items.sort(key=lambda m: "qwen" not in str((m.get("modelVersions") or [{}])[0].get("baseModel", "")).lower())
    except urllib.error.HTTPError as e:
        print("  Search failed:", e.code, e.read().decode()[:200]); continue
    choices = []
    for m in items:
        if not m.get("modelVersions"): continue
        v = m["modelVersions"][0]
        files = [f for f in v.get("files", []) if f.get("name", "").endswith(".safetensors")]
        f = next((f for f in files if f.get("primary")), files[0] if files else None)
        if not f: continue
        choices.append((m, v, f))
        size = f.get('sizeKB', 0) / 1024
        print(f"  {len(choices)}) {m['name']}  [{v.get('name')}, {v.get('baseModel')}, "
              + (f"{size:.0f} MB" if size < 1024 else f"{size / 1024:.1f} GB") + f"] by {m.get('creator', {}).get('username', '?')}")
    if not choices:
        print("  Nothing found."); continue
    pick = ask("  Download which? Number, or Enter to skip: ")
    if not pick.isdigit() or not 1 <= int(pick) <= len(choices):
        print("  Skipped."); continue
    m, v, f = choices[int(pick) - 1]
    out = os.path.join(dest, f["name"])
    if os.path.exists(out) and os.path.getsize(out) > 0.95 * f.get("sizeKB", 0) * 1024:
        print("  Already downloaded:", f["name"]); continue
    url = f"https://civitai.com/api/download/models/{v['id']}?token={token}"
    print("  Downloading", f["name"], "...")
    r = subprocess.run(["curl", "-fL", "-C", "-", "--progress-bar", "-o", out, url])
    if r.returncode:
        print("  Download failed. If it says 401/403, the API key is wrong: rm ~/.civitai_token and run again.")
    else:
        print("  Saved:", out)
    if lora:
        words = [w.strip() for w in v.get("trainedWords") or [] if w.strip()]
        print("  Trigger words:", ", ".join(words) or "(none)")
        try: meta = json.load(open(meta_path))
        except Exception: meta = {}
        meta[f["name"]] = {"name": m["name"], "baseModel": v.get("baseModel", ""), "trainedWords": words}
        json.dump(meta, open(meta_path, "w"), indent=1)

print("\nDone. Reload the edit page; " + ("new LoRAs appear under Extra LoRAs in Describe edit." if lora else "new models appear in its Model list."))
PY
