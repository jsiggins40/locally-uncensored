#!/usr/bin/env bash
# Searches Civitai (civitai.red, its adult-content site; civitai.com has the same
# catalogue) for checkpoints or, with --lora, LoRAs by name, lists what it finds
# with the model each was made for, and downloads the ones you pick into ComfyUI.
# Searching needs nothing; downloading asks once for a free Civitai API key.
# Re-run any time to add more.
#   bash <(curl -fsSL https://raw.githubusercontent.com/jsiggins40/locally-uncensored/rv6-mobile-editor/tools/rv6-mobile/add-models.sh) --lora "krea 2 nsfw"
#   ... add-models.sh) --lora "minimax h3 image" "qwen 2.1 nsfw"     several searches in a row
#   ... add-models.sh) --lora --list "krea 2 nsfw"                   only list, download nothing
#   ... add-models.sh)                                               checkpoints (Anteros XXXL, bigASP)
# LoRAs of real people are left out of the lists.
set -euo pipefail
cd "$HOME/ComfyUI"
venv/bin/python - "$@" <<'PY'
import json, os, re, subprocess, sys, urllib.parse, urllib.request, urllib.error

HOSTS = [os.environ.get("CIVITAI_HOST", "civitai.red"), "civitai.com"]
TOKEN_FILE = os.path.expanduser("~/.civitai_token")
args = sys.argv[1:]
lora, only_list = "--lora" in args, "--list" in args
args = [a for a in args if a not in ("--lora", "--list")]
kind = "LORA" if lora else "Checkpoint"
dest = os.path.join("models", "loras" if lora else "checkpoints")
queries = args or (["qwen edit nsfw"] if lora else ["Anteros XXXL", "bigASP"])
# The edit page reads trigger words and base model for LoRAs from here
meta_path = os.path.join("custom_nodes", "rv6_mobile", "web", "lora_info.json")
try: tty = open("/dev/tty")
except OSError: tty = None

def ask(prompt):
    print(prompt, end="", flush=True)
    return tty.readline().strip() if tty else ""

def token():
    if os.path.exists(TOKEN_FILE) and os.path.getsize(TOKEN_FILE):
        return open(TOKEN_FILE).read().strip()
    print("  Civitai needs an API key to download (free):")
    print("    civitai.com -> your avatar -> Settings -> API Keys -> Add API key")
    key = re.sub(r"\s", "", ask("  Paste the key here and press Enter: "))
    if not key: return None
    with open(TOKEN_FILE, "w") as f: f.write(key)
    os.chmod(TOKEN_FILE, 0o600)
    return key

def search(q):
    err = None
    for host in HOSTS:
        url = f"https://{host}/api/v1/models?" + urllib.parse.urlencode({"query": q, "types": kind, "nsfw": "true", "limit": 20})
        try:
            return host, json.load(urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "rv6-mobile"}), timeout=30)).get("items", [])
        except Exception as e:
            err = e
    raise err

for q in queries:
    try:
        host, items = search(q)
    except Exception as e:
        print(f"\n== Searching for: {q}\n  Search failed: {type(e).__name__}: {str(e)[:200]}"); continue
    print(f"\n== {host}: {q}")
    choices, people = [], 0
    for m in items:
        if m.get("poi") or m.get("minor"):      # real people, or marked as depicting minors: not offered
            people += 1; continue
        if not m.get("modelVersions"): continue
        v = m["modelVersions"][0]
        files = [f for f in v.get("files", []) if f.get("name", "").endswith(".safetensors")]
        f = next((f for f in files if f.get("primary")), files[0] if files else None)
        if not f: continue
        choices.append((m, v, f))
        size = f.get("sizeKB", 0) / 1024
        print(f"  {len(choices):>2}) {m['name'][:60]}\n        made for: {v.get('baseModel') or '?'} · "
              + (f"{size:.0f} MB" if size < 1024 else f"{size / 1024:.1f} GB")
              + f" · {m.get('stats', {}).get('downloadCount', 0):,} downloads · by {m.get('creator', {}).get('username', '?')}")
    if people: print(f"  ({people} left out: real people)")
    if not choices:
        print("  Nothing found."); continue
    if only_list: continue
    pick = ask("  Download which? Numbers with spaces (e.g. 1 3), or Enter to skip: ")
    picks = [int(n) for n in re.findall(r"\d+", pick) if 1 <= int(n) <= len(choices)]
    if not picks:
        print("  Skipped."); continue
    key = token()
    if not key:
        print("  No key given; nothing downloaded."); continue
    for n in picks:
        m, v, f = choices[n - 1]
        out = os.path.join(dest, f["name"])
        if os.path.exists(out) and os.path.getsize(out) > 0.95 * f.get("sizeKB", 0) * 1024:
            print("  Already downloaded:", f["name"]); continue
        print("  Downloading", f["name"], "...")
        ok = False
        for h in dict.fromkeys([host] + HOSTS):
            r = subprocess.run(["curl", "-fL", "-C", "-", "--progress-bar", "-o", out, f"https://{h}/api/download/models/{v['id']}?token={key}"])
            if r.returncode == 0: ok = True; break
        if not ok:
            if os.path.exists(out) and os.path.getsize(out) < 100000: os.remove(out)
            print("  Download failed. If it says 401 or 403, the API key is wrong: rm ~/.civitai_token and run again.")
            continue
        print("  Saved:", out)
        if lora:
            words = [w.strip() for w in v.get("trainedWords") or [] if w.strip()]
            print("  Trigger words:", ", ".join(words) or "(none)")
            try: meta = json.load(open(meta_path))
            except Exception: meta = {}
            meta[f["name"]] = {"name": m["name"], "baseModel": v.get("baseModel", ""), "trainedWords": words}
            os.makedirs(os.path.dirname(meta_path), exist_ok=True)
            json.dump(meta, open(meta_path, "w"), indent=1)

if not only_list:
    print("\nDone. Run edit-restart and reload the edit page: " + ("new LoRAs appear in the LoRA menu of the model they were made for, and in 🐝 Swarm mode."
          if lora else "new models appear in its Model list."))
PY
