#!/usr/bin/env bash
# Gets models from Civitai (civitai.red, its adult-content site; civitai.com has the
# same catalogue) into ComfyUI. Two ways:
#   - give it a Civitai link (a model page or a download link, in quotes): it shows
#     what the link is and downloads it;
#   - give it words: it searches for checkpoints or, with --lora, LoRAs, lists what it
#     finds with the model each was made for, and downloads the ones you pick.
# Searching needs nothing; downloading asks once for a free Civitai API key. Each file
# is checked against the SHA-256 Civitai publishes for it.
# Wan 2.2 image-to-video LoRAs usually come as two files (high noise + low noise):
# both are fetched, into models/loras/wan22_i2v/, and show as one button under
# 🎬 Video → Video LoRAs on the edit page.
# Re-run any time to add more.
#   bash <(curl -fsSL https://raw.githubusercontent.com/jsiggins40/locally-uncensored/rv6-mobile-editor/tools/rv6-mobile/add-models.sh) "https://civitai.red/api/download/models/3061930?fileId=2940564"
#   ... add-models.sh) --lora "krea 2 nsfw"
#   ... add-models.sh) --lora "minimax h3 image" "qwen 2.1 nsfw"     several searches in a row
#   ... add-models.sh) --lora --list "krea 2 nsfw"                   only list, download nothing
#   ... add-models.sh)                                               checkpoints (Anteros XXXL, bigASP)
# LoRAs of real people are left out.
set -euo pipefail
cd "$HOME/ComfyUI"
venv/bin/python - "$@" <<'PY'
import glob, hashlib, json, os, re, subprocess, sys, urllib.parse, urllib.request, urllib.error

HOSTS = list(dict.fromkeys([os.environ.get("CIVITAI_HOST", "civitai.red"), "civitai.com"]))
base = lambda h: h if h.startswith("http") else "https://" + h
TOKEN_FILE = os.path.expanduser("~/.civitai_token")
args = sys.argv[1:]
lora, only_list = "--lora" in args, "--list" in args
args = [a for a in args if a not in ("--lora", "--list")]
is_link = lambda a: re.search(r"civitai\.[a-z]+/.*models/\d+", a) is not None
links, queries = [a for a in args if is_link(a)], [a for a in args if not is_link(a)]
if not args: queries = ["qwen edit nsfw"] if lora else ["Anteros XXXL", "bigASP"]
LORA_TYPES = ("LORA", "LoCon", "DoRA")
# The edit page reads trigger words and base model for LoRAs from here
meta_path = os.path.join("custom_nodes", "rv6_mobile", "web", "lora_info.json")
try: tty = open("/dev/tty")
except OSError: tty = None
got = {"lora": False, "video": False, "checkpoint": False}

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

def api(path, params=None):
    """Asks civitai.red, then civitai.com. Gives back (host that answered, the answer)."""
    err = None
    for host in HOSTS:
        url = f"{base(host)}/api/v1/{path}" + ("?" + urllib.parse.urlencode(params) if params else "")
        try:
            return host, json.load(urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "rv6-mobile"}), timeout=30))
        except Exception as e:
            err = e
    raise err

def pick_file(v, file_id=None):
    files = [f for f in v.get("files", []) if f.get("name", "").endswith(".safetensors")]
    return next((f for f in files if file_id and f.get("id") == file_id), None) \
        or next((f for f in files if f.get("primary")), files[0] if files else None)

size = lambda f: (lambda mb: f"{mb:.0f} MB" if mb < 1024 else f"{mb / 1024:.1f} GB")(f.get("sizeKB", 0) / 1024)
noise_of = lambda v, f: next((n for n in ("high", "low") if re.search(n, v.get("name", "") + " " + f.get("name", ""), re.I)), None)
wan_i2v = lambda v: re.search(r"wan.*2\.2.*i2v", v.get("baseModel") or "", re.I) is not None

def plan(m, v, file_id=None):
    """The files to fetch for one choice: the file itself and, for a Wan 2.2 LoRA that is the
    high-noise or low-noise half of a pair, the other half. Gives back [(version, file, noise)]."""
    f = pick_file(v, file_id)
    if not f: return []
    out = [(v, f, None)]
    if m.get("type") in LORA_TYPES and re.search(r"wan.*2\.2", v.get("baseModel") or "", re.I) and noise_of(v, f):
        mine = noise_of(v, f); other = "low" if mine == "high" else "high"
        strip = lambda s: re.sub(r"high|low", "", s, flags=re.I).lower()
        rest = [(o, pick_file(o)) for o in m.get("modelVersions", []) if o.get("id") != v.get("id") and o.get("baseModel") == v.get("baseModel")]
        rest = [(o, of) for o, of in rest if of and noise_of(o, of) == other]
        twin = [(o, of) for o, of in rest if strip(o.get("name", "")) == strip(v.get("name", ""))] or (rest if len(rest) == 1 else [])
        out = [(v, f, mine)] + [(o, of, other) for o, of in twin[:1]]
    return out

def install(m, v, f, noise, host, key):
    is_lora = m.get("type") in LORA_TYPES
    sub = "wan22_i2v" if is_lora and wan_i2v(v) else ""
    folder = os.path.join("models", "loras" if is_lora else "checkpoints", sub)
    os.makedirs(folder, exist_ok=True)
    out = os.path.join(folder, f["name"])
    want = ((f.get("hashes") or {}).get("SHA256") or "").lower()
    if os.path.exists(out) and os.path.getsize(out) > 0.95 * f.get("sizeKB", 0) * 1024:
        print("  Already downloaded:", f["name"])
    else:
        print("  Downloading", f["name"], "...")
        q = urllib.parse.urlencode({"fileId": f["id"], "token": key} if f.get("id") and not f.get("primary") else {"token": key})
        ok = False
        for h in dict.fromkeys([host] + HOSTS):
            r = subprocess.run(["curl", "-fL", "-C", "-", "--progress-bar", "-o", out, f"{base(h)}/api/download/models/{v['id']}?{q}"])
            if r.returncode == 0: ok = True; break
        if not ok:
            if os.path.exists(out) and os.path.getsize(out) < 100000: os.remove(out)
            print("  Download failed. If it says 401 or 403, the API key is wrong: rm ~/.civitai_token and run again.")
            return False
        if want:
            h = hashlib.sha256()
            with open(out, "rb") as fh:
                for b in iter(lambda: fh.read(1 << 22), b""): h.update(b)
            if h.hexdigest() != want:
                os.remove(out)
                print("  The file doesn't match the SHA-256 Civitai publishes for it; deleted. Run again to retry.")
                return False
        print("  Saved:", out + (" (checked against Civitai's SHA-256)" if want else ""))
    if not is_lora:
        got["checkpoint"] = True; return True
    got["video" if sub else "lora"] = True
    words = [w.strip() for w in v.get("trainedWords") or [] if w.strip()]
    print("  Trigger words:", ", ".join(words) or "(none)")
    try: meta = json.load(open(meta_path))
    except Exception: meta = {}
    entry = {"name": m["name"], "baseModel": v.get("baseModel", ""), "trainedWords": words}
    if sub and noise: entry.update(noise=noise, group=f"civitai-{m.get('id')}")   # the page shows the two halves as one button
    meta[os.path.join(sub, f["name"])] = entry
    os.makedirs(os.path.dirname(meta_path), exist_ok=True)
    json.dump(meta, open(meta_path, "w"), indent=1)
    return True

def line(n, m, v, f, extra=""):
    print(f"  {n}{m['name'][:60]}\n        made for: {v.get('baseModel') or '?'} · {size(f)}{extra}"
          f" · {m.get('stats', {}).get('downloadCount', 0):,} downloads · by {m.get('creator', {}).get('username', '?')}")

# 1. Links: a model page (civitai.red/models/123, with or without ?modelVersionId=) or a download link
for link in links:
    print(f"\n== {link}")
    vid = re.search(r"(?:api/download/models/|modelVersionId=)(\d+)", link)
    mid = re.search(r"civitai\.[a-z]+/models/(\d+)", link)
    fid = re.search(r"fileId=(\d+)", link)
    try:
        if vid and not mid:
            host, ver = api(f"model-versions/{vid.group(1)}"); model_id = ver["modelId"]
        else:
            model_id = mid.group(1)
        host, m = api(f"models/{model_id}")
    except Exception as e:
        print(f"  Couldn't look it up: {type(e).__name__}: {str(e)[:200]}"); continue
    if m.get("poi") or m.get("minor"):
        print("  Left out: Civitai marks it as depicting a real person or a minor."); continue
    if m.get("type") not in LORA_TYPES + ("Checkpoint",):
        print(f"  This is a {m.get('type')}; this installer only does LoRAs and checkpoints."); continue
    versions = m.get("modelVersions") or []
    v = next((x for x in versions if vid and x.get("id") == int(vid.group(1))), versions[0] if versions else None)
    todo = plan(m, v, int(fid.group(1)) if fid else None) if v else []
    if not todo:
        print("  No .safetensors file in it."); continue
    line("", m, v, todo[0][1], f" · {m.get('type')}")
    for o, of, noise in todo:
        print(f"        file: {of['name']} ({size(of)})" + (f" · {noise}-noise half" if noise else ""))
    if any(n for _, _, n in todo) and len(todo) == 1:
        print("        (its other half, for the other noise level, wasn't found: it will work at part strength)")
    if m.get("type") in LORA_TYPES and wan_i2v(v) and not glob.glob("models/diffusion_models/*wan2.2_i2v_high_noise*"):
        print("        NOTE: 🎬 Video (Wan 2.2) isn't installed on this VM yet. This LoRA needs it: add-video.sh (75 GB).")
    if only_list: continue
    key = token()
    if not key:
        print("  No key given; nothing downloaded."); continue
    for o, of, noise in todo: install(m, o, of, noise, host, key)

# 2. Searches
kind = "LORA" if lora else "Checkpoint"
for q in queries:
    try:
        host, items = api("models", {"query": q, "types": kind, "nsfw": "true", "limit": 20})
        items = items.get("items", [])
    except Exception as e:
        print(f"\n== Searching for: {q}\n  Search failed: {type(e).__name__}: {str(e)[:200]}"); continue
    print(f"\n== {host}: {q}")
    choices, people = [], 0
    for m in items:
        if m.get("poi") or m.get("minor"):      # real people, or marked as depicting minors: not offered
            people += 1; continue
        if not m.get("modelVersions"): continue
        v = m["modelVersions"][0]
        f = pick_file(v)
        if not f: continue
        choices.append((m, v))
        line(f"{len(choices):>2}) ", m, v, f)
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
        m, v = choices[n - 1]
        for o, of, noise in plan(m, v): install(m, o, of, noise, host, key)

if not only_list and any(got.values()):
    print("\nDone. Run edit-restart and reload the edit page:")
    if got["video"]: print("  Video LoRAs: 🎬 Video → Local (Wan 2.2) → Video LoRAs.")
    if got["lora"]: print("  LoRAs: in the LoRA menu of the model they were made for, and in 🐝 Swarm mode.")
    if got["checkpoint"]: print("  Checkpoints: in the page's Model list.")
PY
