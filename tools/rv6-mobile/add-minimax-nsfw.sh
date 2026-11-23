#!/usr/bin/env bash
# Uncensored extras for MiniMax H3 Image (add-minimax-image.sh first):
#   - "Heretic" text encoder: an abliterated copy of the Qwen3-VL 32B model that reads
#     your prompt, made to pass explicit wording on as written. 4-bit, 15.7 GB, the
#     same layout as the standard encoder. On the edit page it is the "🔓 Uncensored"
#     switch that appears in 🐝 Swarm mode when a MiniMax model is chosen.
#   - "Full nude" LoRA for MiniMax H3 Image (0.3 GB), in Swarm mode's LoRA list.
# Both are pinned to the exact files that were looked at and checked against their
# SHA-256, then tried on the GPU through SwarmUI. General-purpose ones only: no LoRAs
# of real people, none for a single act. The encoder is Apache-2.0; the LoRA has no
# licence text; MiniMax H3 Image itself has territory limits (see add-minimax-image.sh).
#   bash <(curl -fsSL https://raw.githubusercontent.com/jsiggins40/locally-uncensored/rv6-mobile-editor/tools/rv6-mobile/add-minimax-nsfw.sh)
set -euo pipefail
cd "$HOME/ComfyUI"
ls models/diffusion_models/*minimax_h3_image* >/dev/null 2>&1 \
  || { echo "MiniMax H3 Image isn't installed yet. Run add-minimax-image.sh first, then this."; exit 1; }
[ -x "$HOME/.local/bin/swarm-start" ] || { echo "This needs SwarmUI: run add-swarmui.sh first."; exit 1; }

venv/bin/python - <<'PY'
import hashlib, json, os, shutil, sys
from huggingface_hub import hf_hub_download
free = shutil.disk_usage(os.path.realpath("models")).free / 1e9
print(f"== Free disk: {free:.0f} GB")
# (repo, commit, file, SHA-256, saved as, what it is)
FILES = [
    ("Momoking/Qwen3-VL-32B-Heretic-MiniMax-H3-NVFP4", "0a9486d17cfb20c1a7abc97118c4d95910b9b1f4", "qwen3vl_32b_heretic_minimax_h3_nvfp4.safetensors",
     "a166c7bbbe66a22065159e478335fee4a633c4a3e3bb34c8e8ac4cc91bf4996f", "models/text_encoders/qwen3vl_32b_heretic_minimax_h3_nvfp4.safetensors",
     "Uncensored text encoder (15.7 GB)"),
    ("Tadness/FullNude", "b58435b73227b713885721fd233b80f548b512f6", "FullNude.safetensors",
     "786a4a8e80c79c9ad6cf1a53959cb415e8ed62f74a9ada6ca6df31c4292e5506", "models/loras/minimax_h3_image/fullnude.safetensors",
     "Full nude LoRA (0.3 GB)"),
]
need = [f for f in FILES if not (os.path.exists(f[4]) and os.path.getsize(f[4]) > 0)]
if need and free < 20:
    sys.exit("Need about 20 GB free. Delete something first (disk-report.sh shows what's there).")
for repo, rev, fname, sha, dest, what in FILES:
    print("==", what, f"({repo})")
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        print("   already have", dest); continue
    try:
        tmp = hf_hub_download(repo, fname, revision=rev, local_dir="models/_hf_tmp")
    except Exception as e:
        print(f"   download failed ({type(e).__name__}: {str(e)[:150]}); skipped"); continue
    h = hashlib.sha256()
    with open(tmp, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""): h.update(b)
    if h.hexdigest() != sha:
        os.remove(tmp); print("   doesn't match the checked file; not installed"); continue
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    shutil.move(tmp, dest)
    print("   installed", dest)
shutil.rmtree("models/_hf_tmp", ignore_errors=True)
meta_path = "custom_nodes/rv6_mobile/web/lora_info.json"
try: meta = json.load(open(meta_path))
except Exception: meta = {}
if os.path.exists("models/loras/minimax_h3_image/fullnude.safetensors"):
    meta["minimax_h3_image/fullnude.safetensors"] = {"name": "🔞 Full nude", "baseModel": "MiniMax H3 Image", "trainedWords": [], "note": "nudity (its author gives no description)"}
    os.makedirs(os.path.dirname(meta_path), exist_ok=True)
    json.dump(meta, open(meta_path, "w"), indent=1)
PY

echo "== Tests on the GPU, through SwarmUI (loading the encoder takes a few minutes)"
curl -sf -o /dev/null -m 5 localhost:7801 || "$HOME/.local/bin/swarm-start"
python3 - <<'PY'
import json, os, shutil, time, urllib.request
B = "http://127.0.0.1:7801"
def api(name, body, timeout=3600):
    req = urllib.request.Request(f"{B}/API/{name}", json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))
sid = api("GetNewSession", {}, 60)["session_id"]
try: api("TriggerRefresh", {"session_id": sid}, 600)      # make SwarmUI see the new files
except Exception as e: print("   (couldn't refresh SwarmUI's lists:", type(e).__name__ + "; restart it with swarm-start if the tests can't find the files)")
def names(subtype):
    return [f.get("name") or "" for f in api("ListModels", {"session_id": sid, "path": "", "depth": 8, "subtype": subtype, "allowRemote": False}, 120).get("files") or []]
model = next((n for n in names("Stable-Diffusion") if "minimax_h3_image" in n.lower()), None)
enc = next((n for n in names("Clip") if "heretic" in n.lower() and "minimax" in n.lower()), None)
lora = next((n for n in names("LoRA") if "fullnude" in n.lower()), None)
print("SwarmUI sees: model", model, "| encoder", enc, "| LoRA", lora)
if not model: raise SystemExit("SwarmUI doesn't list MiniMax H3 Image; run add-minimax-image.sh again.")
clean = lambda n: n[:-len(".safetensors")] if n.endswith(".safetensors") else n
def gen(label, extra):
    t = time.time()
    try:
        r = api("GenerateText2Image", {"session_id": sid, "images": 1, "model": clean(model), "width": 1024, "height": 1024, "seed": 1,
                                       "prompt": "a red apple on a wooden table, photo", "presets": ["Auto MiniMax H3 Image"], "donotsave": True, **extra})
    except Exception as e:
        print(f"{label}: failed ({type(e).__name__}: {str(e)[:200]})"); return False
    if r.get("error") or not r.get("images"):
        print(f"{label}: SwarmUI said: {str(r.get('error') or r)[:400]}"); return False
    print(f"{label}: ok in {time.time() - t:.0f}s"); return True
def aside(path, why):
    os.makedirs("models/_disabled", exist_ok=True)
    shutil.move(path, os.path.join("models/_disabled", os.path.basename(path)))
    print(f"   moved {os.path.basename(path)} to models/_disabled ({why}), so the page won't offer it")
# First the model as it is: if that fails the problem is elsewhere, and nothing is moved aside
if not gen("MINIMAX H3 IMAGE, AS IT IS", {}):
    raise SystemExit("The model itself didn't make a picture through SwarmUI, so the extras weren't tested. They stay installed. Send the lines above.")
if enc and not gen("UNCENSORED ENCODER TEST", {"qwenmodel": enc}):
    aside("models/text_encoders/" + enc, "it didn't work")
if lora and not gen("FULL NUDE LORA TEST (on a plain prompt)", {"loras": [clean(lora)], "loraweights": ["1"]}):
    aside("models/loras/" + lora, "it didn't work")
PY

echo
echo "Done. Run edit-restart, reload the edit page, then: 🐝 Swarm → Model → minimax_h3_image…"
echo "  Content: 🔓 Uncensored (the text encoder) · LoRAs: fullnude"
