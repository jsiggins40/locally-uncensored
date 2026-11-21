#!/usr/bin/env bash
# Adds the community "uncensored" LoRAs for Krea 2 to the edit page: they show up
# as buttons under "Krea 2 LoRAs" in ✍️ Create when Krea 2 Turbo is the model.
# Needs Krea 2 Turbo first (add-krea2.sh). About 2.7 GB. Each file is pinned to the
# exact version that was looked at (all are ai-toolkit Krea 2 LoRAs: only weights
# for Krea 2's own layers) and checked against its SHA-256, then tested on the GPU;
# one that fails is moved aside so the page never offers it.
# General-purpose ones only: no LoRAs of real people, none for a single act.
# Most have no licence text; Krea 2 itself is under the Krea 2 Community License.
#   bash <(curl -fsSL https://raw.githubusercontent.com/jsiggins40/locally-uncensored/rv6-mobile-editor/tools/rv6-mobile/add-krea2-loras.sh)
set -euo pipefail
cd "$HOME/ComfyUI"
SRC="https://raw.githubusercontent.com/jsiggins40/locally-uncensored/rv6-mobile-editor/tools/rv6-mobile"

restart() {
  if command -v edit-restart >/dev/null; then edit-restart; else
    sudo curl -fsSL "$SRC/edit-restart.sh" -o /usr/local/bin/edit-restart && sudo chmod +x /usr/local/bin/edit-restart && edit-restart
  fi
}

ls models/diffusion_models/*[Kk]rea*[Tt]urbo* >/dev/null 2>&1 \
  || { echo "Krea 2 Turbo isn't installed yet. Run add-krea2.sh first, then this."; exit 1; }

venv/bin/python - <<'PY'
import hashlib, json, os, shutil
from huggingface_hub import hf_hub_download
# (repo, commit, file, sha256, saved as, name on the page, words added to the prompt, note under the buttons)
LORAS = [
    ("nnndite/krea2nsfwv3", "bff29b1a0724c1dabd92dc18bb31a7c022abf1e6", "krea2_nsfw_v3.safetensors",
     "6ec564e04eed317ae24093b4907004ac59a7c528d19d025bb41b4d719c4d16ec", "krea2_nsfw_v3.safetensors",
     "🔞 NSFW v3", [], "general explicit content; its author suggests strength 0.8 to 1.2"),
    ("stuff123456/krea2SNOFS", "a8da409de53ae426443aab10ec356fba0c3c530f", "snofs_krea_v1_1.safetensors",
     "af6f66910ab1c0ec158b83cdd551a6ccfd14d0f2f79ceee7485f42fddac90a65", "snofs_krea_v1_1.safetensors",
     "🔞 SNOFS", [], "explicit poses and acts; name them plainly in your prompt"),
    ("uzumix/MysticXXX_KREA2_v2.safetensors", "3afee19e65ab1198081967cffab40b1fd42670ec", "MysticXXX_KREA2_v2.safetensors",
     "504378343980b494d8774b7f13461f8543d9c05a6f3ebc10d0356a4b361d26d4", "mysticxxx_krea2_v2.safetensors",
     "🔞 MysticXXX", [], "explicit content (its author gives no description)"),
    ("f23gg/Krea-2-NSFW-V5", "45263ccb4d3072f48746811edbf7a94faa3ca839", "Verite_realism.safetensors",
     "e947ebc07ee627796196b6dc551326c387ca81af173a5df72a789f135cd21f0b", "verite_nsfw_v5.safetensors",
     "📱 Amateur photo NSFW", ["Candid amateur photo"], "explicit content with a phone-photo look"),
    ("uzumix/krea2filterbypass3.safetensors", "66f26e38045ad7f34fd93e87ca6e7206552c7080", "krea2filterbypass3.safetensors",
     "ec5901a2d0b8f4e4e1e7e62fe4567566f0837799f7a413b03a06f72f47934dda", "filter_bypass_3.safetensors",
     "🔓 Filter bypass", [], "turns down the last layers of Krea 2's prompt reader, where prompts get toned down; combine it with another one"),
]
dest_dir = "models/loras/krea2"
os.makedirs(dest_dir, exist_ok=True)
meta_path = "custom_nodes/rv6_mobile/web/lora_info.json"
try: meta = json.load(open(meta_path))
except Exception: meta = {}
for repo, rev, fname, sha, save, name, words, note in LORAS:
    dest = os.path.join(dest_dir, save)
    print(f"== {name}  ({repo})")
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        print("   already have", dest)
    else:
        try:
            tmp = hf_hub_download(repo, fname, revision=rev, local_dir="models/_hf_tmp")
        except Exception as e:
            print(f"   download failed ({type(e).__name__}: {str(e)[:150]}); skipped"); continue
        h = hashlib.sha256()
        with open(tmp, "rb") as f:
            for b in iter(lambda: f.read(1 << 22), b""): h.update(b)
        if h.hexdigest() != sha:
            os.remove(tmp); print("   doesn't match the checked file; not installed"); continue
        shutil.move(tmp, dest)
        print(f"   installed {dest} ({os.path.getsize(dest) / 1e6:.0f} MB)")
    meta["krea2/" + save] = {"name": name, "baseModel": "Krea 2", "trainedWords": words, "note": note}
shutil.rmtree("models/_hf_tmp", ignore_errors=True)
os.makedirs(os.path.dirname(meta_path), exist_ok=True)
json.dump(meta, open(meta_path, "w"), indent=1)
PY

echo "== Tests on the GPU (the first one loads Krea 2: a minute or two)"
curl -sf -o /dev/null localhost:8188 || restart
curl -fsS -o /dev/null -X POST localhost:8188/api/refresh 2>/dev/null || true
venv/bin/python - <<'PY'
import json, os, re, shutil, time, urllib.request, urllib.error
from PIL import Image
def get(path): return json.load(urllib.request.urlopen("http://127.0.0.1:8188" + path))
def opts(n, inp):
    v = get("/api/object_info/" + n).get(n, {}).get("input", {})
    v = v.get("required", {}).get(inp) or v.get("optional", {}).get(inp)
    return (v[0] if isinstance(v[0], list) else v[1].get("options", [])) if v else []
bf = lambda xs: sorted(xs, key=lambda n: ("bf16" not in n.lower(), n))
unet = (bf([n for n in opts("UNETLoader", "unet_name") if "krea" in n.lower() and "turbo" in n.lower()]) or [None])[0]
clip = (bf([n for n in opts("CLIPLoader", "clip_name") if re.search(r"krea|qwen3.?vl.?4b", n, re.I)]) or [None])[0]
vaes = opts("VAELoader", "vae_name")
vae = next((n for n in vaes if "krea" in n.lower()), None) or next((n for n in vaes if "qwen_image_vae" in n.lower()), None)
loras = [n for n in opts("LoraLoaderModelOnly", "lora_name") if re.match(r"krea2[\\/]", n, re.I)]
print("ComfyUI sees:", unet, "|", len(loras), "Krea 2 LoRAs")
if not (unet and clip and vae): raise SystemExit("Krea 2 itself isn't visible to ComfyUI; run add-krea2.sh")

def run(lora):
    p = {"1": {"class_type": "UNETLoader", "inputs": {"unet_name": unet, "weight_dtype": "default"}},
         "18": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["1", 0], "lora_name": lora, "strength_model": 1}},
         "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": clip, "type": "krea2"}},
         "13": {"class_type": "VAELoader", "inputs": {"vae_name": vae}},
         "3": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": "a red apple on a wooden table, photo"}},
         "4": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["3", 0]}},
         "6": {"class_type": "EmptyLatentImage", "inputs": {"width": 768, "height": 768, "batch_size": 1}},
         "7": {"class_type": "KSampler", "inputs": {"model": ["18", 0], "positive": ["3", 0], "negative": ["4", 0], "latent_image": ["6", 0],
               "seed": 1, "steps": 8, "cfg": 1, "sampler_name": "euler", "scheduler": "simple", "denoise": 1}},
         "8": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["13", 0]}},
         "9": {"class_type": "SaveImage", "inputs": {"images": ["8", 0], "filename_prefix": "rv6m_krea_lora_test"}}}
    req = urllib.request.Request("http://127.0.0.1:8188/api/prompt", json.dumps({"prompt": p}).encode(), {"Content-Type": "application/json"})
    try:
        pid = json.load(urllib.request.urlopen(req))["prompt_id"]
    except urllib.error.HTTPError as e:
        return "rejected: " + e.read().decode()[:300]
    t = time.time()
    while time.time() - t < 600:
        h = get("/api/history/" + pid)
        if pid in h:
            st = h[pid]["status"]
            if st["status_str"] != "success": return "error: " + json.dumps(st)[:300]
            imgs = [i for o in h[pid]["outputs"].values() for i in o.get("images", [])]
            if not imgs: return "no picture"
            im = Image.open(os.path.join("output", imgs[0].get("subfolder", ""), imgs[0]["filename"])).convert("RGB").resize((32, 32))
            px = list(im.getdata())
            r, g, b = [sum(c) / len(c) for c in zip(*px)]
            spread = max(max(c) - min(c) for c in zip(*px))
            if max(r, g, b) < 8 or min(r, g, b) > 247 or spread < 12: return "the picture came out blank"
            return f"ok in {time.time() - t:.0f}s"
        time.sleep(2)
    return "timed out"

good = 0
for lora in loras:
    res = run(lora)
    print(f"TEST {os.path.basename(lora)}: {res}")
    if res.startswith("ok"): good += 1
    else:
        os.makedirs("models/_disabled", exist_ok=True)
        shutil.move(os.path.join("models/loras", lora), os.path.join("models/_disabled", os.path.basename(lora)))
        print("   moved to models/_disabled so the page won't offer it")
print(f"{good} of {len(loras)} LoRAs work.")
PY

echo "== Updating the page"
restart
echo
echo "Done. Reload the edit page: ✍️ Create → Krea 2 Turbo → Krea 2 LoRAs (tap to switch on)."
