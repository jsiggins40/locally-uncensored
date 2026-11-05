#!/usr/bin/env bash
# Adds Krea 2 Turbo (Krea, June 2026) to Create mode on the edit page: a 12B
# text-to-image model made for 8 steps at CFG 1 (a few seconds a picture), with
# its Qwen3-VL-4B text encoder and the Qwen-Image VAE. ComfyUI runs it with its
# own nodes (no custom nodes). Gets Comfy-Org's ComfyUI files, tests a picture
# on the GPU, then updates the page: "Krea 2 Turbo" appears under Create model.
# Krea 2 ships under the Krea 2 Community License (https://www.krea.ai/krea-2-licensing).
#   bash <(curl -fsSL https://raw.githubusercontent.com/jsiggins40/locally-uncensored/rv6-mobile-editor/tools/rv6-mobile/add-krea2.sh)
set -euo pipefail
cd "$HOME/ComfyUI"
SRC="https://raw.githubusercontent.com/jsiggins40/locally-uncensored/rv6-mobile-editor/tools/rv6-mobile"

restart() {
  if command -v edit-restart >/dev/null; then edit-restart; else
    sudo curl -fsSL "$SRC/edit-restart.sh" -o /usr/local/bin/edit-restart && sudo chmod +x /usr/local/bin/edit-restart && edit-restart
  fi
}

echo "== ComfyUI must know Krea 2 (added in ComfyUI 0.26)"
if ! grep -q "krea2" comfy/supported_models.py 2>/dev/null; then
  echo "   updating ComfyUI"
  git pull -q --ff-only && venv/bin/pip install -q -r requirements.txt
  restart
fi
grep -q "krea2" comfy/supported_models.py || { echo "This ComfyUI can't run Krea 2 even after updating; stopping."; exit 1; }

venv/bin/python - <<'PY'
import os, shutil, sys
from huggingface_hub import HfApi, hf_hub_download
free = shutil.disk_usage(os.path.realpath("models")).free / 1e9
print(f"== Free disk: {free:.0f} GB")
if free < 45:
    sys.exit("Need about 45 GB free. Delete something first.")
R = "Comfy-Org/Krea-2"
try:
    fl = [(s.rfilename, s.size or 0) for s in HfApi().model_info(R, files_metadata=True).siblings]
except Exception as e:
    sys.exit(f"Can't list {R}: {type(e).__name__}: {str(e)[:200]}")
st = [(f, sz) for f, sz in fl if f.endswith(".safetensors")]
for f, sz in st: print(f"     {f} ({sz / 1e9:.1f} GB)")
# bf16 first (the H200 has the memory), then fp8; int8/fp4 last
rank = lambda f: ("bf16" not in f.lower(), "fp8" not in f.lower(), any(w in f.lower() for w in ("int8", "fp4", "nvfp4")), f)
unet = sorted([f for f, _ in st if "diffusion_models/" in f and "turbo" in f.lower()], key=rank)
te = sorted([f for f, _ in st if "text_encoders/" in f], key=rank)
vae = sorted([f for f, _ in st if "vae/" in f], key=rank)
if not unet or not te:
    sys.exit(f"{R} has no Turbo model or text encoder (files above). Nothing changed.")
def fetch(path, sub, must):
    name = os.path.basename(path)
    if not any(m in name.lower() for m in must):  # the page finds it by name
        name = "krea2_" + name
    dest = os.path.join("models", sub, name)
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        print("   already have", dest); return
    print(f"   downloading {path}")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    shutil.move(hf_hub_download(R, path, local_dir="models/_hf_tmp"), dest)
print("== Krea 2 Turbo")
fetch(unet[0], "diffusion_models", ["krea"])
print("== Text encoder (Qwen3-VL-4B)")
fetch(te[0], "text_encoders", ["krea", "qwen3vl_4b", "qwen3-vl-4b", "qwen3_vl_4b"])
if vae:
    print("== VAE (Qwen-Image's)")
    fetch(vae[0], "vae", ["krea", "qwen_image_vae"])
elif not any("qwen_image_vae" in f for f in os.listdir("models/vae")):
    sys.exit(f"{R} has no VAE and the Qwen-Image VAE isn't installed (add-edit-model.sh has it).")
shutil.rmtree("models/_hf_tmp", ignore_errors=True)
PY

echo "== Test on the GPU (first load takes a minute or two)"
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
print("ComfyUI sees:", unet, "|", clip, "|", vae, "| encoder type krea2:", "krea2" in opts("CLIPLoader", "type"))
if not (unet and clip and vae):
    raise SystemExit("KREA 2 TEST: files not visible to ComfyUI")
p = {"1": {"class_type": "UNETLoader", "inputs": {"unet_name": unet, "weight_dtype": "default"}},
     "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": clip, "type": "krea2"}},
     "13": {"class_type": "VAELoader", "inputs": {"vae_name": vae}},
     "3": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": "a red apple on a wooden table, photo"}},
     "4": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["3", 0]}},
     "6": {"class_type": "EmptyLatentImage", "inputs": {"width": 1024, "height": 1024, "batch_size": 1}},
     "7": {"class_type": "KSampler", "inputs": {"model": ["1", 0], "positive": ["3", 0], "negative": ["4", 0], "latent_image": ["6", 0],
           "seed": 1, "steps": 8, "cfg": 1, "sampler_name": "euler", "scheduler": "simple", "denoise": 1}},
     "8": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["13", 0]}},
     "9": {"class_type": "SaveImage", "inputs": {"images": ["8", 0], "filename_prefix": "rv6m_krea_test"}}}
ok = False
req = urllib.request.Request("http://127.0.0.1:8188/api/prompt", json.dumps({"prompt": p}).encode(), {"Content-Type": "application/json"})
try:
    pid = json.load(urllib.request.urlopen(req))["prompt_id"]
except urllib.error.HTTPError as e:
    print("KREA 2 TEST REJECTED:", e.read().decode()[:600]); pid = None
t = time.time()
while pid and time.time() - t < 900:
    h = get("/api/history/" + pid)
    if pid in h:
        s = h[pid]["status"]
        print("KREA 2 TEST:", s["status_str"], f"in {time.time() - t:.0f}s")
        if s["status_str"] != "success":
            print("  ", json.dumps(s)[:600]); break
        imgs = [i for o in h[pid]["outputs"].values() for i in o.get("images", [])]
        if imgs:
            im = Image.open(os.path.join("output", imgs[0].get("subfolder", ""), imgs[0]["filename"])).convert("RGB")
            r, g, b = [sum(c) / len(c) for c in zip(*im.resize((32, 32)).getdata())]
            print(f"   average colour R{r:.0f} G{g:.0f} B{b:.0f}")
            ok = not (max(r, g, b) < 8 or min(r, g, b) > 247)
            if not ok: print("   the picture came out blank")
        break
    time.sleep(2)
if not ok:
    os.makedirs("models/_disabled", exist_ok=True)
    shutil.move(os.path.join("models/diffusion_models", unet), os.path.join("models/_disabled", os.path.basename(unet)))
    raise SystemExit(f"Krea 2 failed its test; moved {unet} to models/_disabled so the page won't offer it.")
print("The test picture is in ComfyUI's output folder (rv6m_krea_test_*).")
PY

echo "== Updating the page"
restart
echo
echo "Done. Reload the edit page: Create mode → Create model → Krea 2 Turbo (8 steps)."
