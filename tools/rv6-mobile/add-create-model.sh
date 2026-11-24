#!/usr/bin/env bash
# Adds Qwen-Image-2512 (Qwen's newest text-to-image model, Dec 2025) for the
# page's "✍️ Create" mode: makes new images from a description, no photo
# needed. Uses the same text encoder and VAE as Qwen-Image-Edit (already
# installed). Downloads about 41 GB (bf16) plus its Lightning LoRA, then runs
# one test image on the GPU.
#   bash <(curl -fsSL https://raw.githubusercontent.com/jsiggins40/locally-uncensored/rv6-mobile-editor/tools/rv6-mobile/add-create-model.sh)
set -euo pipefail
cd "$HOME/ComfyUI"

venv/bin/python - <<'PY'
import os, re, shutil, sys
from huggingface_hub import HfApi, hf_hub_download

api = HfApi()
free_gb = shutil.disk_usage(os.path.realpath("models")).free / 1e9  # models may live on another disk (symlink)
print(f"== Free disk: {free_gb:.0f} GB")
want = "bf16" if free_gb >= 60 else "fp8"
if free_gb < 30:
    sys.exit("Need about 30-45 GB free for the Create model. Delete something first.")

def version(name):
    m = re.search(r"(\d{4})", os.path.basename(name)); return int(m.group(1)) if m else 0

def fetch(repo, path, subdir):
    dest = os.path.join("models", subdir, os.path.basename(path))
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        print("   already have", dest); return os.path.basename(path)
    print(f"   downloading {os.path.basename(path)} from {repo}")
    tmp = hf_hub_download(repo, path, local_dir="models/_hf_tmp")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    shutil.move(tmp, dest)
    return os.path.basename(path)

print("== Qwen-Image text-to-image model (newest version)")
base = api.list_repo_files("Comfy-Org/Qwen-Image_ComfyUI")
models = [f for f in base if "diffusion_models" in f and f.endswith(".safetensors")
          and os.path.basename(f).lower().startswith("qwen_image_") and "edit" not in f.lower()
          and "distill" not in f.lower() and version(f) > 0]
if not models:
    sys.exit("No versioned Qwen-Image model found in Comfy-Org/Qwen-Image_ComfyUI:\n" + "\n".join(base))
pick = max(models, key=lambda f: (version(f), want in f.lower()))
name = fetch("Comfy-Org/Qwen-Image_ComfyUI", pick, "diffusion_models")
v = version(name)

print("== Text encoder and VAE (shared with the edit model)")
te = [f for f in base if "text_encoders" in f and "qwen_2.5_vl" in f and f.endswith(".safetensors")]
vae = [f for f in base if f.endswith("qwen_image_vae.safetensors")]
have = lambda sub, pat: any(pat in n for n in os.listdir(os.path.join("models", sub))) if os.path.isdir(os.path.join("models", sub)) else False
if not have("text_encoders", "qwen_2.5_vl") and te: fetch("Comfy-Org/Qwen-Image_ComfyUI", max(te, key=lambda f: "fp8" in f), "text_encoders")
if not have("vae", "qwen_image_vae") and vae: fetch("Comfy-Org/Qwen-Image_ComfyUI", vae[0], "vae")

print("== Fast Lightning LoRA (optional)")
found = None
for repo in (f"lightx2v/Qwen-Image-{v}-Lightning", "lightx2v/Qwen-Image-Lightning"):
    try:
        lfiles = api.list_repo_files(repo)
    except Exception as e:
        print(f"   {repo}: not available ({type(e).__name__})"); continue
    loras = [f for f in lfiles if f.endswith(".safetensors") and "lightning" in f.lower() and "edit" not in f.lower()
             and re.search(r"[48]step", f.lower()) and version(f) == v
             and "e4m3fn" not in f.lower() and not os.path.basename(f).lower().startswith("qwen_image")]
    if loras:
        found = (repo, max(loras, key=lambda f: ("8step" in f.lower(), (re.findall(r"V(\d+(?:\.\d+)?)", f) or ["0"])[-1],
                                                 "bf16" in f.lower())))
        break
if found: fetch(*found, "loras")
else: print("   none matches this model version; Create will use 40 steps")
shutil.rmtree("models/_hf_tmp", ignore_errors=True)
PY

echo "== Test image on the GPU (first load takes a minute)"
curl -fsS -o /dev/null -X POST localhost:8188/api/refresh 2>/dev/null || true
venv/bin/python - <<'PY'
import json, re, time, urllib.request, urllib.error
def get(path):
    return json.load(urllib.request.urlopen("http://127.0.0.1:8188" + path))
def opts(node, inp):
    v = get("/api/object_info/" + node).get(node, {}).get("input", {})
    v = v.get("required", {}).get(inp) or v.get("optional", {}).get(inp)
    return (v[0] if isinstance(v[0], list) else v[1].get("options", [])) if v else []
ver = lambda n: int((re.findall(r"(\d{4})", n) or ["0"])[0])
unets = [n for n in opts("UNETLoader", "unet_name") if n.lower().startswith("qwen_image_") and "edit" not in n.lower() and ver(n)]
unet = max(unets, key=lambda n: (ver(n), "bf16" in n)) if unets else None
clip = (sorted(n for n in opts("CLIPLoader", "clip_name") if "qwen_2.5_vl" in n.lower()) or [None])[-1]
vae = (sorted(n for n in opts("VAELoader", "vae_name") if "qwen_image_vae" in n.lower()) or [None])[-1]
lights = [n for n in opts("LoraLoaderModelOnly", "lora_name") if "lightning" in n.lower() and "edit" not in n.lower() and unet and ver(n) == ver(unet)]
lora = (sorted(n for n in lights if "8step" in n.lower()) or sorted(lights) or [None])[-1]
print("ComfyUI sees:", unet, "|", clip, "|", vae, "| lora:", lora)
if not (unet and clip and vae):
    raise SystemExit("CREATE TEST: files not visible to ComfyUI")
steps = (4 if "4step" in lora.lower() and "8step" not in lora.lower() else 8) if lora else 40
p = {
    "1": {"class_type": "UNETLoader", "inputs": {"unet_name": unet, "weight_dtype": "default"}},
    "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": clip, "type": "qwen_image"}},
    "13": {"class_type": "VAELoader", "inputs": {"vae_name": vae}},
    "3": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": "a red apple on a wooden table, photo"}},
    "4": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": ""}},
    "6": {"class_type": "EmptySD3LatentImage", "inputs": {"width": 1024, "height": 1024, "batch_size": 1}},
    "12": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["1", 0], "shift": 3.1}},
    "7": {"class_type": "KSampler", "inputs": {"model": ["12", 0], "positive": ["3", 0], "negative": ["4", 0], "latent_image": ["6", 0],
          "seed": 1, "steps": steps, "cfg": 1 if lora else 4, "sampler_name": "euler", "scheduler": "simple", "denoise": 1}},
    "8": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["13", 0]}},
    "9": {"class_type": "SaveImage", "inputs": {"filename_prefix": "rv6_create_test", "images": ["8", 0]}},
}
if lora:
    p["11"] = {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["1", 0], "lora_name": lora, "strength_model": 1}}
    p["12"]["inputs"]["model"] = ["11", 0]
req = urllib.request.Request("http://127.0.0.1:8188/api/prompt", json.dumps({"prompt": p}).encode(), {"Content-Type": "application/json"})
try:
    pid = json.load(urllib.request.urlopen(req))["prompt_id"]
except urllib.error.HTTPError as e:
    raise SystemExit("CREATE TEST REJECTED: " + e.read().decode()[:800])
t = time.time()
while True:
    h = get("/api/history/" + pid)
    if pid in h:
        st = h[pid]["status"]
        print("CREATE TEST:", st["status_str"], f"in {time.time() - t:.0f}s")
        if st["status_str"] != "success": print(json.dumps(st)[:800])
        break
    if time.time() - t > 900: raise SystemExit("CREATE TEST TIMED OUT")
    time.sleep(2)
PY
echo
echo "Done. Reload the edit page: there is now a ✍️ Create mode."
