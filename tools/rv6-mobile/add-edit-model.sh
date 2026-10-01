#!/usr/bin/env bash
# Downloads Qwen-Image-Edit (instruction-based editing: "make her hair red")
# plus its text encoder, VAE and, when one matches, the 8-step Lightning LoRA,
# then runs one test edit on the GPU.
# QWEN_PRECISION=bf16 picks the bf16 weights (about 41 GB): faster on A100s,
# which have no native fp8, at twice the download. Default fp8 (about 20 GB).
#   bash <(curl -fsSL https://raw.githubusercontent.com/jsiggins40/locally-uncensored/rv6-mobile-editor/tools/rv6-mobile/add-edit-model.sh)
set -euo pipefail
cd "$HOME/ComfyUI"

venv/bin/python - <<'PY'
import os, re, shutil, sys
from huggingface_hub import HfApi, hf_hub_download

api = HfApi()
want = os.environ.get("QWEN_PRECISION", "fp8")
need_gb = 55 if want == "bf16" else 35
free_gb = shutil.disk_usage(".").free / 1e9
print(f"== Free disk: {free_gb:.0f} GB")
if free_gb < need_gb:
    sys.exit(f"Need about {need_gb} GB free for the edit model. Delete some checkpoints first.")

def version(name):
    m = re.search(r"(\d{4})", name)
    return int(m.group(1)) if m else 0

def precision(name):
    # fp8 halves the download and VRAM with little quality loss
    return 2 if "fp8" in name else 1 if "bf16" in name else 0

def edit_rank(name):
    return 2 if want in name else 1 if ("fp8" in name or "bf16" in name) else 0

def fetch(repo, path, subdir):
    dest = os.path.join("models", subdir, os.path.basename(path))
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        print("   already have", dest); return os.path.basename(path)
    print(f"   downloading {os.path.basename(path)} from {repo}")
    tmp = hf_hub_download(repo, path, local_dir="models/_hf_tmp")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    shutil.move(tmp, dest)
    return os.path.basename(path)

print("== Edit model")
files = api.list_repo_files("Comfy-Org/Qwen-Image-Edit_ComfyUI")
edits = [f for f in files if f.endswith(".safetensors") and "diffusion_models" in f and "qwen_image_edit" in f.lower()]
if not edits:
    sys.exit("No Qwen-Image-Edit files found in Comfy-Org/Qwen-Image-Edit_ComfyUI:\n" + "\n".join(files))
edit = max(edits, key=lambda f: (version(f), edit_rank(f)))
edit_name = fetch("Comfy-Org/Qwen-Image-Edit_ComfyUI", edit, "diffusion_models")
# Keep only the edit model just fetched so ComfyUI and the page use it
for old in os.listdir("models/diffusion_models"):
    if "qwen_image_edit" in old.lower() and old != edit_name:
        os.remove(os.path.join("models/diffusion_models", old)); print("   removed older", old)

print("== Text encoder and VAE")
base = api.list_repo_files("Comfy-Org/Qwen-Image_ComfyUI")
te = [f for f in base if "text_encoders" in f and "qwen_2.5_vl" in f and f.endswith(".safetensors")]
vae = [f for f in base if f.endswith("qwen_image_vae.safetensors")]
if not te or not vae:
    sys.exit("Text encoder / VAE not found in Comfy-Org/Qwen-Image_ComfyUI:\n" + "\n".join(base))
fetch("Comfy-Org/Qwen-Image_ComfyUI", max(te, key=precision), "text_encoders")
fetch("Comfy-Org/Qwen-Image_ComfyUI", vae[0], "vae")

print("== Fast 8-step LoRA (optional)")
try:
    lfiles = api.list_repo_files("lightx2v/Qwen-Image-Lightning")
    v = version(edit_name)
    loras = [f for f in lfiles if f.endswith(".safetensors") and "edit" in f.lower() and "8step" in f.lower()
             and (version(f) == v)]
    if loras:
        lora = max(loras, key=lambda f: (re.findall(r"V(\d+(?:\.\d+)?)", f) or ["0"])[-1] + str(precision(f)))
        fetch("lightx2v/Qwen-Image-Lightning", lora, "loras")
    else:
        print("   none matches this edit model version; edits will use 30 steps")
except Exception as e:
    print("   skipped:", e)
shutil.rmtree("models/_hf_tmp", ignore_errors=True)
PY

echo "== Test edit on the GPU (first load of a 20B model takes a minute or two)"
curl -fsS -o /dev/null -X POST localhost:8188/api/refresh 2>/dev/null || true
venv/bin/python - <<'PY'
import json, time, urllib.request, urllib.error
def get(path):
    return json.load(urllib.request.urlopen("http://127.0.0.1:8188" + path))
def opts(node, inp):
    v = get("/api/object_info/" + node).get(node, {}).get("input", {})
    v = v.get("required", {}).get(inp) or v.get("optional", {}).get(inp)
    return (v[0] if isinstance(v[0], list) else v[1].get("options", [])) if v else []
pick = lambda xs, f: sorted(x for x in xs if f(x.lower()))[-1:] or [None]
unet = pick(opts("UNETLoader", "unet_name"), lambda n: "qwen" in n and "edit" in n)[0]
clip = pick(opts("CLIPLoader", "clip_name"), lambda n: "qwen_2.5_vl" in n)[0]
vae = pick(opts("VAELoader", "vae_name"), lambda n: "qwen_image_vae" in n)[0]
lora = pick(opts("LoraLoaderModelOnly", "lora_name"), lambda n: "edit" in n and "lightning" in n)[0]
plus = bool(get("/api/object_info/TextEncodeQwenImageEditPlus"))
print("ComfyUI sees:", unet, "|", clip, "|", vae, "| lora:", lora, "| Plus encoder:", plus)
if not (unet and clip and vae):
    raise SystemExit("EDIT TEST: files not visible to ComfyUI")
enc = lambda t: ({"class_type": "TextEncodeQwenImageEditPlus", "inputs": {"clip": ["2", 0], "prompt": t, "vae": ["13", 0], "image1": ["5", 0]}}
                 if plus else
                 {"class_type": "TextEncodeQwenImageEdit", "inputs": {"clip": ["2", 0], "prompt": t, "vae": ["13", 0], "image": ["5", 0]}})
p = {
    "1": {"class_type": "UNETLoader", "inputs": {"unet_name": unet, "weight_dtype": "default"}},
    "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": clip, "type": "qwen_image"}},
    "13": {"class_type": "VAELoader", "inputs": {"vae_name": vae}},
    "5": {"class_type": "LoadImage", "inputs": {"image": "rv6m_test_img.png"}},
    "3": enc("make the whole picture bright red"), "4": enc(""),
    "6": {"class_type": "VAEEncode", "inputs": {"pixels": ["5", 0], "vae": ["13", 0]}},
    "12": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["1", 0], "shift": 3}},
    "7": {"class_type": "KSampler", "inputs": {"model": ["12", 0], "positive": ["3", 0], "negative": ["4", 0], "latent_image": ["6", 0],
          "seed": 1, "steps": 8 if lora else 30, "cfg": 1 if lora else 2.5, "sampler_name": "euler", "scheduler": "simple", "denoise": 1}},
    "8": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["13", 0]}},
    "9": {"class_type": "SaveImage", "inputs": {"filename_prefix": "rv6_edit_test", "images": ["8", 0]}},
}
if lora:
    p["11"] = {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["1", 0], "lora_name": lora, "strength_model": 1}}
    p["12"]["inputs"]["model"] = ["11", 0]
req = urllib.request.Request("http://127.0.0.1:8188/api/prompt", json.dumps({"prompt": p}).encode(), {"Content-Type": "application/json"})
try:
    pid = json.load(urllib.request.urlopen(req))["prompt_id"]
except urllib.error.HTTPError as e:
    raise SystemExit("EDIT TEST REJECTED: " + e.read().decode()[:800])
t = time.time()
while True:
    h = get("/api/history/" + pid)
    if pid in h:
        st = h[pid]["status"]
        print("EDIT TEST:", st["status_str"], f"in {time.time() - t:.0f}s")
        if st["status_str"] != "success": print(json.dumps(st)[:800])
        break
    if time.time() - t > 900: raise SystemExit("EDIT TEST TIMED OUT")
    time.sleep(2)
PY
echo
echo "Done. Reload the edit page and choose ✨ Describe edit."
