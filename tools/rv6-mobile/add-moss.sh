#!/usr/bin/env bash
# Adds MOSS-Image (OpenMOSS, Oct 2026; Apache-2.0) to Create mode on the edit page:
# a 5B text-to-image model, strong with long detailed prompts and with text in
# pictures, with its Qwen3.5-9B text encoder and the FLUX.2 small VAE. ComfyUI
# runs it with its own nodes (no custom nodes). Also gets an uncensored LoRA for
# explicit photographic content (axiomofmind/Moss-image-nsfw-uncensored-lora),
# checked against its published SHA-256. Tests a picture on the GPU, then updates
# the page: "MOSS-Image" appears under Create model, with a 🔞 Uncensored switch.
# About 18 GB. Skip the LoRA with:  MOSS_NO_NSFW=1 bash <(curl ...)
#   bash <(curl -fsSL https://raw.githubusercontent.com/jsiggins40/locally-uncensored/rv6-mobile-editor/tools/rv6-mobile/add-moss.sh)
set -euo pipefail
cd "$HOME/ComfyUI"
SRC="https://raw.githubusercontent.com/jsiggins40/locally-uncensored/rv6-mobile-editor/tools/rv6-mobile"

restart() {
  if command -v edit-restart >/dev/null; then edit-restart; else
    sudo curl -fsSL "$SRC/edit-restart.sh" -o /usr/local/bin/edit-restart && sudo chmod +x /usr/local/bin/edit-restart && edit-restart
  fi
}

echo "== ComfyUI must know MOSS-Image (native support from late Oct 2026)"
if ! grep -rqi "moss_image" comfy/ 2>/dev/null; then
  echo "   updating ComfyUI"
  git pull -q --ff-only && venv/bin/pip install -q -r requirements.txt
  restart
fi
grep -rqi "moss_image" comfy/ || { echo "This ComfyUI can't run MOSS-Image even after updating; stopping."; exit 1; }

MOSS_NO_NSFW="${MOSS_NO_NSFW:-}" venv/bin/python - <<'PY'
import hashlib, os, shutil, sys
from huggingface_hub import hf_hub_download
free = shutil.disk_usage(os.path.realpath("models")).free / 1e9
print(f"== Free disk: {free:.0f} GB")
if free < 22:
    sys.exit("Need about 22 GB free. Delete something first.")
R = "Comfy-Org/MOSS-Image"
def fetch(repo, path, sub, name=None, sha=None):
    dest = os.path.join("models", sub, name or os.path.basename(path))
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        print("   already have", dest); return dest
    print(f"   downloading {repo}/{path}")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = hf_hub_download(repo, path, local_dir="models/_hf_tmp")
    if sha:
        h = hashlib.sha256()
        with open(tmp, "rb") as f:
            for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
        if h.hexdigest().lower() != sha.lower():
            os.remove(tmp)
            sys.exit(f"{path} doesn't match its published SHA-256; not installed.")
    shutil.move(tmp, dest)
    return dest
print("== MOSS-Image SFT (bf16, 10 GB)")
fetch(R, "diffusion_models/moss_image_sft_bf16.safetensors", "diffusion_models")
print("== Text encoder (Qwen3.5-9B, the version Comfy's own template uses)")
fetch(R, "text_encoders/qwen3.5_9b_w4a8_gptq.safetensors", "text_encoders")
print("== VAE (FLUX.2 small)")
fetch(R, "vae/flux2-vae_small.safetensors", "vae")
if not os.environ.get("MOSS_NO_NSFW"):
    print("== Uncensored LoRA (ComfyUI version; its author suggests strength 0.8, trigger moss_xvstyle)")
    fetch("axiomofmind/Moss-image-nsfw-uncensored-lora", "MOSS-Image-NSFW-Uncensored-LoRA-ComfyUI.safetensors", "loras",
          "moss_image_nsfw_uncensored_comfyui.safetensors",
          "4480E3154FD663D5EE6DAA86AC59D0FD211CE495B9DFEEDFB914C38F777F3A0B")
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
unet = next((n for n in opts("UNETLoader", "unet_name") if re.search(r"moss_image_sft", n, re.I)), None)
clip = next((n for n in opts("CLIPLoader", "clip_name") if re.search(r"qwen3\.?5.?9b", n, re.I)), None)
vae = next((n for n in opts("VAELoader", "vae_name") if re.search(r"flux2.?vae.?small", n, re.I)), None)
lora = next((n for n in opts("LoraLoaderModelOnly", "lora_name") if re.search(r"moss.*(nsfw|uncensored)", n, re.I)), None)
print("ComfyUI sees:", unet, "|", clip, "|", vae, "| LoRA:", lora, "| encoder type moss_image:", "moss_image" in opts("CLIPLoader", "type"))
if not (unet and clip and vae):
    raise SystemExit("MOSS-IMAGE TEST: files not visible to ComfyUI")
p = {"1": {"class_type": "UNETLoader", "inputs": {"unet_name": unet, "weight_dtype": "default"}},
     "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": clip, "type": "moss_image"}},
     "13": {"class_type": "VAELoader", "inputs": {"vae_name": vae}},
     "15": {"class_type": "SkipLayerGuidanceDiTSimple", "inputs": {"model": ["1", 0], "double_layers": "", "single_layers": "22",
            "start_percent": 0, "end_percent": 0.5}},
     "3": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": "Studio photograph of a red apple on a wooden table, soft window light"}},
     "4": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": "nsfw, worst quality, bad anatomy, watermark, jpeg artifact"}},
     "6": {"class_type": "EmptyFlux2LatentImage", "inputs": {"width": 1024, "height": 1024, "batch_size": 1}},
     "7": {"class_type": "KSampler", "inputs": {"model": ["15", 0], "positive": ["3", 0], "negative": ["4", 0], "latent_image": ["6", 0],
           "seed": 1, "steps": 25, "cfg": 2.5, "sampler_name": "er_sde", "scheduler": "simple", "denoise": 1}},
     "8": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["13", 0]}},
     "9": {"class_type": "SaveImage", "inputs": {"images": ["8", 0], "filename_prefix": "rv6m_moss_test"}}}
ok = False
req = urllib.request.Request("http://127.0.0.1:8188/api/prompt", json.dumps({"prompt": p}).encode(), {"Content-Type": "application/json"})
try:
    pid = json.load(urllib.request.urlopen(req))["prompt_id"]
except urllib.error.HTTPError as e:
    print("MOSS-IMAGE TEST REJECTED:", e.read().decode()[:600]); pid = None
t = time.time()
while pid and time.time() - t < 900:
    h = get("/api/history/" + pid)
    if pid in h:
        s = h[pid]["status"]
        print("MOSS-IMAGE TEST:", s["status_str"], f"in {time.time() - t:.0f}s")
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
    raise SystemExit(f"MOSS-Image failed its test; moved {unet} to models/_disabled so the page won't offer it.")
print("The test picture is in ComfyUI's output folder (rv6m_moss_test_*).")
PY

echo "== Updating the page"
restart
echo
echo "Done. Reload the edit page: Create mode → Create model → MOSS-Image (25 steps)."
echo "With the uncensored LoRA installed, a 🔞 Uncensored LoRA switch appears under it."
