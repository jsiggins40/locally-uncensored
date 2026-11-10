#!/usr/bin/env bash
# Adds MOSS-Image (OpenMOSS, Oct 2026; Apache-2.0) to Create mode on the edit page:
# a 5B text-to-image model, strong with long detailed prompts and with text in
# pictures, with its Qwen3.5-9B text encoder and the FLUX.2 small VAE. ComfyUI
# runs it with its own nodes (no custom nodes). Also gets an uncensored LoRA for
# explicit photographic content (axiomofmind/Moss-image-nsfw-uncensored-lora),
# checked against its published SHA-256, and two edit LoRAs for Describe edit:
# akhaliq's MOSS-Image-Edit-HQ (armV5, the one its author recommends) and chinoll's
# Moss-Image-Edit (Comfy's own). Tests a picture and an edit on the GPU, then updates
# the page: "MOSS-Image" under Create model, "MOSS Edit" under Edit model, and a
# Safe / Uncensored / Uncensored + LoRA control. About 19 GB.
# Skip the uncensored LoRA with:  MOSS_NO_NSFW=1 bash <(curl ...)
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
print("== Edit LoRAs (about 800 MB)")
fetch("akhaliq/MOSS-Image-Edit-HQ", "weights/armV5/armV5.safetensors", "loras", "moss_image_edit_hq_armV5.safetensors")
fetch(R, "loras/moss_image_edit_lora_rank_64_bf16.safetensors", "loras")
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

# Edit test: change the apple in that picture with the HQ edit LoRA
edit = next((n for n in opts("LoraLoaderModelOnly", "lora_name") if re.search(r"moss.*edit.*(hq|armv5)", n, re.I)), None)
fmts = opts("TextEncodeMOSSImageEdit", "edit_format")
if not edit or "ai-toolkit" not in fmts:
    print("MOSS EDIT TEST: skipped (edit LoRA or ComfyUI's MOSS edit node missing):", edit, fmts)
    raise SystemExit
shutil.copy(os.path.join("output", imgs[0].get("subfolder", ""), imgs[0]["filename"]), "input/rv6m_moss_edit_test.png")
enc = lambda text: {"class_type": "TextEncodeMOSSImageEdit", "inputs": {"clip": ["2", 0], "prompt": text, "vae": ["13", 0],
                    "edit_format": "ai-toolkit", "image_strength": 1, "images.image_0": ["60", 0]}}
e = {"1": p["1"], "2": p["2"], "13": p["13"],
     "16": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["1", 0], "lora_name": edit, "strength_model": 1}},
     "15": {"class_type": "SkipLayerGuidanceDiTSimple", "inputs": {"model": ["16", 0], "double_layers": "", "single_layers": "22",
            "start_percent": 0, "end_percent": 1}},
     "50": {"class_type": "LoadImage", "inputs": {"image": "rv6m_moss_edit_test.png"}},
     "60": {"class_type": "ImageScaleToTotalPixels", "inputs": {"image": ["50", 0], "upscale_method": "lanczos", "megapixels": 1, "resolution_steps": 16}},
     "27": {"class_type": "GetImageSize", "inputs": {"image": ["60", 0]}},
     "28": {"class_type": "EmptyFlux2LatentImage", "inputs": {"width": ["27", 0], "height": ["27", 1], "batch_size": 1}},
     "39": enc("Change only the apple to a green pear. Keep the table, the light and the composition unchanged."),
     "36": enc("nsfw, worst quality, bad anatomy, watermark, jpeg artifact"),
     "7": {"class_type": "KSampler", "inputs": {"model": ["15", 0], "positive": ["39", 0], "negative": ["36", 0], "latent_image": ["28", 0],
           "seed": 1, "steps": 30, "cfg": 3, "sampler_name": "euler", "scheduler": "normal", "denoise": 1}},
     "8": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["13", 0]}},
     "9": {"class_type": "SaveImage", "inputs": {"images": ["8", 0], "filename_prefix": "rv6m_moss_edit_test"}}}
req = urllib.request.Request("http://127.0.0.1:8188/api/prompt", json.dumps({"prompt": e}).encode(), {"Content-Type": "application/json"})
try:
    pid = json.load(urllib.request.urlopen(req))["prompt_id"]
except urllib.error.HTTPError as err:
    print("MOSS EDIT TEST REJECTED:", err.read().decode()[:600]); raise SystemExit
t = time.time()
while time.time() - t < 900:
    h = get("/api/history/" + pid)
    if pid in h:
        s = h[pid]["status"]
        print("MOSS EDIT TEST:", s["status_str"], f"in {time.time() - t:.0f}s (rv6m_moss_edit_test_* in the output folder: the apple should be a pear)")
        if s["status_str"] != "success":
            print("  ", json.dumps(s)[:600])
            print("  MOSS-Image itself works; MOSS Edit may not. Send this to Claude.")
        break
    time.sleep(2)
PY

echo "== Updating the page"
restart
echo
echo "Done. Reload the edit page:"
echo "  ✍️ Create → MOSS-Image (25 steps)"
echo "  ✨ Describe edit → Edit model → MOSS Edit (up to 4 photos)"
echo "  Both have a Safe / 🔞 Uncensored / 🔞 + LoRA control."
