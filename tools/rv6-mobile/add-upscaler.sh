#!/usr/bin/env bash
# Adds ⬆️ Upscale to the edit page: SeedVR2 7B (ByteDance's one-step restorer,
# built into ComfyUI) makes any result 2× or 4K-8K with real added detail.
# Gets the full-precision (fp16) 7B and 7B "sharp" models: the H200 has the
# memory, so none of the int8/fp8/nvfp4 low-memory versions. About 34 GB.
# Tests both on the GPU at the end.
#   bash <(curl -fsSL https://raw.githubusercontent.com/jsiggins40/locally-uncensored/rv6-mobile-editor/tools/rv6-mobile/add-upscaler.sh)
set -euo pipefail
cd "$HOME/ComfyUI"
SRC="https://raw.githubusercontent.com/jsiggins40/locally-uncensored/rv6-mobile-editor/tools/rv6-mobile"

restart() {
  if command -v edit-restart >/dev/null; then edit-restart; else
    sudo curl -fsSL "$SRC/edit-restart.sh" -o /usr/local/bin/edit-restart && sudo chmod +x /usr/local/bin/edit-restart && edit-restart
  fi
}

echo "== ComfyUI must have the SeedVR2 nodes"
if ! grep -q "SeedVR2Conditioning" comfy_extras/nodes_seedvr.py 2>/dev/null; then
  echo "   updating ComfyUI"
  git pull -q --ff-only && venv/bin/pip install -q -r requirements.txt
  restart
fi
grep -q "SeedVR2Conditioning" comfy_extras/nodes_seedvr.py || { echo "ComfyUI still has no SeedVR2 nodes; stopping."; exit 1; }

venv/bin/python - <<'PY'
import os, re, shutil, sys
from huggingface_hub import HfApi, hf_hub_download
free = shutil.disk_usage(os.path.realpath("models")).free / 1e9
print(f"== Free disk: {free:.0f} GB")
if free < 45:
    sys.exit("Need about 45 GB free. Delete something first.")
R = "Comfy-Org/SeedVR2"
try:
    fl = [f for f in HfApi().list_repo_files(R) if f.endswith(".safetensors")]
except Exception as e:
    sys.exit(f"Can't list {R}: {type(e).__name__}: {str(e)[:200]}")
for f in fl: print("    ", f)
# The 7B (better than the 3B), normal and sharp. Precision, best first: fp16,
# then the 8-bit files; nvfp4 last (made for Blackwell GPUs, not the H200)
rank = lambda f: (0 if re.search("fp16|bf16", f, re.I) else 1 if re.search("int8|fp8", f, re.I) else 3 if "nvfp4" in f.lower() else 2, f)
seven = [f for f in fl if f.startswith("diffusion_models/") and "7b" in f.lower()]
normal = sorted([f for f in seven if "sharp" not in f.lower()], key=rank)
sharp = sorted([f for f in seven if "sharp" in f.lower()], key=rank)
va = sorted([f for f in fl if f.startswith("vae/") and "seedvr2" in f.lower()])
if not (normal and va):
    sys.exit("Could not find the SeedVR2 7B model and VAE in " + R)
print("== SeedVR2 7B (normal + sharp) + VAE")
for path, sub in [(normal[0], "diffusion_models")] + [(sharp[0], "diffusion_models")] * bool(sharp) + [(va[0], "vae")]:
    dest = os.path.join("models", sub, os.path.basename(path))
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        print("   already have", dest); continue
    print("   downloading", path)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    shutil.move(hf_hub_download(R, path, local_dir="models/_hf_tmp"), dest)
shutil.rmtree("models/_hf_tmp", ignore_errors=True)
PY

echo "== Restarting ComfyUI and updating the page"
restart

echo "== Tests on the GPU (the first load takes a minute or two)"
venv/bin/python - <<'PY'
import json, os, re, shutil, time, urllib.request, urllib.error
from PIL import Image, ImageDraw
# A test picture with edges and text, so a broken upscale is easy to spot
im = Image.new("RGB", (1024, 768), (90, 120, 150))
d = ImageDraw.Draw(im)
for i in range(0, 1024, 64): d.line([(i, 0), (1023 - i, 767)], fill=(230, 200, 90), width=3)
d.ellipse([312, 184, 712, 584], outline=(250, 250, 250), width=8); d.text((40, 40), "rv6 upscale test", fill=(255, 255, 255))
im.save("output/rv6m_up_test.png")
def get(path): return json.load(urllib.request.urlopen("http://127.0.0.1:8188" + path))
def opts(node, inp):
    v = get("/api/object_info/" + node).get(node, {}).get("input", {})
    v = v.get("required", {}).get(inp) or v.get("optional", {}).get(inp)
    return (v[0] if isinstance(v[0], list) else v[1].get("options", [])) if v else []
if "SeedVR2Conditioning" not in get("/api/object_info/SeedVR2Conditioning"):
    raise SystemExit("UPSCALE TEST: ComfyUI didn't load the SeedVR2 nodes. See: tmux attach -t comfy")
# Same choice as the page: 7B, fp16 first, normal and sharp
rank = lambda n: ("7b" in n.lower(), bool(re.search("fp16|bf16", n, re.I)), bool(re.search("int8|fp8", n, re.I)), n)
svr = [n for n in opts("UNETLoader", "unet_name") if re.search("seedvr2", n, re.I)]
normal = sorted((n for n in svr if "sharp" not in n.lower()), key=rank)
sharp = sorted((n for n in svr if "sharp" in n.lower()), key=rank)
vae = sorted(n for n in opts("VAELoader", "vae_name") if re.search("seedvr2", n, re.I))
print("ComfyUI sees:", normal[-1:], sharp[-1:], vae[-1:])
if not (normal and vae): raise SystemExit("UPSCALE TEST: files not visible to ComfyUI")
tiles = {"tile_size": 512, "overlap": 128, "temporal_size": 4096, "temporal_overlap": 8}

def run(label, model, w, h):
    # Same graph the page builds
    p = {"u1": {"class_type": "UNETLoader", "inputs": {"unet_name": model, "weight_dtype": "default"}},
         "u2": {"class_type": "VAELoader", "inputs": {"vae_name": vae[-1]}},
         "5": {"class_type": "LoadImage", "inputs": {"image": "rv6m_up_test.png [output]"}},
         "u3": {"class_type": "ImageScale", "inputs": {"image": ["5", 0], "upscale_method": "lanczos", "width": w, "height": h, "crop": "disabled"}},
         "u4": {"class_type": "SeedVR2Preprocess", "inputs": {"resized_images": ["u3", 0]}},
         "u5": {"class_type": "VAEEncodeTiled", "inputs": {"pixels": ["u4", 0], "vae": ["u2", 0], **tiles}},
         "u6": {"class_type": "SeedVR2Conditioning", "inputs": {"model": ["u1", 0], "vae_conditioning": ["u5", 0]}},
         "7": {"class_type": "KSampler", "inputs": {"model": ["u1", 0], "positive": ["u6", 0], "negative": ["u6", 1], "latent_image": ["u5", 0],
               "seed": 1, "steps": 1, "cfg": 1, "sampler_name": "euler", "scheduler": "simple", "denoise": 1}},
         "u8": {"class_type": "VAEDecodeTiled", "inputs": {"samples": ["7", 0], "vae": ["u2", 0], **tiles}},
         "u9": {"class_type": "SeedVR2PostProcessing", "inputs": {"images": ["u8", 0], "original_resized_images": ["u3", 0], "color_correction_method": "lab"}},
         "9": {"class_type": "SaveImage", "inputs": {"images": ["u9", 0], "filename_prefix": "rv6m_up_test_out"}}}
    req = urllib.request.Request("http://127.0.0.1:8188/api/prompt", json.dumps({"prompt": p}).encode(), {"Content-Type": "application/json"})
    try:
        pid = json.load(urllib.request.urlopen(req))["prompt_id"]
    except urllib.error.HTTPError as e:
        print(label, "REJECTED:", e.read().decode()[:800]); return False
    t = time.time()
    while time.time() - t < 1200:
        h_ = get("/api/history/" + pid)
        if pid in h_:
            st = h_[pid]["status"]
            out = (h_[pid].get("outputs", {}).get("9") or {}).get("images") or []
            size = Image.open("output/" + out[0]["filename"]).size if out else None
            print(f"{label}: {st['status_str']} in {time.time() - t:.0f}s ->", f"{size[0]}x{size[1]}" if size else "no file")
            if st["status_str"] != "success": print("  ", json.dumps(st)[:800])
            return st["status_str"] == "success"
        time.sleep(2)
    print(label, "TIMED OUT"); return False

ok = run("UPSCALE TEST 2x", normal[-1], 2048, 1536)
ok = run("UPSCALE TEST 4K", normal[-1], 3840, 2880) and ok
if sharp and not run("UPSCALE TEST 4K SHARP", sharp[-1], 3840, 2880):
    os.makedirs("models/_disabled", exist_ok=True)  # so the page doesn't offer a model that fails
    shutil.move(os.path.join("models/diffusion_models", sharp[-1]), os.path.join("models/_disabled", os.path.basename(sharp[-1])))
    print("   moved", sharp[-1], "to models/_disabled; the page will offer Natural only")
if not ok: raise SystemExit("The upscaler failed its test; see above.")
PY
echo
echo "Done. Reload the edit page: results now have a ⬆️ Upscale button (2×, 4K, 6K, 8K; Natural or Sharp)."
