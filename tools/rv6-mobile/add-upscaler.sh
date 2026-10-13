#!/usr/bin/env bash
# Adds ⬆️ Upscale to the edit page: SeedVR2 7B (ByteDance's one-step restorer,
# built into ComfyUI) makes any result 2× or 4K-8K with real added detail.
# About 9 GB. Tests a 2× and a 4K upscale on the GPU at the end.
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
import os, shutil, sys
from huggingface_hub import HfApi, hf_hub_download
free = shutil.disk_usage(os.path.realpath("models")).free / 1e9
print(f"== Free disk: {free:.0f} GB")
if free < 25:
    sys.exit("Need about 25 GB free. Delete something first.")
R = "Comfy-Org/SeedVR2"
try:
    fl = [f for f in HfApi().list_repo_files(R) if f.endswith(".safetensors")]
except Exception as e:
    sys.exit(f"Can't list {R}: {type(e).__name__}: {str(e)[:200]}")
for f in fl: print("    ", f)
# The 7B model (better than the 3B); full precision when the repo has it, else int8
dm = sorted([f for f in fl if f.startswith("diffusion_models/") and "7b" in f.lower()],
            key=lambda f: ("int8" in f.lower() or "fp8" in f.lower(), "sharp" in f.lower(), f))
va = [f for f in fl if f.startswith("vae/") and "seedvr2" in f.lower()]
if not (dm and va):
    sys.exit("Could not find the SeedVR2 7B model and VAE in " + R)
print("== SeedVR2 7B + VAE")
for path, sub in ((dm[0], "diffusion_models"), (va[0], "vae")):
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
import json, re, time, urllib.request, urllib.error
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
unet = sorted((n for n in opts("UNETLoader", "unet_name") if re.search("seedvr2", n, re.I)),
              key=lambda n: ("7b" in n.lower(), not re.search("int8|fp8", n, re.I)))
vae = [n for n in opts("VAELoader", "vae_name") if re.search("seedvr2", n, re.I)]
print("ComfyUI sees:", unet[-1:] , vae[-1:])
if not (unet and vae): raise SystemExit("UPSCALE TEST: files not visible to ComfyUI")
tiles = {"tile_size": 512, "overlap": 128, "temporal_size": 4096, "temporal_overlap": 8}

def run(label, w, h):
    # Same graph the page builds
    p = {"u1": {"class_type": "UNETLoader", "inputs": {"unet_name": unet[-1], "weight_dtype": "default"}},
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

ok = run("UPSCALE TEST 2x", 2048, 1536)
ok = run("UPSCALE TEST 4K", 3840, 2880) and ok
if not ok: raise SystemExit("The upscaler failed its test; see above.")
PY
echo
echo "Done. Reload the edit page: results now have a ⬆️ Upscale button (2×, 4K, 6K, 8K)."
