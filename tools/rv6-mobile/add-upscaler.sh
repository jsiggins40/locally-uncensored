#!/usr/bin/env bash
# Adds ⬆️ Upscale to the edit page: SeedVR2 7B (ByteDance's one-step restorer,
# built into ComfyUI) makes any result 2× or 4K-8K with real added detail.
# Gets the full-precision (fp16) 7B and 7B "sharp" models: the H200 has the
# memory, so none of the int8/fp8/nvfp4 low-memory versions. About 34 GB.
# Tests both on the GPU at the end, plus a video upscale. Safe to run again
# (it skips what it already has and just re-runs the tests).
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
import av, numpy as np
from PIL import Image, ImageDraw
# A test picture with edges and text, so a broken upscale is easy to spot
im = Image.new("RGB", (1024, 768), (90, 120, 150))
d = ImageDraw.Draw(im)
for i in range(0, 1024, 64): d.line([(i, 0), (1023 - i, 767)], fill=(230, 200, 90), width=3)
d.ellipse([312, 184, 712, 584], outline=(250, 250, 250), width=8); d.text((40, 40), "rv6 upscale test", fill=(255, 255, 255))
im.save("output/rv6m_up_test.png")
# A 33-frame video of the same picture sliding sideways (16 fps, like local Wan)
c = av.open("output/rv6m_up_test.mp4", "w"); s = c.add_stream("h264", rate=16); s.width, s.height, s.pix_fmt = 320, 240, "yuv420p"
small = im.resize((320, 240))
for i in range(33):
    f = Image.new("RGB", (320, 240)); f.paste(small, (i * 3 - 48, 0))
    for pk in s.encode(av.VideoFrame.from_ndarray(np.asarray(f), format="rgb24")): c.mux(pk)
for pk in s.encode(): c.mux(pk)
c.close()
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

def run(label, model, w, h, video=None):
    # Same graphs the page builds
    if video:
        tl = {**tiles, "temporal_size": 64}
        p = {"u1": {"class_type": "UNETLoader", "inputs": {"unet_name": model, "weight_dtype": "default"}},
             "u2": {"class_type": "VAELoader", "inputs": {"vae_name": vae[-1]}},
             "5": {"class_type": "LoadVideo", "inputs": {"file": "rv6m_up_test.mp4 [output]"}},
             "gc": {"class_type": "GetVideoComponents", "inputs": {"video": ["5", 0]}},
             "u3": {"class_type": "ImageScale", "inputs": {"image": ["gc", 0], "upscale_method": "lanczos", "width": w, "height": h, "crop": "disabled"}},
             "u4": {"class_type": "SeedVR2Preprocess", "inputs": {"resized_images": ["u3", 0]}},
             "u5": {"class_type": "VAEEncodeTiled", "inputs": {"pixels": ["u4", 0], "vae": ["u2", 0], **tl}},
             "ch": {"class_type": "SeedVR2TemporalChunk", "inputs": {"latent": ["u5", 0], "temporal_overlap": 2, **video}},
             "u6": {"class_type": "SeedVR2Conditioning", "inputs": {"model": ["u1", 0], "vae_conditioning": ["ch", 0]}},
             "7": {"class_type": "KSampler", "inputs": {"model": ["u1", 0], "positive": ["u6", 0], "negative": ["u6", 1], "latent_image": ["ch", 0],
                   "seed": 1, "steps": 1, "cfg": 1, "sampler_name": "euler", "scheduler": "simple", "denoise": 1}},
             "mg": {"class_type": "SeedVR2TemporalMerge", "inputs": {"latents": ["7", 0], "temporal_overlap": ["ch", 1]}},
             "u8": {"class_type": "VAEDecodeTiled", "inputs": {"samples": ["mg", 0], "vae": ["u2", 0], **tl}},
             "u9": {"class_type": "SeedVR2PostProcessing", "inputs": {"images": ["u8", 0], "original_resized_images": ["u3", 0], "color_correction_method": "lab"}},
             "mv": {"class_type": "CreateVideo", "inputs": {"images": ["u9", 0], "fps": ["gc", 2], "audio": ["gc", 1]}},
             "9": {"class_type": "SaveVideo", "inputs": {"video": ["mv", 0], "filename_prefix": "rv6m_up_test_vid", "format": "mp4", "format.codec": "h264"}}}
        return wait(label, p)
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
    return wait(label, p)

def wait(label, p):
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
            if out and out[0]["filename"].endswith(".mp4"):
                c = av.open("output/" + out[0]["filename"]); v = c.streams.video[0]
                n = sum(1 for _ in c.decode(video=0)); c.close()
                desc = f"{v.width}x{v.height}, {n} frames"
            else:
                size = Image.open("output/" + out[0]["filename"]).size if out else None
                desc = f"{size[0]}x{size[1]}" if size else "no file"
            print(f"{label}: {st['status_str']} in {time.time() - t:.0f}s ->", desc)
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
# Video: in one piece ("auto"), then forced into 3 overlapping pieces, as a long video would be
ok = run("UPSCALE TEST VIDEO", normal[-1], 640, 480, {"chunking_mode": "auto"}) and ok
ok = run("UPSCALE TEST VIDEO IN PIECES", normal[-1], 640, 480, {"chunking_mode": "manual", "chunking_mode.frames_per_chunk": 17}) and ok
print("   (both video tests should say 640x480, 33 frames)")
ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
print(f"   VM memory: {ram:.0f} GB" + ("" if ram >= 150 else
      "  -- under 150 GB: long videos at 1080p+ may run out of memory; upscale shorter clips or use 2x"))
if not ok: raise SystemExit("The upscaler failed its test; see above.")
PY
echo
echo "Done. Reload the edit page: results now have a ⬆️ Upscale button (pictures: 2×, 4K, 6K, 8K; videos: 2×, 1080p, 1440p; Natural or Sharp)."
