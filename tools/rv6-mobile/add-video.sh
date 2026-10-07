#!/usr/bin/env bash
# Adds 🎬 Video to the edit page: Wan 2.2 14B image-to-video (high + low noise
# models, full precision), the LightX2V 4-step LoRAs and the SVI 2.0 Pro LoRAs
# that chain 5-second clips into one long video (5-60 s on the page), plus
# KJNodes for the SVI node. About 75 GB. Runs a short 2-clip test video at the end.
#   bash <(curl -fsSL https://raw.githubusercontent.com/jsiggins40/locally-uncensored/rv6-mobile-editor/tools/rv6-mobile/add-video.sh)
set -euo pipefail
cd "$HOME/ComfyUI"
SRC="https://raw.githubusercontent.com/jsiggins40/locally-uncensored/rv6-mobile-editor/tools/rv6-mobile"

restart() {
  if command -v edit-restart >/dev/null; then edit-restart; else
    sudo curl -fsSL "$SRC/edit-restart.sh" -o /usr/local/bin/edit-restart && sudo chmod +x /usr/local/bin/edit-restart && edit-restart
  fi
}

echo "== KJNodes (has the SVI 2.0 Pro node)"
KJ=custom_nodes/ComfyUI-KJNodes
if [ -d "$KJ/.git" ]; then git -C "$KJ" pull -q --ff-only || true
else git clone -q --depth 1 https://github.com/kijai/ComfyUI-KJNodes.git "$KJ"; fi
[ -f "$KJ/requirements.txt" ] && venv/bin/pip install -q -r "$KJ/requirements.txt"
grep -q "WanImageToVideoSVIPro" "$KJ"/nodes/*.py || { echo "This KJNodes has no SVI Pro node; stopping."; exit 1; }

venv/bin/python - <<'PY'
import os, shutil, sys
from huggingface_hub import HfApi, hf_hub_download
api = HfApi()
free = shutil.disk_usage(os.path.realpath("models")).free / 1e9
print(f"== Free disk: {free:.0f} GB")
if free < 85:
    sys.exit("Need about 85 GB free. Delete something first.")

def ls(repo):
    try:
        return api.list_repo_files(repo)
    except Exception as e:
        sys.exit(f"Can't list {repo}: {type(e).__name__}: {str(e)[:200]}")

def fetch(repo, path, subdir):
    dest = os.path.join("models", subdir, os.path.basename(path))
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        print("   already have", dest); return
    print(f"   downloading {os.path.basename(path)} from {repo}")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    shutil.move(hf_hub_download(repo, path, local_dir="models/_hf_tmp"), dest)

def pick(files, *must, prefer=("fp16", "bf16")):
    c = [f for f in files if f.endswith(".safetensors") and all(m in f.lower() for m in must)]
    if not c:
        sys.exit("Not found: " + " + ".join(must))
    return sorted(c, key=lambda f: ([p not in f.lower() for p in prefer], "fp8" in f.lower(), f))[0]

W22 = "Comfy-Org/Wan_2.2_ComfyUI_Repackaged"
f22 = ls(W22)
print("== Wan 2.2 image-to-video 14B (high + low noise)")
fetch(W22, pick(f22, "diffusion_models/", "wan2.2_i2v_high_noise_14b"), "diffusion_models")
fetch(W22, pick(f22, "diffusion_models/", "wan2.2_i2v_low_noise_14b"), "diffusion_models")
print("== LightX2V 4-step LoRAs")
fetch(W22, pick(f22, "loras/", "i2v", "lightx2v", "high_noise"), "loras")
fetch(W22, pick(f22, "loras/", "i2v", "lightx2v", "low_noise"), "loras")
print("== VAE and text encoder")
fetch(W22, pick(f22, "vae/", "wan_2.1_vae"), "vae")
W21 = "Comfy-Org/Wan_2.1_ComfyUI_repackaged"
fetch(W21, pick(ls(W21), "text_encoders/", "umt5_xxl"), "text_encoders")
print("== SVI 2.0 Pro LoRAs (join clips into long videos)")
S = "vita-video-gen/svi-model"
fs = ls(S)
fetch(S, pick(fs, "svi_wan2.2-i2v-a14b_high_noise", "pro", prefer=()), "loras")
fetch(S, pick(fs, "svi_wan2.2-i2v-a14b_low_noise", "pro", prefer=()), "loras")
shutil.rmtree("models/_hf_tmp", ignore_errors=True)
PY

echo "== Restarting ComfyUI so it loads KJNodes, and updating the page"
restart

echo "== Test: a short 2-clip video (first load of the 14B models takes a few minutes)"
venv/bin/python - <<'PY'
import json, re, time, urllib.request, urllib.error
from PIL import Image
Image.new("RGB", (512, 512), (120, 140, 160)).save("input/rv6m_test_img.png")
def get(path): return json.load(urllib.request.urlopen("http://127.0.0.1:8188" + path))
def opts(node, inp):
    v = get("/api/object_info/" + node).get(node, {}).get("input", {})
    v = v.get("required", {}).get(inp) or v.get("optional", {}).get(inp)
    return (v[0] if isinstance(v[0], list) else v[1].get("options", [])) if v else []
if "WanImageToVideoSVIPro" not in get("/api/object_info/WanImageToVideoSVIPro"):
    raise SystemExit("VIDEO TEST: ComfyUI didn't load the SVI node (KJNodes). See: tmux attach -t comfy")
units, clips, loras = opts("UNETLoader", "unet_name"), opts("CLIPLoader", "clip_name"), opts("LoraLoaderModelOnly", "lora_name")
one = lambda xs, rx: (sorted(n for n in xs if re.search(rx, n, re.I)) or [None])[-1]
v = dict(high=one(units, r"wan2\.2_i2v_high_noise"), low=one(units, r"wan2\.2_i2v_low_noise"), clip=one(clips, "umt5"),
         vae=one(opts("VAELoader", "vae_name"), r"wan_2\.1_vae"), lightH=one(loras, "i2v.*lightx2v.*high"), lightL=one(loras, "i2v.*lightx2v.*low"),
         sviH=one(loras, "svi.*high.*pro"), sviL=one(loras, "svi.*low.*pro"))
print("ComfyUI sees:", v)
if not all(v.values()): raise SystemExit("VIDEO TEST: some files are not visible to ComfyUI")
# Same graph the page builds, tiny: 2 clips of 17 frames at 320x320
p = {
    "m1": {"class_type": "UNETLoader", "inputs": {"unet_name": v["high"], "weight_dtype": "default"}},
    "m2": {"class_type": "UNETLoader", "inputs": {"unet_name": v["low"], "weight_dtype": "default"}},
    "te": {"class_type": "CLIPLoader", "inputs": {"clip_name": v["clip"], "type": "wan"}},
    "va": {"class_type": "VAELoader", "inputs": {"vae_name": v["vae"]}},
    "h1": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["m1", 0], "lora_name": v["lightH"], "strength_model": 1}},
    "h2": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["h1", 0], "lora_name": v["sviH"], "strength_model": 1}},
    "hs": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["h2", 0], "shift": 5}},
    "l1": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["m2", 0], "lora_name": v["lightL"], "strength_model": 1}},
    "l2": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["l1", 0], "lora_name": v["sviL"], "strength_model": 1}},
    "ls": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["l2", 0], "shift": 5}},
    "pos": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["te", 0], "text": "the camera slowly pushes in"}},
    "neg": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["te", 0], "text": "blurry"}},
    "img": {"class_type": "LoadImage", "inputs": {"image": "rv6m_test_img.png"}},
    "fit": {"class_type": "ImageScale", "inputs": {"image": ["img", 0], "upscale_method": "lanczos", "width": 320, "height": 320, "crop": "center"}},
    "anc": {"class_type": "VAEEncode", "inputs": {"pixels": ["fit", 0], "vae": ["va", 0]}},
}
frames = None
for i in range(2):
    c = f"c{i}"
    p[c + "i"] = {"class_type": "WanImageToVideoSVIPro", "inputs": {"positive": ["pos", 0], "negative": ["neg", 0], "length": 17,
                  "anchor_samples": ["anc", 0], "motion_latent_count": 1, **({"prev_samples": [f"c{i-1}l", 0]} if i else {})}}
    for s, model, a, b, noise in (("h", "hs", 0, 2, "enable"), ("l", "ls", 2, 10000, "disable")):
        p[c + s] = {"class_type": "KSamplerAdvanced", "inputs": {"model": [model, 0], "add_noise": noise, "noise_seed": 1 + i, "steps": 4, "cfg": 1,
                    "sampler_name": "euler", "scheduler": "simple", "positive": [c + "i", 0], "negative": [c + "i", 1],
                    "latent_image": [c + "i", 2] if s == "h" else [c + "h", 0], "start_at_step": a, "end_at_step": b,
                    "return_with_leftover_noise": "enable" if s == "h" else "disable"}}
    p[c + "d"] = {"class_type": "VAEDecode", "inputs": {"samples": [c + "l", 0], "vae": ["va", 0]}}
    if i == 0: frames = [c + "d", 0]; continue
    p[c + "x"] = {"class_type": "ImageFromBatch", "inputs": {"image": [c + "d", 0], "batch_index": 1, "length": 4096}}
    p[c + "j"] = {"class_type": "ImageBatch", "inputs": {"image1": frames, "image2": [c + "x", 0]}}
    frames = [c + "j", 0]
p["mv"] = {"class_type": "CreateVideo", "inputs": {"images": frames, "fps": 16}}
p["9"] = {"class_type": "SaveVideo", "inputs": {"video": ["mv", 0], "filename_prefix": "rv6_video_test", "format": "mp4", "format.codec": "h264"}}
req = urllib.request.Request("http://127.0.0.1:8188/api/prompt", json.dumps({"prompt": p}).encode(), {"Content-Type": "application/json"})
try:
    pid = json.load(urllib.request.urlopen(req))["prompt_id"]
except urllib.error.HTTPError as e:
    raise SystemExit("VIDEO TEST REJECTED: " + e.read().decode()[:1200])
t = time.time()
while True:
    h = get("/api/history/" + pid)
    if pid in h:
        st = h[pid]["status"]
        out = (h[pid].get("outputs", {}).get("9") or {}).get("images") or []
        print("VIDEO TEST:", st["status_str"], f"in {time.time() - t:.0f}s", "->", out[0]["filename"] if out else "no file")
        if st["status_str"] != "success": print(json.dumps(st)[:1200])
        break
    if time.time() - t > 1800: raise SystemExit("VIDEO TEST TIMED OUT")
    time.sleep(3)
PY
echo
echo "Done. Reload the edit page: there is now a 🎬 Video mode."
