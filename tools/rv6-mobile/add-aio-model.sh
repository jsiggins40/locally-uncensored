#!/usr/bin/env bash
# Adds Phr00t's Qwen-Image-Edit Rapid AIO (NSFW build): Qwen-Image-Edit 2511
# with NSFW LoRAs, the Lightning speed-up, text encoder and VAE merged into one
# fp8 checkpoint. 4-8 steps at CFG 1. The page offers it as an edit model next
# to "Qwen 2511 + LoRA". About 28 GB. Picks the newest NSFW version in the repo.
#   bash <(curl -fsSL https://raw.githubusercontent.com/jsiggins40/locally-uncensored/rv6-mobile-editor/tools/rv6-mobile/add-aio-model.sh)
set -euo pipefail
cd "$HOME/ComfyUI"

venv/bin/python - <<'PY'
import os, re, shutil, sys
from huggingface_hub import HfApi, hf_hub_download

repo = "Phr00t/Qwen-Image-Edit-Rapid-AIO"
files = [f for f in HfApi().list_repo_files(repo) if f.endswith(".safetensors")]
ver = lambda f: int((re.findall(r"v(\d+)", f.lower()) or ["0"])[-1])
nsfw = [f for f in files if "nsfw" in f.lower()]
if not nsfw:
    sys.exit(f"No NSFW checkpoint found in {repo}:\n" + "\n".join(files))
pick = max(nsfw, key=ver)
print(f"== Newest NSFW version: {pick}  (of {len(nsfw)})")
os.makedirs("models/checkpoints", exist_ok=True)
name = os.path.basename(pick)
if "aio" not in name.lower():  # keep "Rapid-AIO" in the name: the page looks for it
    name = "Qwen-Rapid-AIO-" + name
dest = os.path.join("models/checkpoints", name)
for old in os.listdir("models/checkpoints"):  # one AIO at a time: drop older versions
    if re.search(r"rapid.?aio", old, re.I) and old != name:
        os.remove(os.path.join("models/checkpoints", old)); print("   removed older", old)
if os.path.exists(dest) and os.path.getsize(dest) > 0:
    print("   already have", dest); sys.exit(0)
if shutil.disk_usage(os.path.realpath("models")).free / 1e9 < 35:
    sys.exit("Need about 35 GB free. Delete something first.")
shutil.move(hf_hub_download(repo, pick, local_dir="models/_hf_tmp"), dest)
shutil.rmtree("models/_hf_tmp", ignore_errors=True)
print("Installed:", dest)
PY

echo "== Test edit with Rapid AIO"
curl -fsS -o /dev/null -X POST localhost:8188/api/refresh 2>/dev/null || true
venv/bin/python - <<'PY'
import json, re, time, urllib.request, urllib.error
from PIL import Image
Image.new("RGB", (512, 512), (120, 120, 120)).save("input/rv6m_test_img.png")
def get(path):
    return json.load(urllib.request.urlopen("http://127.0.0.1:8188" + path))
v = get("/api/object_info/CheckpointLoaderSimple")["CheckpointLoaderSimple"]["input"]["required"]["ckpt_name"]
ckpts = v[0] if isinstance(v[0], list) else v[1].get("options", [])
ckpt = next((n for n in ckpts if re.search(r"rapid.?aio", n, re.I)), None)
print("ComfyUI sees:", ckpt)
if not ckpt:
    raise SystemExit("AIO TEST: checkpoint not visible to ComfyUI")
enc = lambda t: {"class_type": "TextEncodeQwenImageEditPlus", "inputs": {"clip": ["1", 1], "prompt": t, "vae": ["1", 2], "image1": ["5", 0]}}
p = {
    "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": ckpt}},
    "5": {"class_type": "LoadImage", "inputs": {"image": "rv6m_test_img.png"}},
    "3": enc("make the whole picture bright red"), "4": enc(""),
    "6": {"class_type": "VAEEncode", "inputs": {"pixels": ["5", 0], "vae": ["1", 2]}},
    "7": {"class_type": "KSampler", "inputs": {"model": ["1", 0], "positive": ["3", 0], "negative": ["4", 0], "latent_image": ["6", 0],
          "seed": 1, "steps": 4, "cfg": 1, "sampler_name": "euler_ancestral", "scheduler": "beta", "denoise": 1}},
    "8": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["1", 2]}},
    "9": {"class_type": "PreviewImage", "inputs": {"images": ["8", 0]}},
}
req = urllib.request.Request("http://127.0.0.1:8188/api/prompt", json.dumps({"prompt": p}).encode(), {"Content-Type": "application/json"})
try:
    pid = json.load(urllib.request.urlopen(req))["prompt_id"]
except urllib.error.HTTPError as e:
    raise SystemExit("AIO TEST REJECTED: " + e.read().decode()[:800])
t = time.time()
while True:
    h = get("/api/history/" + pid)
    if pid in h:
        st = h[pid]["status"]
        print("AIO TEST:", st["status_str"], f"in {time.time() - t:.0f}s")
        if st["status_str"] != "success": print(json.dumps(st)[:800])
        break
    if time.time() - t > 900: raise SystemExit("AIO TEST TIMED OUT")
    time.sleep(2)
PY
echo
echo "Done. Reload the edit page: Describe edit / Change now offer ⚡ Rapid AIO as the edit model."
