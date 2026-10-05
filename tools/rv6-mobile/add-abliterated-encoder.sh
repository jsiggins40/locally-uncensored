#!/usr/bin/env bash
# Adds an abliterated Qwen2.5-VL 7B text encoder for Qwen-Image-Edit 2511 and
# Qwen-Image 2512. Qwen reads your instruction through this language model;
# the abliterated one (refusal direction removed from its language layers,
# vision part untouched) passes explicit instructions on unsoftened. The page
# gets a "Text encoder: Standard / 🔓 Abliterated" switch to compare.
# About 16 GB (bf16). Then runs one test edit with it.
#   bash <(curl -fsSL https://raw.githubusercontent.com/jsiggins40/locally-uncensored/rv6-mobile-editor/tools/rv6-mobile/add-abliterated-encoder.sh)
set -euo pipefail
cd "$HOME/ComfyUI"

venv/bin/python - <<'PY'
import os, shutil, sys
from huggingface_hub import HfApi, hf_hub_download

api = HfApi()
NAME = "qwen_2.5_vl_7b_abliterated_bf16.safetensors"  # fixed name the page recognises
dest = os.path.join("models", "text_encoders", NAME)
if os.path.exists(dest) and os.path.getsize(dest) > 0:
    print("== Already have", dest); sys.exit(0)
if shutil.disk_usage(os.path.realpath("models")).free / 1e9 < 20:
    sys.exit("Need about 20 GB free. Delete something first.")
# ComfyUI-ready single files only (the original huihui-ai repo is split in the
# transformers layout, which ComfyUI can't load directly)
for repo in ("martossien/qwen_2.5_vl_7b_uncensored_comfy_ready",):
    try:
        files = [s for s in api.model_info(repo, files_metadata=True).siblings if s.rfilename.endswith(".safetensors")]
    except Exception as e:
        print(f"   {repo}: {type(e).__name__}: {e}"); continue
    for s in files: print(f"   found {s.rfilename} ({(s.size or 0) / 1e9:.1f} GB)")
    if not files: continue
    best = max(files, key=lambda s: ("bf16" in s.rfilename.lower(), s.size or 0))
    print(f"== Downloading {best.rfilename} from {repo}")
    shutil.move(hf_hub_download(repo, best.rfilename, local_dir="models/_hf_tmp"), dest)
    shutil.rmtree("models/_hf_tmp", ignore_errors=True)
    print("Installed:", dest); break
else:
    sys.exit("Could not download an abliterated text encoder.")
PY

echo "== Test edit with the abliterated encoder"
curl -fsS -o /dev/null -X POST localhost:8188/api/refresh 2>/dev/null || true
venv/bin/python - <<'PY'
import json, re, time, urllib.request, urllib.error
from PIL import Image
Image.new("RGB", (512, 512), (120, 120, 120)).save("input/rv6m_test_img.png")
def get(path):
    return json.load(urllib.request.urlopen("http://127.0.0.1:8188" + path))
def opts(node, inp):
    v = get("/api/object_info/" + node).get(node, {}).get("input", {})
    v = v.get("required", {}).get(inp) or v.get("optional", {}).get(inp)
    return (v[0] if isinstance(v[0], list) else v[1].get("options", [])) if v else []
unet = (sorted(n for n in opts("UNETLoader", "unet_name") if "qwen" in n.lower() and "edit" in n.lower()) or [None])[-1]
clip = next((n for n in opts("CLIPLoader", "clip_name") if "abliterated" in n.lower()), None)
vae = (sorted(n for n in opts("VAELoader", "vae_name") if "qwen_image_vae" in n.lower()) or [None])[-1]
print("ComfyUI sees:", unet, "|", clip, "|", vae)
if not (unet and clip and vae):
    raise SystemExit("ENCODER TEST: files not visible to ComfyUI")
enc = lambda t: {"class_type": "TextEncodeQwenImageEditPlus", "inputs": {"clip": ["2", 0], "prompt": t, "vae": ["13", 0], "image1": ["5", 0]}}
p = {
    "1": {"class_type": "UNETLoader", "inputs": {"unet_name": unet, "weight_dtype": "default"}},
    "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": clip, "type": "qwen_image"}},
    "13": {"class_type": "VAELoader", "inputs": {"vae_name": vae}},
    "5": {"class_type": "LoadImage", "inputs": {"image": "rv6m_test_img.png"}},
    "3": enc("make the whole picture bright red"), "4": enc(""),
    "6": {"class_type": "VAEEncode", "inputs": {"pixels": ["5", 0], "vae": ["13", 0]}},
    "12": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["1", 0], "shift": 3}},
    "7": {"class_type": "KSampler", "inputs": {"model": ["12", 0], "positive": ["3", 0], "negative": ["4", 0], "latent_image": ["6", 0],
          "seed": 1, "steps": 4, "cfg": 2.5, "sampler_name": "euler", "scheduler": "simple", "denoise": 1}},
    "8": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["13", 0]}},
    "9": {"class_type": "PreviewImage", "inputs": {"images": ["8", 0]}},
}
req = urllib.request.Request("http://127.0.0.1:8188/api/prompt", json.dumps({"prompt": p}).encode(), {"Content-Type": "application/json"})
try:
    pid = json.load(urllib.request.urlopen(req))["prompt_id"]
except urllib.error.HTTPError as e:
    raise SystemExit("ENCODER TEST REJECTED: " + e.read().decode()[:800])
t = time.time()
while True:
    h = get("/api/history/" + pid)
    if pid in h:
        st = h[pid]["status"]
        print("ENCODER TEST:", st["status_str"], f"in {time.time() - t:.0f}s")
        if st["status_str"] != "success": print(json.dumps(st)[:800])
        break
    if time.time() - t > 900: raise SystemExit("ENCODER TEST TIMED OUT")
    time.sleep(2)
PY
echo
echo "Done. Reload the edit page: Describe edit / Change / Create now show a Text encoder switch."
