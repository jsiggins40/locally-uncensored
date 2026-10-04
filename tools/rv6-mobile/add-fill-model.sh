#!/usr/bin/env bash
# Adds FLUX.1 Fill dev OneReward (ByteDance's RL-tuned inpainting model) for
# the page's "🩹 Fix" engine in Selected area: paint a spot and it repairs or
# removes it with natural edges and colours, with or without a prompt.
# Downloads about 35 GB: the model (bf16 on GPUs with room for it), the two
# Flux text encoders and the Flux VAE, then runs one test fill on the GPU.
#   bash <(curl -fsSL https://raw.githubusercontent.com/jsiggins40/locally-uncensored/rv6-mobile-editor/tools/rv6-mobile/add-fill-model.sh)
set -euo pipefail
cd "$HOME/ComfyUI"

venv/bin/python - <<'PY'
import os, shutil, sys
from huggingface_hub import HfApi, hf_hub_download

api = HfApi()
free_gb = shutil.disk_usage(os.path.realpath("models")).free / 1e9  # models may live on another disk (symlink)
print(f"== Free disk: {free_gb:.0f} GB")
if free_gb < 40:
    sys.exit("Need about 40 GB free for the Fix model. Delete something first.")

def fetch(repo, path, subdir, name=None):
    dest = os.path.join("models", subdir, name or os.path.basename(path))
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        print("   already have", dest); return
    print(f"   downloading {os.path.basename(path)} from {repo}")
    tmp = hf_hub_download(repo, path, local_dir="models/_hf_tmp")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    shutil.move(tmp, dest)

print("== FLUX.1 Fill dev OneReward")
# A ComfyUI-ready single file of bytedance-research/OneReward. Plain OneReward
# (not "Dynamic"), full precision when there is one: the A100 has the memory.
repo = "yichengup/flux.1-fill-dev-OneReward"
files = [s for s in api.model_info(repo, files_metadata=True).siblings
         if s.rfilename.endswith(".safetensors") and "lora" not in s.rfilename.lower()]
if not files:
    sys.exit(f"No model file found in {repo}")
for s in files: print(f"   found {s.rfilename} ({(s.size or 0) / 1e9:.1f} GB)")
best = max(files, key=lambda s: ("dynamic" not in s.rfilename.lower(), "fp8" not in s.rfilename.lower(), s.size or 0))
# Saved under a fixed name so the page and the test always recognise it
fetch(repo, best.rfilename, "diffusion_models",
      "flux1-fill-dev-OneReward" + ("-fp8" if "fp8" in best.rfilename.lower() else "") + ".safetensors")

print("== Flux text encoders and VAE")
fetch("comfyanonymous/flux_text_encoders", "clip_l.safetensors", "text_encoders")
fetch("comfyanonymous/flux_text_encoders", "t5xxl_fp16.safetensors", "text_encoders")
# The Flux VAE: FLUX.1-schnell ships the same one as dev without dev's gate;
# Comfy-Org's Lumina repack carries the identical file as a fallback
for repo, path in (("black-forest-labs/FLUX.1-schnell", "ae.safetensors"),
                   ("Comfy-Org/Lumina_Image_2.0_Repackaged", "split_files/vae/ae.safetensors")):
    try:
        fetch(repo, path, "vae"); break
    except Exception as e:
        print(f"   {repo}: {type(e).__name__}, trying the next source")
else:
    sys.exit("Could not download the Flux VAE (ae.safetensors)")
shutil.rmtree("models/_hf_tmp", ignore_errors=True)
PY

echo "== Test fill on the GPU (first load takes a minute)"
curl -fsS -o /dev/null -X POST localhost:8188/api/refresh 2>/dev/null || true
venv/bin/python - <<'PY'
import json, time, urllib.request, urllib.error
from PIL import Image, ImageDraw
Image.new("RGB", (512, 512), (120, 120, 120)).save("input/rv6m_test_img.png")
m = Image.new("RGB", (512, 512), "black"); ImageDraw.Draw(m).rectangle([160, 160, 352, 352], fill="white")
m.save("input/rv6m_test_mask.png")
def get(path):
    return json.load(urllib.request.urlopen("http://127.0.0.1:8188" + path))
def opts(node, inp):
    v = get("/api/object_info/" + node).get(node, {}).get("input", {})
    v = v.get("required", {}).get(inp) or v.get("optional", {}).get(inp)
    return (v[0] if isinstance(v[0], list) else v[1].get("options", [])) if v else []
pick = lambda xs, f: (sorted(x for x in xs if f(x.lower())) or [None])[-1]
unet = pick(opts("UNETLoader", "unet_name"), lambda n: "fill" in n and "flux" in n)
t5 = pick(opts("DualCLIPLoader", "clip_name1"), lambda n: "t5xxl" in n)
cl = pick(opts("DualCLIPLoader", "clip_name2"), lambda n: n.startswith("clip_l"))
vae = pick(opts("VAELoader", "vae_name"), lambda n: n == "ae.safetensors")
print("ComfyUI sees:", unet, "|", t5, "|", cl, "|", vae)
if not (unet and t5 and cl and vae):
    raise SystemExit("FILL TEST: files not visible to ComfyUI")
p = {
    "1": {"class_type": "UNETLoader", "inputs": {"unet_name": unet, "weight_dtype": "default"}},
    "2": {"class_type": "DualCLIPLoader", "inputs": {"clip_name1": t5, "clip_name2": cl, "type": "flux"}},
    "3": {"class_type": "VAELoader", "inputs": {"vae_name": vae}},
    "5": {"class_type": "LoadImage", "inputs": {"image": "rv6m_test_img.png"}},
    "6": {"class_type": "LoadImageMask", "inputs": {"image": "rv6m_test_mask.png", "channel": "red"}},
    "7": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": "a red apple"}},
    "8": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": ""}},
    "9": {"class_type": "FluxGuidance", "inputs": {"conditioning": ["7", 0], "guidance": 30}},
    "10": {"class_type": "InpaintModelConditioning", "inputs": {"positive": ["9", 0], "negative": ["8", 0], "vae": ["3", 0],
           "pixels": ["5", 0], "mask": ["6", 0], "noise_mask": True}},
    "11": {"class_type": "DifferentialDiffusion", "inputs": {"model": ["1", 0]}},
    "12": {"class_type": "KSampler", "inputs": {"model": ["11", 0], "positive": ["10", 0], "negative": ["10", 1], "latent_image": ["10", 2],
           "seed": 1, "steps": 20, "cfg": 1, "sampler_name": "euler", "scheduler": "normal", "denoise": 1}},
    "13": {"class_type": "VAEDecode", "inputs": {"samples": ["12", 0], "vae": ["3", 0]}},
    "14": {"class_type": "SaveImage", "inputs": {"filename_prefix": "rv6_fill_test", "images": ["13", 0]}},
}
req = urllib.request.Request("http://127.0.0.1:8188/api/prompt", json.dumps({"prompt": p}).encode(), {"Content-Type": "application/json"})
try:
    pid = json.load(urllib.request.urlopen(req))["prompt_id"]
except urllib.error.HTTPError as e:
    raise SystemExit("FILL TEST REJECTED: " + e.read().decode()[:800])
t = time.time()
while True:
    h = get("/api/history/" + pid)
    if pid in h:
        st = h[pid]["status"]
        print("FILL TEST:", st["status_str"], f"in {time.time() - t:.0f}s")
        if st["status_str"] != "success": print(json.dumps(st)[:800])
        break
    if time.time() - t > 900: raise SystemExit("FILL TEST TIMED OUT")
    time.sleep(2)
PY
echo
echo "Done. Reload the edit page: Selected area now has a 🩹 Fix engine."
