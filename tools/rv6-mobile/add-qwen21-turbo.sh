#!/usr/bin/env bash
# Adds Qwen-Image 2.1 Turbo (Oct 2026): the official 2.1 made for 8 steps at
# CFG 1, about 3x faster. Needs Qwen 2.1 itself first (add-qwen21.sh): Turbo
# uses its text encoder and VAE. Gets Comfy-Org's Turbo model (or, if that's all
# there is, their Turbo LoRA for the official model), tests it on the GPU with
# the 8-step schedule it was made for, then updates the page: ⚡ Turbo appears
# next to Noct Q / Official under 🆕 Qwen 2.1.
#   bash <(curl -fsSL https://raw.githubusercontent.com/jsiggins40/locally-uncensored/rv6-mobile-editor/tools/rv6-mobile/add-qwen21-turbo.sh)
set -euo pipefail
cd "$HOME/ComfyUI"
SRC="https://raw.githubusercontent.com/jsiggins40/locally-uncensored/rv6-mobile-editor/tools/rv6-mobile"

restart() {
  if command -v edit-restart >/dev/null; then edit-restart; else
    sudo curl -fsSL "$SRC/edit-restart.sh" -o /usr/local/bin/edit-restart && sudo chmod +x /usr/local/bin/edit-restart && edit-restart
  fi
}

ls models/text_encoders/*qwen3vl_8b* models/vae/*qwen_image_2.1_vae* >/dev/null 2>&1 \
  || { echo "Qwen 2.1 isn't installed yet. Run add-qwen21.sh first, then this."; exit 1; }

echo "== ComfyUI must have the ManualSigmas node (Turbo's 8-step schedule)"
if ! grep -q "ManualSigmas" comfy_extras/*.py 2>/dev/null; then
  echo "   updating ComfyUI"
  git pull -q --ff-only && venv/bin/pip install -q -r requirements.txt
  restart
fi
grep -q "ManualSigmas" comfy_extras/*.py || { echo "This ComfyUI has no ManualSigmas node; stopping."; exit 1; }

venv/bin/python - <<'PY'
import os, shutil, sys
from huggingface_hub import HfApi, hf_hub_download
free = shutil.disk_usage(os.path.realpath("models")).free / 1e9
print(f"== Free disk: {free:.0f} GB")
if free < 50:
    sys.exit("Need about 50 GB free. Delete something first.")
R = "Comfy-Org/Qwen-Image-2.1"
try:
    fl = [(s.rfilename, s.size or 0) for s in HfApi().model_info(R, files_metadata=True).siblings]
except Exception as e:
    sys.exit(f"Can't list {R}: {type(e).__name__}: {str(e)[:200]}")
turbo = [(f, sz) for f, sz in fl if "turbo" in f.lower() and f.endswith(".safetensors")]
for f, sz in turbo: print(f"     {f} ({sz / 1e9:.1f} GB)")
# bf16 first (the H200 has the memory), then fp8, int8 last
rank = lambda f: ("bf16" not in f.lower(), "fp8" not in f.lower(), "int8" in f.lower(), f)
models = sorted([f for f, _ in turbo if "diffusion_models/" in f], key=rank)
loras = sorted([f for f, _ in turbo if "loras/" in f], key=rank)
pick, sub = (models[0], "diffusion_models") if models else (loras[0], "loras") if loras else (None, None)
if not pick:
    sys.exit(f"No Turbo files in {R} yet. Nothing changed.")
name = os.path.basename(pick)
if "qwen" not in name.lower():  # the page finds it by name
    name = "qwen_image_2.1_" + name
dest = os.path.join("models", sub, name)
print("== Qwen-Image 2.1 Turbo" + (" (LoRA for the official model)" if sub == "loras" else ""))
if os.path.exists(dest) and os.path.getsize(dest) > 0:
    print("   already have", dest)
else:
    print(f"   downloading {pick}")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    shutil.move(hf_hub_download(R, pick, local_dir="models/_hf_tmp"), dest)
shutil.rmtree("models/_hf_tmp", ignore_errors=True)
PY

echo "== Test on the GPU (first load takes a minute or two)"
curl -sf -o /dev/null localhost:8188 || restart
curl -fsS -o /dev/null -X POST localhost:8188/api/refresh 2>/dev/null || true
venv/bin/python - <<'PY'
import json, os, shutil, time, urllib.request, urllib.error
from PIL import Image
Image.new("RGB", (512, 512), (120, 120, 120)).save("input/rv6m_test_img.png")
SIGMAS = "1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568, 0.0"
def get(path): return json.load(urllib.request.urlopen("http://127.0.0.1:8188" + path))
def node(n): return get("/api/object_info/" + n).get(n, {}).get("input", {})
def opts(n, inp):
    v = node(n); v = v.get("required", {}).get(inp) or v.get("optional", {}).get(inp)
    return (v[0] if isinstance(v[0], list) else v[1].get("options", [])) if v else []
bf = lambda xs: sorted(xs, key=lambda n: ("bf16" not in n.lower(), n))
units, loras = opts("UNETLoader", "unet_name"), opts("LoraLoaderModelOnly", "lora_name")
t_unet = (bf([n for n in units if "2.1" in n and "turbo" in n.lower() and "noct" not in n.lower()]) or [None])[0]
t_lora = next((n for n in loras if "2.1" in n and "turbo" in n.lower()), None)
base = (bf([n for n in units if "qwen_image_2.1" in n.lower() and not any(w in n.lower() for w in ("noct", "turbo", "uncensor"))]) or [None])[0]
clip = (bf([n for n in opts("CLIPLoader", "clip_name") if "qwen3vl_8b" in n.lower() and "heretic" not in n.lower()]) or [None])[0]
vae = next((n for n in opts("VAELoader", "vae_name") if "qwen_image_2.1_vae" in n.lower()), None)
sig_in = next(iter(node("ManualSigmas").get("required", {})), None)
print("ComfyUI sees:", t_unet or t_lora, "|", clip, "|", vae, "| ManualSigmas input:", sig_in)
if not ((t_unet or (t_lora and base)) and clip and vae and sig_in):
    raise SystemExit("TURBO TEST: files or nodes not visible to ComfyUI")

def run(label, edit):
    model = ["1", 0]
    p = {"1": {"class_type": "UNETLoader", "inputs": {"unet_name": t_unet or base, "weight_dtype": "default"}},
         "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": clip, "type": "qwen_image"}},
         "13": {"class_type": "VAELoader", "inputs": {"vae_name": vae}},
         "3": {"class_type": "TextEncodeQwenImage21", "inputs": {"clip": ["2", 0], "prompt": "make the whole picture bright red" if edit else "a red apple on a table",
               "negative_prompt": "", "vae": ["13", 0], "resolution": 0 if edit else 1024}},
         "8": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["13", 0]}},
         "9": {"class_type": "SaveImage", "inputs": {"images": ["8", 0], "filename_prefix": "rv6m_turbo_test"}}}
    if not t_unet:
        p["18"] = {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["1", 0], "lora_name": t_lora, "strength_model": 1}}
        model = ["18", 0]
    p.update({"70": {"class_type": "RandomNoise", "inputs": {"noise_seed": 1}},
              "71": {"class_type": "CFGGuider", "inputs": {"model": model, "positive": ["3", 0], "negative": ["3", 1], "cfg": 1}},
              "72": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
              "73": {"class_type": "ManualSigmas", "inputs": {sig_in: SIGMAS}},
              "7": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["70", 0], "guider": ["71", 0], "sampler": ["72", 0],
                    "sigmas": ["73", 0], "latent_image": ["3", 2] if edit else ["6", 0]}}})
    if edit:
        p["5"] = {"class_type": "LoadImage", "inputs": {"image": "rv6m_test_img.png"}}
        p["3"]["inputs"]["images.image_1"] = ["5", 0]
    else:
        p["6"] = {"class_type": "EmptyLatentImage", "inputs": {"width": 1024, "height": 1024, "batch_size": 1}}
    req = urllib.request.Request("http://127.0.0.1:8188/api/prompt", json.dumps({"prompt": p}).encode(), {"Content-Type": "application/json"})
    try:
        pid = json.load(urllib.request.urlopen(req))["prompt_id"]
    except urllib.error.HTTPError as e:
        print(label, "REJECTED:", e.read().decode()[:600]); return False
    t = time.time()
    while time.time() - t < 900:
        h = get("/api/history/" + pid)
        if pid in h:
            st = h[pid]["status"]
            print(label + ":", st["status_str"], f"in {time.time() - t:.0f}s")
            if st["status_str"] != "success":
                print("  ", json.dumps(st)[:600]); return False
            imgs = [i for o in h[pid]["outputs"].values() for i in o.get("images", [])]
            if imgs:
                im = Image.open(os.path.join("output", imgs[0].get("subfolder", ""), imgs[0]["filename"])).convert("RGB")
                r, g, b = [sum(c) / len(c) for c in zip(*im.resize((32, 32)).getdata())]
                print(f"   average colour R{r:.0f} G{g:.0f} B{b:.0f}" + ("  (red, as asked)" if r > g + 30 and r > b + 30 else ""))
                if max(r, g, b) < 8 or min(r, g, b) > 247:
                    print("   the picture came out blank"); return False
            return True
        time.sleep(2)
    print(label, "TIMED OUT"); return False

ok = run("TURBO CREATE TEST", False) and run("TURBO EDIT TEST", True)
if not ok:
    bad = ("diffusion_models", t_unet) if t_unet else ("loras", t_lora)
    os.makedirs("models/_disabled", exist_ok=True)
    shutil.move(os.path.join("models", *bad), os.path.join("models/_disabled", os.path.basename(bad[1])))
    raise SystemExit(f"Turbo failed its test; moved {bad[1]} to models/_disabled so the page won't offer it.")
print("Open ComfyUI's output folder to look at rv6m_turbo_test_* if you want to see the test pictures.")
PY

echo "== Updating the page"
restart
echo
echo "Done. Reload the edit page: under 🆕 Qwen 2.1 there's now ⚡ Turbo (8 steps) next to Noct Q / Official."
