#!/usr/bin/env bash
# Adds Qwen-Image 2.1 (Sept 2026, 7B): text-to-image and editing with up to 10
# photos in one model, far less censored than 2512. Also fetches the Noct Q
# uncensored fine-tune and the Heretic (abliterated) Qwen3-VL text encoder when
# they are in a form ComfyUI can load. About 50 GB. Each part is tested on the
# GPU; a part that fails its test is moved aside so the page never offers it.
#   bash <(curl -fsSL https://raw.githubusercontent.com/jsiggins40/locally-uncensored/rv6-mobile-editor/tools/rv6-mobile/add-qwen21.sh)
set -euo pipefail
cd "$HOME/ComfyUI"
SRC="https://raw.githubusercontent.com/jsiggins40/locally-uncensored/rv6-mobile-editor/tools/rv6-mobile"

restart() {
  if command -v edit-restart >/dev/null; then edit-restart; else
    sudo curl -fsSL "$SRC/edit-restart.sh" -o /usr/local/bin/edit-restart && sudo chmod +x /usr/local/bin/edit-restart && edit-restart
  fi
}

echo "== ComfyUI must know Qwen-Image 2.1 (native since 20 Sept 2026)"
if ! grep -q "TextEncodeQwenImage21" comfy_extras/nodes_qwen.py 2>/dev/null; then
  echo "   updating ComfyUI"
  git pull -q --ff-only && venv/bin/pip install -q -r requirements.txt
  restart
fi
grep -q "TextEncodeQwenImage21" comfy_extras/nodes_qwen.py || { echo "ComfyUI still has no Qwen-Image 2.1 support; stopping."; exit 1; }

venv/bin/python - <<'PY'
import os, re, shutil, sys
from huggingface_hub import HfApi, hf_hub_download
api = HfApi()
free = shutil.disk_usage(os.path.realpath("models")).free / 1e9
print(f"== Free disk: {free:.0f} GB")
if free < 60:
    sys.exit("Need about 60 GB free. Delete something first.")

def files(repo):
    try:
        return [(s.rfilename, s.size or 0) for s in api.model_info(repo, files_metadata=True).siblings]
    except Exception as e:
        print(f"   {repo}: can't list it ({type(e).__name__}: {str(e)[:150]})")
        if "401" in str(e) or "403" in str(e) or "gated" in str(e).lower():
            print("   It may need you to accept its terms on huggingface.co and log in here:")
            print("   ~/ComfyUI/venv/bin/huggingface-cli login   (then run this again)")
        return []

def fetch(repo, path, subdir, name=None):
    name = name or os.path.basename(path)
    dest = os.path.join("models", subdir, name)
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        print("   already have", dest); return name
    print(f"   downloading {path} from {repo}")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    shutil.move(hf_hub_download(repo, path, local_dir="models/_hf_tmp"), dest)
    return name

def show(repo, fl):
    for f, sz in fl:
        if f.endswith((".safetensors", ".gguf")): print(f"     {f} ({sz / 1e9:.1f} GB)")

# 1. Official files, as in ComfyUI's own 2.1 templates; bf16 when there is one
R = "Comfy-Org/Qwen-Image-2.1"
fl = files(R); show(R, fl)
sts = [f for f, _ in fl if f.endswith(".safetensors")]
bf16_first = lambda xs: sorted(xs, key=lambda f: ("bf16" not in f.lower(), "int8" in f.lower(), f))
dm = bf16_first([f for f in sts if "diffusion_models/" in f and "qwen_image_2.1" in f.lower()])
te = bf16_first([f for f in sts if "text_encoders/" in f and "qwen3vl_8b" in f.lower()])
va = [f for f in sts if "vae/" in f and "qwen_image_2.1_vae" in f.lower()]
if not (dm and te and va):
    sys.exit("Could not find the Qwen-Image 2.1 model, text encoder and VAE in " + R)
print("== Qwen-Image 2.1")
fetch(R, dm[0], "diffusion_models"); fetch(R, te[0], "text_encoders"); fetch(R, va[0], "vae")

def single_file(repo, label, want_gb):
    fl = files(repo)
    if not fl: return None
    show(repo, fl)
    st = [(f, s) for f, s in fl if f.endswith(".safetensors") and s > want_gb * 1e9]
    if any(re.search(r"-0000\d-of-", f) for f, _ in st):
        print(f"   {label}: only in split (transformers) form, which ComfyUI can't load; skipped"); return None
    if not st:
        print(f"   {label}: no single ComfyUI file found; skipped"); return None
    ver = lambda f: [int(x) for x in re.findall(r"v(\d+)", f.lower())] or [0]
    return max(st, key=lambda x: (ver(x[0]), "bf16" in x[0].lower(), x[1]))[0]

# 2. Noct Q: uncensored fine-tune of 2.1 (photoreal line), newest version
print("== Noct Q uncensored")
f = single_file("Noctaluna/Noct-Q-Uncensored-Qwen-Image-2.1", "Noct Q", 3)
if f:
    name = os.path.basename(f)
    fetch("Noctaluna/Noct-Q-Uncensored-Qwen-Image-2.1", f, "diffusion_models",
          name if "noct" in name.lower() else "noct_q_" + name)

# 3. Heretic text encoder: Qwen3-VL 8B with refusals removed
print("== Heretic (abliterated) text encoder")
f = single_file("kkxao/Qwen-Image-2.1-Text-Encoder-Heretic", "Heretic encoder", 5)
if f:
    name = os.path.basename(f)
    fetch("kkxao/Qwen-Image-2.1-Text-Encoder-Heretic", f, "text_encoders",
          name if "heretic" in name.lower() and "qwen3vl" in name.lower() else "qwen3vl_8b_heretic_" + name)
shutil.rmtree("models/_hf_tmp", ignore_errors=True)
PY

echo "== Tests on the GPU (each first load takes a minute or two)"
curl -sf -o /dev/null localhost:8188 || restart  # ComfyUI isn't running yet (e.g. right after the VM started)
curl -fsS -o /dev/null -X POST localhost:8188/api/refresh 2>/dev/null || true
venv/bin/python - <<'PY'
import json, os, shutil, time, urllib.request, urllib.error
from PIL import Image
Image.new("RGB", (512, 512), (120, 120, 120)).save("input/rv6m_test_img.png")
def get(path): return json.load(urllib.request.urlopen("http://127.0.0.1:8188" + path))
def opts(node, inp):
    v = get("/api/object_info/" + node).get(node, {}).get("input", {})
    v = v.get("required", {}).get(inp) or v.get("optional", {}).get(inp)
    return (v[0] if isinstance(v[0], list) else v[1].get("options", [])) if v else []
units, clips = opts("UNETLoader", "unet_name"), opts("CLIPLoader", "clip_name")
bf = lambda xs: sorted(xs, key=lambda n: ("bf16" not in n.lower(), n))
unet = (bf([n for n in units if "qwen_image_2.1" in n.lower() and "noct" not in n.lower()]) or [None])[0]
noct = next((n for n in units if "noct" in n.lower()), None)
clip = (bf([n for n in clips if "qwen3vl_8b" in n.lower() and "heretic" not in n.lower()]) or [None])[0]
heretic = next((n for n in clips if "heretic" in n.lower()), None)
vae = next((n for n in opts("VAELoader", "vae_name") if "qwen_image_2.1_vae" in n.lower()), None)
print("ComfyUI sees:", unet, "|", noct, "|", clip, "|", heretic, "|", vae)
if not (unet and clip and vae): raise SystemExit("QWEN 2.1 TEST: files not visible to ComfyUI")

def run(label, u, c, edit):
    p = {"1": {"class_type": "UNETLoader", "inputs": {"unet_name": u, "weight_dtype": "default"}},
         "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": c, "type": "qwen_image"}},
         "13": {"class_type": "VAELoader", "inputs": {"vae_name": vae}},
         "3": {"class_type": "TextEncodeQwenImage21", "inputs": {"clip": ["2", 0], "prompt": "make the whole picture bright red" if edit else "a red apple on a table",
               "negative_prompt": "", "vae": ["13", 0], "resolution": 0 if edit else 1024}},
         "7": {"class_type": "KSampler", "inputs": {"model": ["1", 0], "positive": ["3", 0], "negative": ["3", 1], "latent_image": ["3", 2] if edit else ["6", 0],
               "seed": 1, "steps": 6, "cfg": 1, "sampler_name": "euler", "scheduler": "simple", "denoise": 1}},
         "8": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["13", 0]}},
         "9": {"class_type": "PreviewImage", "inputs": {"images": ["8", 0]}}}
    if edit:
        p["5"] = {"class_type": "LoadImage", "inputs": {"image": "rv6m_test_img.png"}}
        p["3"]["inputs"]["images.image_1"] = ["5", 0]
    else:
        p["6"] = {"class_type": "EmptyLatentImage", "inputs": {"width": 512, "height": 512, "batch_size": 1}}
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
            if st["status_str"] != "success": print("  ", json.dumps(st)[:600])
            return st["status_str"] == "success"
        time.sleep(2)
    print(label, "TIMED OUT"); return False

def disable(sub, name):
    os.makedirs("models/_disabled", exist_ok=True)
    shutil.move(os.path.join("models", sub, name), os.path.join("models/_disabled", name))
    print(f"   moved {name} to models/_disabled (it didn't work), so the page won't offer it")

ok = run("QWEN 2.1 CREATE TEST", unet, clip, False)
ok = run("QWEN 2.1 EDIT TEST", unet, clip, True) and ok
if noct and not run("NOCT Q TEST", noct, clip, False): disable("diffusion_models", noct)
if heretic and not run("HERETIC ENCODER TEST", unet, heretic, False): disable("text_encoders", heretic)
if not ok: raise SystemExit("Qwen 2.1 itself failed its test; see above.")
PY

echo "== Updating the page"
restart
echo
echo "Done. Reload the edit page: ✍️ Create and ✨ Describe edit now offer 🆕 Qwen 2.1."
