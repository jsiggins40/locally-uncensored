#!/usr/bin/env bash
# Adds Krea 2 photo editing to the edit page (Describe edit → Edit model →
# Krea 2). Needs Krea 2 Turbo first (add-krea2.sh). Installs:
#   - the comfyui-krea2edit nodes (lbouaraba), pinned to the reviewed version:
#     they give Krea 2 the photo twice, as picture tokens (appearance) and to its
#     Qwen3-VL text encoder (what's where), the way the edit LoRAs were trained
#   - Krea 2 General Edit V2 (instruction edits) and Identity Edit V2 (restage
#     the person in the photo) from conradlocke/krea2-identity-edit
# Turbo, 8 steps, CFG 1, one photo. Tests an edit on the GPU, then updates the page.
# The LoRAs are under the Krea 2 Community License (derivative models of Krea 2).
#   bash <(curl -fsSL https://raw.githubusercontent.com/jsiggins40/locally-uncensored/rv6-mobile-editor/tools/rv6-mobile/add-krea2-edit.sh)
set -euo pipefail
cd "$HOME/ComfyUI"
SRC="https://raw.githubusercontent.com/jsiggins40/locally-uncensored/rv6-mobile-editor/tools/rv6-mobile"
NODES_REPO=https://github.com/lbouaraba/comfyui-krea2edit.git
NODES_COMMIT=13ef60f   # v2.0.0, 2026-10-16: the version that was read through before adding it here

restart() {
  if command -v edit-restart >/dev/null; then edit-restart; else
    sudo curl -fsSL "$SRC/edit-restart.sh" -o /usr/local/bin/edit-restart && sudo chmod +x /usr/local/bin/edit-restart && edit-restart
  fi
}

ls models/diffusion_models/*[Kk]rea*[Tt]urbo* >/dev/null 2>&1 \
  || { echo "Krea 2 Turbo isn't installed yet. Run add-krea2.sh first, then this."; exit 1; }

echo "== Krea 2 edit nodes (comfyui-krea2edit, pinned to $NODES_COMMIT)"
N=custom_nodes/comfyui-krea2edit
if [ ! -d "$N/.git" ]; then git clone -q "$NODES_REPO" "$N"; fi
git -C "$N" fetch -q origin 2>/dev/null || true
git -C "$N" checkout -q "$NODES_COMMIT" 2>/dev/null \
  || { git -C "$N" fetch -q --unshallow origin 2>/dev/null || true; git -C "$N" checkout -q "$NODES_COMMIT"; }
grep -q "Krea2EditGroundedEncode" "$N/__init__.py" || { echo "That node pack has no Krea2EditGroundedEncode node; stopping."; exit 1; }

venv/bin/python - <<'PY'
import os, re, shutil, sys
from huggingface_hub import HfApi, hf_hub_download
R = "conradlocke/krea2-identity-edit"
try:
    files = [f for f in HfApi().list_repo_files(R) if f.endswith(".safetensors")]
except Exception as e:
    sys.exit(f"Can't list {R}: {type(e).__name__}: {str(e)[:200]}")
for f in files: print("    ", f)
def newest(kind):
    c = [f for f in files if kind in os.path.basename(f).lower()]
    ver = lambda f: [int(x) for x in re.findall(r"\d+", os.path.basename(f))] or [0]
    return sorted(c, key=ver)[-1] if c else None
got = 0
for kind, label in (("general_edit", "General Edit"), ("identity_edit", "Identity Edit")):
    f = newest(kind)
    if not f:
        print(f"   no {label} LoRA in {R}"); continue
    name = os.path.basename(f)
    if "krea" not in name.lower():  # the page finds it by name
        name = "krea2_" + name
    dest = os.path.join("models/loras", name)
    print(f"== Krea 2 {label} LoRA")
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        print("   already have", dest)
    else:
        print(f"   downloading {f}")
        os.makedirs("models/loras", exist_ok=True)
        shutil.move(hf_hub_download(R, f, local_dir="models/_hf_tmp"), dest)
    got += 1
shutil.rmtree("models/_hf_tmp", ignore_errors=True)
if not got:
    sys.exit("No edit LoRAs found. Nothing changed on the page.")
PY

echo "== Restarting ComfyUI so it loads the new nodes"
restart

echo "== Test on the GPU (first load takes a minute or two)"
venv/bin/python - <<'PY'
import json, os, re, shutil, time, urllib.request, urllib.error
from PIL import Image
im = Image.new("RGB", (768, 768), (200, 200, 200))
for x in range(268, 500):
    for y in range(268, 500): im.putpixel((x, y), (30, 60, 200))  # a blue square
im.save("input/rv6m_krea_edit_test.png")
def get(path): return json.load(urllib.request.urlopen("http://127.0.0.1:8188" + path))
def opts(n, inp):
    v = get("/api/object_info/" + n).get(n, {}).get("input", {})
    v = v.get("required", {}).get(inp) or v.get("optional", {}).get(inp)
    return (v[0] if isinstance(v[0], list) else v[1].get("options", [])) if v else []
bf = lambda xs: sorted(xs, key=lambda n: ("bf16" not in n.lower(), n))
unet = (bf([n for n in opts("UNETLoader", "unet_name") if "krea" in n.lower() and "turbo" in n.lower()]) or [None])[0]
clip = (bf([n for n in opts("CLIPLoader", "clip_name") if re.search(r"krea|qwen3.?vl.?4b", n, re.I)]) or [None])[0]
vaes = opts("VAELoader", "vae_name")
vae = next((n for n in vaes if "krea" in n.lower()), None) or next((n for n in vaes if "qwen_image_vae" in n.lower()), None)
lora = sorted([n for n in opts("LoraLoaderModelOnly", "lora_name") if re.search(r"krea2.*general.?edit", n, re.I)])
lora = lora[-1] if lora else (sorted([n for n in opts("LoraLoaderModelOnly", "lora_name") if re.search(r"krea2.*identity.?edit", n, re.I)]) or [None])[-1]
nodes = "Krea2EditModelPatch" in get("/api/object_info/Krea2EditModelPatch")
print("ComfyUI sees:", unet, "|", clip, "|", vae, "|", lora, "| edit nodes:", nodes)
if not (unet and clip and vae and lora and nodes):
    raise SystemExit("KREA 2 EDIT TEST: files or nodes not visible to ComfyUI")
p = {"1": {"class_type": "UNETLoader", "inputs": {"unet_name": unet, "weight_dtype": "default"}},
     "18": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["1", 0], "lora_name": lora, "strength_model": 1}},
     "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": clip, "type": "krea2"}},
     "13": {"class_type": "VAELoader", "inputs": {"vae_name": vae}},
     "5": {"class_type": "LoadImage", "inputs": {"image": "rv6m_krea_edit_test.png"}},
     "6": {"class_type": "EmptyLatentImage", "inputs": {"width": 768, "height": 768, "batch_size": 1}},
     "10": {"class_type": "VAEEncode", "inputs": {"pixels": ["5", 0], "vae": ["13", 0]}},
     "20": {"class_type": "Krea2EditModelPatch", "inputs": {"model": ["18", 0], "source_latent": ["10", 0], "vae": ["13", 0],
            "source_image": ["5", 0], "target_latent": ["6", 0]}},
     "3": {"class_type": "Krea2EditGroundedEncode", "inputs": {"clip": ["2", 0], "prompt": "make the blue square bright red", "image": ["5", 0]}},
     "4": {"class_type": "Krea2EditGroundedEncode", "inputs": {"clip": ["2", 0], "prompt": "", "image": ["5", 0]}},
     "7": {"class_type": "KSampler", "inputs": {"model": ["20", 0], "positive": ["3", 0], "negative": ["4", 0], "latent_image": ["6", 0],
           "seed": 1, "steps": 8, "cfg": 1, "sampler_name": "euler", "scheduler": "simple", "denoise": 1}},
     "8": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["13", 0]}},
     "9": {"class_type": "SaveImage", "inputs": {"images": ["8", 0], "filename_prefix": "rv6m_krea_edit_test"}}}
ok = False
req = urllib.request.Request("http://127.0.0.1:8188/api/prompt", json.dumps({"prompt": p}).encode(), {"Content-Type": "application/json"})
try:
    pid = json.load(urllib.request.urlopen(req))["prompt_id"]
except urllib.error.HTTPError as e:
    print("KREA 2 EDIT TEST REJECTED:", e.read().decode()[:600]); pid = None
t = time.time()
while pid and time.time() - t < 900:
    h = get("/api/history/" + pid)
    if pid in h:
        s = h[pid]["status"]
        print("KREA 2 EDIT TEST:", s["status_str"], f"in {time.time() - t:.0f}s")
        if s["status_str"] != "success":
            print("  ", json.dumps(s)[:600]); break
        imgs = [i for o in h[pid]["outputs"].values() for i in o.get("images", [])]
        if imgs:
            out = Image.open(os.path.join("output", imgs[0].get("subfolder", ""), imgs[0]["filename"])).convert("RGB").resize((768, 768))
            sq = out.crop((300, 300, 470, 470)).resize((16, 16)); r, g, b = [sum(c) / len(c) for c in zip(*sq.getdata())]
            bg = out.crop((20, 20, 180, 180)).resize((16, 16)); br = [sum(c) / len(c) for c in zip(*bg.getdata())]
            print(f"   square now R{r:.0f} G{g:.0f} B{b:.0f} (was blue); background R{br[0]:.0f} G{br[1]:.0f} B{br[2]:.0f} (was light grey)")
            ok = max(r, g, b) > 8 and min(br) > 60  # a picture came back and the background survived
            print("   the square turned red, as asked" if r > b + 30 else "   (the square didn't turn clearly red; the edit model ran but check its results on the page)")
            if not ok: print("   the picture came out wrong")
        break
    time.sleep(2)
if not ok:
    os.makedirs("models/_disabled", exist_ok=True)
    for n in os.listdir("models/loras"):
        if re.search(r"krea2.*(general|identity).?edit", n, re.I):
            shutil.move(os.path.join("models/loras", n), os.path.join("models/_disabled", n))
    raise SystemExit("Krea 2 editing failed its test; moved the edit LoRAs to models/_disabled so the page won't offer it.")
print("The test picture is in ComfyUI's output folder (rv6m_krea_edit_test_*).")
PY

echo "== Updating the page"
restart
echo
echo "Done. Reload the edit page: Describe edit → Edit model → Krea 2."
