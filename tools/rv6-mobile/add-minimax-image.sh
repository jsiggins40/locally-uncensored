#!/usr/bin/env bash
# Adds MiniMax H3 Image (MiniMax, Nov 2026): one 20B model that makes pictures from
# text and edits photos by instruction (up to 16 reference photos), 6 steps at CFG 1.
# It runs through SwarmUI, so on the edit page it is in 🐝 Swarm mode (New picture and
# Edit my photo). Needs SwarmUI first (add-swarmui.sh).
# Gets Comfy-Org's ComfyUI files, pinned and checked against their SHA-256:
#   the model, 8-bit (21 GB); its Qwen3-VL 32B text encoder, 4-bit (16 GB, made for
#   Blackwell GPUs like the RTX PRO 6000); its own VAE (1 GB). About 38 GB.
# Then makes a test picture and a test edit on the GPU through SwarmUI.
# LICENCE: the MiniMax-H3-Image Community License only grants rights in its "Applicable
# Territory": worldwide EXCLUDING the European Union, the United Kingdom, the Republic
# of Korea and the United States (it says to ask MiniMax, or use their hosted API,
# there). Check where your VM and you are before installing; this script asks.
#   bash <(curl -fsSL https://raw.githubusercontent.com/jsiggins40/locally-uncensored/rv6-mobile-editor/tools/rv6-mobile/add-minimax-image.sh)
set -euo pipefail
cd "$HOME/ComfyUI"
SRC="https://raw.githubusercontent.com/jsiggins40/locally-uncensored/rv6-mobile-editor/tools/rv6-mobile"
[ -x "$HOME/.local/bin/swarm-start" ] || { echo "MiniMax H3 Image runs through SwarmUI: run add-swarmui.sh first, then this."; exit 1; }

if [ -z "${MINIMAX_LICENCE_OK:-}" ]; then
  echo "LICENCE NOTE: MiniMax's licence for this model covers use worldwide EXCEPT in the"
  echo "European Union, the United Kingdom, South Korea and the United States."
  echo "Full text: https://huggingface.co/MiniMaxAI/MiniMax-H3-Image/blob/main/LICENSE"
  read -rp "Type yes to install anyway, anything else to stop: " ok </dev/tty || ok=""
  [ "$ok" = yes ] || { echo "Stopped; nothing was downloaded."; exit 0; }
fi

echo "== ComfyUI must know MiniMax H3 Image (native support from mid-Nov 2026)"
if ! grep -q "class MiniMaxH3Image" comfy/supported_models.py 2>/dev/null; then
  echo "   updating ComfyUI"
  git pull -q --ff-only && venv/bin/pip install -q -r requirements.txt
fi
grep -q "class MiniMaxH3Image" comfy/supported_models.py || { echo "This ComfyUI can't run MiniMax H3 Image even after updating; stopping."; exit 1; }

venv/bin/python - <<'PY'
import hashlib, os, shutil, sys
from huggingface_hub import hf_hub_download
free = shutil.disk_usage(os.path.realpath("models")).free / 1e9
print(f"== Free disk: {free:.0f} GB")
R, REV = "Comfy-Org/MiniMax-H3-Image", "89c635085f539d03ad35485ed542c5a286e3c6d6"
# (file in the repo, saved as, SHA-256). The names are the ones SwarmUI looks for,
# so it doesn't download its own copies.
FILES = [
    ("diffusion_models/minimax_h3_image_pruned_int8_convrot.safetensors", "models/diffusion_models/minimax_h3_image_pruned_int8_convrot.safetensors",
     "83eb140f1ff95c068f8a6886e6d3f3d0d4ab2e07bb2fe6d2ef992a51146ee622", "MiniMax H3 Image, 8-bit (21 GB)"),
    ("text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors", "models/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
     "35a88d51044231fe332301d7a62aa81e3f2cba62febeb446e2c1e3e0ef76f2c6", "Text encoder: Qwen3-VL 32B, 4-bit (16 GB)"),
    ("vae/minimax_h3_image_vae_fp16.safetensors", "models/vae/MiniMaxH3/minimax_h3_image_vae_fp16.safetensors",
     "d80e9c2e5cfc9844869961ff5a58c9a9e52182df6bfb30b26018dd7f348721af", "VAE (1 GB)"),
]
need = sum(1 for _, dest, _, _ in FILES if not (os.path.exists(dest) and os.path.getsize(dest) > 0))
if need and free < 45:
    sys.exit("Need about 45 GB free. Delete something first (disk-report.sh shows what's there).")
for path, dest, sha, what in FILES:
    print("==", what)
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        print("   already have", dest); continue
    tmp = hf_hub_download(R, path, revision=REV, local_dir="models/_hf_tmp")
    h = hashlib.sha256()
    with open(tmp, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""): h.update(b)
    if h.hexdigest() != sha:
        os.remove(tmp); sys.exit(f"{path} doesn't match its published SHA-256; not installed.")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    shutil.move(tmp, dest)
    print("   installed", dest)
shutil.rmtree("models/_hf_tmp", ignore_errors=True)
PY

echo "== SwarmUI: restart with the new model, and its preset (6 steps, CFG 1)"
bash <(curl -fsSL "$SRC/add-swarmui.sh")

echo "== Tests on the GPU, through SwarmUI (the first load takes a few minutes)"
python3 - <<'PY'
import base64, json, time, urllib.error, urllib.parse, urllib.request
B = "http://127.0.0.1:7801"
def api(name, body, timeout=3600):
    req = urllib.request.Request(f"{B}/API/{name}", json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))
sid = api("GetNewSession", {}, 60)["session_id"]
files = api("ListModels", {"session_id": sid, "path": "", "depth": 8, "subtype": "Stable-Diffusion", "allowRemote": False}, 120).get("files") or []
m = next((f for f in files if "minimax_h3_image" in (f.get("name") or "").lower()), None)
if not m:
    raise SystemExit("MINIMAX TEST: SwarmUI doesn't list the model. Open SwarmUI, Models tab, press its refresh button, and run this again.")
print("SwarmUI sees:", m["name"], "| type:", m.get("class") or "NOT RECOGNISED")
name = m["name"][:-len(".safetensors")] if m["name"].endswith(".safetensors") else m["name"]
def gen(label, extra):
    t = time.time()
    try:
        r = api("GenerateText2Image", {"session_id": sid, "images": 1, "model": name, "width": 1024, "height": 1024, "seed": 1,
                                       "presets": ["Auto MiniMax H3 Image"], "donotsave": True, **extra})
    except Exception as e:
        print(f"{label}: failed ({type(e).__name__}: {str(e)[:200]})"); return None
    if r.get("error") or not r.get("images"):
        print(f"{label}: SwarmUI said: {str(r.get('error') or r)[:400]}"); return None
    print(f"{label}: ok in {time.time() - t:.0f}s")
    return r["images"][0]
img = gen("MINIMAX H3 IMAGE TEST (new picture)", {"prompt": "a red apple on a wooden table, photo"})
if img:
    if not img.startswith("data:"):
        raw = urllib.request.urlopen(B + "/" + urllib.parse.quote(img.lstrip("/")), timeout=120).read()
        img = "data:image/png;base64," + base64.b64encode(raw).decode()
    gen("MINIMAX H3 IMAGE TEST (edit)", {"prompt": "Make the apple green. Keep everything else unchanged.", "promptimages": [img]})
PY

echo
echo "Done. Reload the edit page: 🐝 Swarm → Model → minimax_h3_image…"
echo "  ✍️ New picture, or ✨ Edit my photo (say what to change)."
