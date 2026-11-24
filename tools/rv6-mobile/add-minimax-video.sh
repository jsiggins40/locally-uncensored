#!/usr/bin/env bash
# Adds MiniMax H3 (MiniMax, Nov 2026), the video model: text to video and photo to
# video (your photo as the first frame), with sound, up to about 15 s at 24 fps. It
# runs through SwarmUI (add-swarmui.sh first). On the edit page it is 🎬 Video →
# 🐝 MiniMax H3; SwarmUI's own page (https://<vm>.ts.net:8444) has every setting.
# Gets Comfy-Org's ComfyUI files, pinned and checked against their SHA-256:
#   the model "fl2va" (text / first-last frame to video with audio), pruned 8-bit (21 GB),
#   the one ComfyUI's own templates use; its video and audio VAEs (3.4 GB); the 8-step
#   Turbo LoRA (2 GB); and the Qwen3-VL 32B text encoder (16 GB) that MiniMax H3 Image
#   also uses, if it isn't there yet. About 42 GB.
#   "ref"   also gets "ref2va" (video from reference photos of people and things, 21 GB)
#   "full"  gets the full 8-bit models instead of the pruned ones (34 GB each)
# Then makes a short test video on the GPU through SwarmUI.
# LICENCE: the MiniMax-H3 Community License only grants rights in its "Applicable
# Territory": worldwide EXCLUDING the European Union, the United Kingdom, the Republic
# of Korea and the United States. Check where your VM and you are; this script asks.
#   bash <(curl -fsSL https://raw.githubusercontent.com/jsiggins40/locally-uncensored/rv6-mobile-editor/tools/rv6-mobile/add-minimax-video.sh)
set -euo pipefail
cd "$HOME/ComfyUI"
SRC="https://raw.githubusercontent.com/jsiggins40/locally-uncensored/rv6-mobile-editor/tools/rv6-mobile"
[ -x "$HOME/.local/bin/swarm-start" ] || { echo "MiniMax H3 runs through SwarmUI: run add-swarmui.sh first, then this."; exit 1; }

if [ -z "${MINIMAX_LICENCE_OK:-}" ]; then
  echo "LICENCE NOTE: MiniMax's licence for this model covers use worldwide EXCEPT in the"
  echo "European Union, the United Kingdom, South Korea and the United States."
  echo "Full text: https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE"
  read -rp "Type yes to install anyway, anything else to stop: " ok </dev/tty || ok=""
  [ "$ok" = yes ] || { echo "Stopped; nothing was downloaded."; exit 0; }
fi

echo "== ComfyUI must know MiniMax H3 video (native support from mid-Nov 2026)"
has_node() { grep -rqs "MiniMaxH3ImageToVideo" comfy_extras comfy_api_nodes nodes.py 2>/dev/null; }
if ! has_node; then
  echo "   updating ComfyUI"
  git pull -q --ff-only && venv/bin/pip install -q -r requirements.txt
fi
has_node || { echo "This ComfyUI can't run MiniMax H3 video even after updating; stopping."; exit 1; }

WANT_REF=""; WANT_FULL=""
for a in "$@"; do case "$a" in ref) WANT_REF=1 ;; full) WANT_FULL=1 ;; esac; done
WANT_REF="$WANT_REF" WANT_FULL="$WANT_FULL" venv/bin/python - <<'PY'
import hashlib, os, shutil, sys
from huggingface_hub import hf_hub_download
free = shutil.disk_usage(os.path.realpath("models")).free / 1e9
print(f"== Free disk: {free:.0f} GB")
R, REV = "Comfy-Org/MiniMax-H3", "4b53a35191437de05370da548e59aeabe2a5a09a"
full, ref = bool(os.environ.get("WANT_FULL")), bool(os.environ.get("WANT_REF"))
MODELS = {   # (file, SHA-256, GB)
    ("fl2va", False): ("minimax_h3_fl2va_pruned_int8_convrot.safetensors", "e889202c41dafb67b10d67b97f0d8541508036a6090af23425a5c2615d03c47a", 21),
    ("fl2va", True): ("minimax_h3_fl2va_int8_convrot.safetensors", "7ad4c73e6e378b822ffd1629f27f632d3787d95f5e468e3af958f98c58df96a5", 34),
    ("ref2va", False): ("minimax_h3_ref2va_pruned_int8_convrot.safetensors", "9255f52b6677845ad238f20dfaafa94727053694127ab7f255c048f0f9365779", 21),
    ("ref2va", True): ("minimax_h3_ref2va_int8_convrot.safetensors", "9eef934046a0671bc8a5daf87100705e1478419c574cfde70c50fbe6885f76a9", 34),
}
# (file in the repo, saved as, SHA-256, what it is). The VAE names and folder are the
# ones SwarmUI looks for, so it doesn't download its own copies.
FILES = []
for kind in ["fl2va"] + (["ref2va"] if ref else []):
    name, sha, gb = MODELS[(kind, full)]
    FILES.append((f"diffusion_models/{name}", f"models/diffusion_models/{name}", sha,
                  f"MiniMax H3 {kind}, {'full' if full else 'pruned'} 8-bit ({gb} GB)"))
FILES += [
    ("vae/minimax_h3_video_vae_int8_convrot.safetensors", "models/vae/MiniMaxH3/minimax_h3_video_vae_int8_convrot.safetensors",
     "52a2c8c73583c86e4f41cdcce3a6ad0ea562987bc0bf3d60a0cef5f5c8e60c0e", "Video VAE (2.8 GB)"),
    ("vae/minimax_h3_audio_vae_fp32.safetensors", "models/vae/MiniMaxH3/minimax_h3_audio_vae_fp32.safetensors",
     "8e505d95dd1561d47abd43d4238fd40d9bb1ae9e147ed0a4cba778d76ae4db48", "Audio VAE (0.6 GB)"),
    ("loras/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors", "models/loras/minimax_h3/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors",
     "2339acdf19bfe123f46b971ea35d367a84adb85de43627e1eceafa5a5b2b111e", "Turbo LoRA, 8 steps (2 GB)"),
    ("text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors", "models/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
     "35a88d51044231fe332301d7a62aa81e3f2cba62febeb446e2c1e3e0ef76f2c6", "Text encoder: Qwen3-VL 32B, 4-bit (16 GB; runs on any recent NVIDIA card)"),
]
have = lambda dest: os.path.exists(dest) and os.path.getsize(dest) > 0
need_gb = sum({"diffusion_models": 34 if full else 21, "text_encoders": 16, "vae": 3, "loras": 2}[p.split("/")[0]] for p, d, _, _ in FILES if not have(d))
if need_gb and free < need_gb + 10:
    sys.exit(f"Need about {need_gb + 10} GB free. Delete something first (disk-report.sh shows what's there).")
for path, dest, sha, what in FILES:
    print("==", what)
    if have(dest):
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
# The edit page's and SwarmUI's LoRA lists show this name
import json
meta_path = "custom_nodes/rv6_mobile/web/lora_info.json"
try: meta = json.load(open(meta_path))
except Exception: meta = {}
meta["minimax_h3/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors"] = {
    "name": "⚡ Turbo 8-step (video)", "baseModel": "MiniMax H3", "trainedWords": [],
    "note": "for the video model: with it at strength 1, 6-8 steps are enough instead of 20"}
os.makedirs(os.path.dirname(meta_path), exist_ok=True)
json.dump(meta, open(meta_path, "w"), indent=1)
PY

echo "== SwarmUI: restart with the new model, and its preset (20 steps, CFG 1)"
bash <(curl -fsSL "$SRC/add-swarmui.sh")

echo "== Test on the GPU, through SwarmUI: a 1-second video (the first load takes a few minutes)"
python3 - <<'PY'
import json, subprocess, threading, time, urllib.request
B = "http://127.0.0.1:7801"
LIMIT = 1200
def api(name, body, timeout=120):
    req = urllib.request.Request(f"{B}/API/{name}", json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))
sid = api("GetNewSession", {}, 60)["session_id"]
files = api("ListModels", {"session_id": sid, "path": "", "depth": 8, "subtype": "Stable-Diffusion", "allowRemote": False}).get("files") or []
# the 8-bit files first (the pruned one before the full one): the 66 GB bf16 file may not fit next to the chat model
cands = [f for f in files if "minimax_h3_fl2va" in (f.get("name") or "").lower()]
cands.sort(key=lambda f: ("int8" not in f["name"].lower(), "pruned" not in f["name"].lower(), f["name"]))
m = cands[0] if cands else None
if not m:
    raise SystemExit("MINIMAX H3 VIDEO TEST: SwarmUI doesn't list the model. Open SwarmUI, Models tab, press its refresh button, and run this again.")
print("SwarmUI sees:", m["name"], "| type:", m.get("class") or "NOT RECOGNISED", flush=True)
name = m["name"][:-len(".safetensors")] if m["name"].endswith(".safetensors") else m["name"]
def state():
    try: gpu = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader"], capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception: gpu = "?"
    try:
        q = json.load(urllib.request.urlopen("http://127.0.0.1:8188/api/queue", timeout=10))
        jobs = f"{len(q.get('queue_running') or [])} running, {len(q.get('queue_pending') or [])} waiting"
    except Exception: jobs = "ComfyUI didn't answer"
    return f"GPU {gpu} | ComfyUI jobs: {jobs}"
res = {}
def work():
    try:   # 22 frames is the shortest length on the model's frame grid after 5; 608x352 is its smallest listed size
        res["r"] = api("GenerateText2Image", {"session_id": sid, "images": 1, "model": name, "width": 608, "height": 352, "seed": 1,
                                              "text2videoframes": 22, "presets": ["Auto MiniMax H3 Video"], "donotsave": True,
                                              "prompt": "A red apple on a wooden table, the camera slowly moves closer. Quiet room sound."}, LIMIT + 60)
    except Exception as e:
        res["e"] = f"{type(e).__name__}: {str(e)[:200]}"
print("MINIMAX H3 VIDEO TEST: started", flush=True)
th = threading.Thread(target=work, daemon=True); t = time.time(); th.start()
while th.is_alive() and time.time() - t < LIMIT:
    th.join(20)
    if th.is_alive(): print(f"    {time.time() - t:>4.0f}s  still working… {state()}", flush=True)
if th.is_alive():
    try: api("InterruptAll", {"session_id": sid, "other_sessions": False}, 30)
    except Exception: pass
    print(f"MINIMAX H3 VIDEO TEST: no answer after {LIMIT // 60} minutes; cancelled. Last state: {state()}")
elif "e" in res: print(f"MINIMAX H3 VIDEO TEST: failed ({res['e']})")
elif res["r"].get("error") or not res["r"].get("images"): print(f"MINIMAX H3 VIDEO TEST: SwarmUI said: {str(res['r'].get('error') or res['r'])[:500]}")
else: print(f"MINIMAX H3 VIDEO TEST: ok in {time.time() - t:.0f}s ({str(res['r']['images'][0])[:40].split(';')[0]})")
PY

HOST=$(tailscale status --json 2>/dev/null | python3 -c 'import json,sys; print(json.load(sys.stdin)["Self"]["DNSName"].rstrip("."))' 2>/dev/null || true)
echo
echo "Done. Run edit-restart and reload the edit page: 🎬 Video → 🐝 MiniMax H3 (your photo is the first frame;"
echo "  with no photo chosen it makes the video from your text). Length, size, Turbo and LoRAs are on that panel."
echo "It is also in SwarmUI's own page, with every setting: https://${HOST:-<your-vm>.ts.net}:8444/"
echo "  Text to video: Models tab → minimax_h3_fl2va… → write a prompt (say what moves, and the sound) → Generate."
echo "    Its Auto preset sets 20 steps, CFG 1. Length: 'Text To Video' → Frames (24 a second; 73 = 3 s, 124 = 5 s)."
echo "  Photo to video: 'Init Image' → your photo, Creativity 0; 'Image To Video' → Video Model = minimax_h3_fl2va…,"
echo "    Video Steps 20, Video CFG 1; any picture model selected as the main model."
echo "  Sizes it likes: 864x480 (fast) up to 1344x768, or the same upright."
