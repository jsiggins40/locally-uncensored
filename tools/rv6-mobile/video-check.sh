#!/usr/bin/env bash
# Finds what turns 🎬 Video into static: renders your last video photo three
# ways (2 s each, 480p) and prints a link to each result.
#   A  plain Wan 2.2 image-to-video + LightX2V      (no SVI)
#   B  plain Wan 2.2 image-to-video + LightX2V + SVI LoRAs
#   C  the page's graph: SVI Pro node + LightX2V + SVI LoRAs
#   bash <(curl -fsSL https://raw.githubusercontent.com/jsiggins40/locally-uncensored/rv6-mobile-editor/tools/rv6-mobile/video-check.sh)
set -euo pipefail
cd "$HOME/ComfyUI"
curl -sf -o /dev/null localhost:8188 || { echo "ComfyUI isn't running: run edit-restart first"; exit 1; }
venv/bin/python - <<'PY'
import glob, json, math, os, re, subprocess, time, urllib.request, urllib.error
from PIL import Image
def get(path): return json.load(urllib.request.urlopen("http://127.0.0.1:8188" + path))
def opts(node, inp):
    v = get("/api/object_info/" + node).get(node, {}).get("input", {})
    v = v.get("required", {}).get(inp) or v.get("optional", {}).get(inp)
    return (v[0] if isinstance(v[0], list) else v[1].get("options", [])) if v else []
units, clips, loras = opts("UNETLoader", "unet_name"), opts("CLIPLoader", "clip_name"), opts("LoraLoaderModelOnly", "lora_name")
one = lambda xs, rx: (sorted(n for n in xs if re.search(rx, n, re.I)) or [None])[-1]
v = dict(high=one(units, r"wan2\.2_i2v_high_noise"), low=one(units, r"wan2\.2_i2v_low_noise"), clip=one(clips, "umt5"),
         vae=one(opts("VAELoader", "vae_name"), r"wan_2\.1_vae"), lightH=one(loras, "i2v.*lightx2v.*high"), lightL=one(loras, "i2v.*lightx2v.*low"),
         sviH=one(loras, "svi.*high.*pro"), sviL=one(loras, "svi.*low.*pro"))
print("Files:", json.dumps(v, indent=1))
if not all(v.values()): raise SystemExit("Some video files are missing; run add-video.sh")
# the photo the page last used for a video (uploaded as rv6m_<time>_img.jpg)
pics = sorted(glob.glob("input/rv6m_*_img.jpg"), key=os.path.getmtime)
img = os.path.basename(pics[-1]) if pics else None
if not img:
    Image.new("RGB", (640, 640), (150, 120, 100)).save("input/rv6m_check.png"); img = "rv6m_check.png"
w0, h0 = Image.open("input/" + img).size
r = w0 / h0; W = round(math.sqrt(832 * 480 * r) / 16) * 16; H = round(math.sqrt(832 * 480 / r) / 16) * 16
print(f"Photo: {img} -> {W}x{H}, 33 frames (2 s)")
prompt = "the person turns their head slowly and smiles, the camera slowly pushes in"

def graph(kind):
    svi = kind in "BC"
    p = {
        "m1": {"class_type": "UNETLoader", "inputs": {"unet_name": v["high"], "weight_dtype": "default"}},
        "m2": {"class_type": "UNETLoader", "inputs": {"unet_name": v["low"], "weight_dtype": "default"}},
        "te": {"class_type": "CLIPLoader", "inputs": {"clip_name": v["clip"], "type": "wan"}},
        "va": {"class_type": "VAELoader", "inputs": {"vae_name": v["vae"]}},
        "h1": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["m1", 0], "lora_name": v["lightH"], "strength_model": 1}},
        "l1": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["m2", 0], "lora_name": v["lightL"], "strength_model": 1}},
        "pos": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["te", 0], "text": prompt}},
        "neg": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["te", 0], "text": "blurry, static"}},
        "img": {"class_type": "LoadImage", "inputs": {"image": img}},
        "fit": {"class_type": "ImageScale", "inputs": {"image": ["img", 0], "upscale_method": "lanczos", "width": W, "height": H, "crop": "center"}},
    }
    hm, lm = ["h1", 0], ["l1", 0]
    if svi:
        p["h2"] = {"class_type": "LoraLoaderModelOnly", "inputs": {"model": hm, "lora_name": v["sviH"], "strength_model": 1}}
        p["l2"] = {"class_type": "LoraLoaderModelOnly", "inputs": {"model": lm, "lora_name": v["sviL"], "strength_model": 1}}
        hm, lm = ["h2", 0], ["l2", 0]
    p["hs"] = {"class_type": "ModelSamplingSD3", "inputs": {"model": hm, "shift": 5}}
    p["ls"] = {"class_type": "ModelSamplingSD3", "inputs": {"model": lm, "shift": 5}}
    if kind == "C":
        p["anc"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["fit", 0], "vae": ["va", 0]}}
        p["i"] = {"class_type": "WanImageToVideoSVIPro", "inputs": {"positive": ["pos", 0], "negative": ["neg", 0], "length": 33,
                  "anchor_samples": ["anc", 0], "motion_latent_count": 1}}
    else:
        p["i"] = {"class_type": "WanImageToVideo", "inputs": {"positive": ["pos", 0], "negative": ["neg", 0], "vae": ["va", 0],
                  "width": W, "height": H, "length": 33, "batch_size": 1, "start_image": ["fit", 0]}}
    common = {"steps": 4, "cfg": 1, "sampler_name": "euler", "scheduler": "simple", "positive": ["i", 0], "negative": ["i", 1]}
    p["sh"] = {"class_type": "KSamplerAdvanced", "inputs": {**common, "model": ["hs", 0], "add_noise": "enable", "noise_seed": 7,
               "latent_image": ["i", 2], "start_at_step": 0, "end_at_step": 2, "return_with_leftover_noise": "enable"}}
    p["sl"] = {"class_type": "KSamplerAdvanced", "inputs": {**common, "model": ["ls", 0], "add_noise": "disable", "noise_seed": 0,
               "latent_image": ["sh", 0], "start_at_step": 2, "end_at_step": 10000, "return_with_leftover_noise": "disable"}}
    p["dec"] = {"class_type": "VAEDecode", "inputs": {"samples": ["sl", 0], "vae": ["va", 0]}}
    p["mv"] = {"class_type": "CreateVideo", "inputs": {"images": ["dec", 0], "fps": 16}}
    p["9"] = {"class_type": "SaveVideo", "inputs": {"video": ["mv", 0], "filename_prefix": f"rv6_check_{kind}", "format": "mp4", "format.codec": "h264"}}
    return p

host = ""
try:
    host = json.loads(subprocess.run(["tailscale", "status", "--json"], capture_output=True, text=True).stdout)["Self"]["DNSName"].rstrip(".")
except Exception: pass
for kind, label in (("A", "plain Wan, no SVI"), ("B", "plain Wan + SVI LoRAs"), ("C", "page's SVI Pro graph")):
    req = urllib.request.Request("http://127.0.0.1:8188/api/prompt", json.dumps({"prompt": graph(kind)}).encode(), {"Content-Type": "application/json"})
    try:
        pid = json.load(urllib.request.urlopen(req))["prompt_id"]
    except urllib.error.HTTPError as e:
        print(f"{kind} ({label}): REJECTED", e.read().decode()[:600]); continue
    t = time.time()
    while True:
        h = get("/api/history/" + pid)
        if pid in h: break
        if time.time() - t > 1800: print(kind, "timed out"); break
        time.sleep(3)
    st = h.get(pid, {}).get("status", {})
    out = (h.get(pid, {}).get("outputs", {}).get("9") or {}).get("images") or []
    print(f"{kind} ({label}): {st.get('status_str')} in {time.time() - t:.0f}s")
    if out: print(f"   open: https://{host or '<vm>'}/api/view?filename={out[0]['filename']}&type=output")
    else: print("  ", json.dumps(st)[:600])
PY
echo "== Warnings from ComfyUI during the test"
tmux capture-pane -pt comfy -S -400 | grep -i "lora key not loaded\|nan\|error\|warning" | sort | uniq -c | tail -10 || true
