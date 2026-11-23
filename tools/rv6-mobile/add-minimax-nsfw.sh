#!/usr/bin/env bash
# Uncensored extras for MiniMax H3 Image (add-minimax-image.sh) and MiniMax H3 video
# (add-minimax-video.sh); run this after either or both:
#   - "Heretic" text encoder: an abliterated copy of the Qwen3-VL 32B model that reads
#     your prompt, made to pass explicit wording on as written. 4-bit, 15.7 GB, the
#     same layout as the standard encoder. On the edit page it is the "🔓 Uncensored"
#     switch that appears in 🐝 Swarm mode when a MiniMax model is chosen.
#   - "Full nude" LoRA for MiniMax H3 Image (0.3 GB), in Swarm mode's LoRA list.
#   - HearmemanAI's stills-trained anatomy LoRAs (Hearmeman/minimax-h3-loras: HMPussy V2,
#     HMBreasts V2, HMPenis, HMInnie; 0.8 GB). They were made for the MiniMax H3 VIDEO
#     model; they change the same layers the image model has, so they load on it, but
#     how well they work on stills from the image model is not known.
#   - When the video model is installed: HearmemanAI's motion LoRA HMNSFW V2.5 (0.1 GB;
#     word "hmmotion", strength 0.5-0.9), which teaches the video model sex motion. Add
#     "all" for his single-act ones too (cumshot, squirt, masturbation; 1 GB).
# Both are pinned to the exact files that were looked at and checked against their
# SHA-256, then tried on the GPU through SwarmUI. General-purpose ones only: no LoRAs
# of real people, none for a single act. The encoder is Apache-2.0; the LoRA has no
# licence text; MiniMax H3 Image itself has territory limits (see add-minimax-image.sh).
#   bash <(curl -fsSL https://raw.githubusercontent.com/jsiggins40/locally-uncensored/rv6-mobile-editor/tools/rv6-mobile/add-minimax-nsfw.sh)
set -euo pipefail
cd "$HOME/ComfyUI"
ls models/diffusion_models/*minimax_h3_image* models/diffusion_models/*minimax_h3_fl2va* models/diffusion_models/*minimax_h3_ref2va* >/dev/null 2>&1 \
  || { echo "No MiniMax H3 model is installed yet. Run add-minimax-image.sh and/or add-minimax-video.sh first, then this."; exit 1; }
WANT_ALL=""; for a in "$@"; do [ "$a" = all ] && WANT_ALL=1; done
export WANT_ALL
[ -x "$HOME/.local/bin/swarm-start" ] || { echo "This needs SwarmUI: run add-swarmui.sh first."; exit 1; }

venv/bin/python - <<'PY'
import glob, hashlib, json, os, shutil, sys
from huggingface_hub import hf_hub_download
has_video = bool(glob.glob("models/diffusion_models/*minimax_h3_fl2va*") + glob.glob("models/diffusion_models/*minimax_h3_ref2va*"))
free = shutil.disk_usage(os.path.realpath("models")).free / 1e9
print(f"== Free disk: {free:.0f} GB")
# (repo, commit, file, SHA-256, saved as, what it is)
FILES = [
    ("Momoking/Qwen3-VL-32B-Heretic-MiniMax-H3-NVFP4", "0a9486d17cfb20c1a7abc97118c4d95910b9b1f4", "qwen3vl_32b_heretic_minimax_h3_nvfp4.safetensors",
     "a166c7bbbe66a22065159e478335fee4a633c4a3e3bb34c8e8ac4cc91bf4996f", "models/text_encoders/qwen3vl_32b_heretic_minimax_h3_nvfp4.safetensors",
     "Uncensored text encoder (15.7 GB)"),
    ("Tadness/FullNude", "b58435b73227b713885721fd233b80f548b512f6", "FullNude.safetensors",
     "786a4a8e80c79c9ad6cf1a53959cb415e8ed62f74a9ada6ca6df31c4292e5506", "models/loras/minimax_h3_image/fullnude.safetensors",
     "Full nude LoRA (0.3 GB)"),
]
HM, HMREV = "Hearmeman/minimax-h3-loras", "e5f4fb9030d1ac27caa0f773b96fbd2d4cafadbb"
FILES += [
    (HM, HMREV, "HMPussyV2.safetensors", "7b9559077d8ac623d52b64b438d8ad60cb85b31b6fecf011aab238b70962a2ed",
     "models/loras/minimax_h3_image/hm_pussy_v2.safetensors", "HMPussy V2 (made for the video model)"),
    (HM, HMREV, "HMBreastsV2.safetensors", "d260653bdf10775380a44c8f1486bcc4690bd57085cb034002a8fb878f1ad36f",
     "models/loras/minimax_h3_image/hm_breasts_v2.safetensors", "HMBreasts V2 (made for the video model)"),
    (HM, HMREV, "HMPenis_v2_e35.safetensors", "c6c58e9fee848b45e99f97d2520aba4ac63dfc354c07e13c29ac5d8a31a68060",
     "models/loras/minimax_h3_image/hm_penis_v1.safetensors", "HMPenis v1.0 (made for the video model)"),
    (HM, HMREV, "HMInnie_v1_e50.safetensors", "499196c9d0e5f81ff575ba39a82987112c3bb1e09fbede858877cd950d6c8833",
     "models/loras/minimax_h3_image/hm_innie_v1.safetensors", "HMInnie v1.0 (made for the video model)"),
]
if has_video:      # motion LoRAs: only useful on the video model, so they go in their own folder
    FILES += [(HM, HMREV, "HMNSFW-AIO-V2.5.safetensors", "a07732a84fd733085eb5d910f602f918fa7a3658117116927e4329f5951a9d2d",
               "models/loras/minimax_h3/hm_nsfw_motion_v2_5.safetensors", "HMNSFW V2.5, sex motion (video model only)")]
    if os.environ.get("WANT_ALL"):
        FILES += [
            (HM, HMREV, "HMCumshot_V2.safetensors", "1a5b7948bb97f27737e62c3dd5497a3afb77517f230787f45e45c7d8fe3dc24d",
             "models/loras/minimax_h3/hm_cumshot_v0_5.safetensors", "HMCumshot V0.5 (video model only)"),
            (HM, HMREV, "HMSquirtV0.1.safetensors", "b25cc54db8e12012e9c1a60c514d767208c855cbf6e81605827d86913249cf24",
             "models/loras/minimax_h3/hm_squirt_v0_1.safetensors", "HMSquirt v0.1 (video model only)"),
            (HM, HMREV, "hmmasturbation_v2.safetensors", "f0e4bfbe5baebe972880d2250a227e2f475aa8c18d6d01ed897288d9741659d2",
             "models/loras/minimax_h3/hm_masturbation_v2.safetensors", "HMMasturbation v2 (video model only)"),
        ]
need = [f for f in FILES if not (os.path.exists(f[4]) and os.path.getsize(f[4]) > 0)]
if need and free < 20:
    sys.exit("Need about 20 GB free. Delete something first (disk-report.sh shows what's there).")
for repo, rev, fname, sha, dest, what in FILES:
    print("==", what, f"({repo})")
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        print("   already have", dest); continue
    try:
        tmp = hf_hub_download(repo, fname, revision=rev, local_dir="models/_hf_tmp")
    except Exception as e:
        print(f"   download failed ({type(e).__name__}: {str(e)[:150]}); skipped"); continue
    h = hashlib.sha256()
    with open(tmp, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""): h.update(b)
    if h.hexdigest() != sha:
        os.remove(tmp); print("   doesn't match the checked file; not installed"); continue
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    shutil.move(tmp, dest)
    print("   installed", dest)
shutil.rmtree("models/_hf_tmp", ignore_errors=True)
meta_path = "custom_nodes/rv6_mobile/web/lora_info.json"
try: meta = json.load(open(meta_path))
except Exception: meta = {}
# Names, prompt words the page adds by itself, and how to prompt each (from their authors)
INFO = {
    "fullnude": ("🔞 Full nude", [], "nudity (its author gives no description)"),
    "hm_pussy_v2": ("🔞 HM Pussy", [], "write it in plain words (pussy, vagina, anus); made for the video model"),
    "hm_breasts_v2": ("🔞 HM Breasts", [], "describe the chest in plain prose, no special word; made for the video model"),
    "hm_penis_v1": ("🔞 HM Penis", ["HMPenis"], "say the camera direction (front, back, side); made for the video model"),
    "hm_innie_v1": ("🔞 HM Innie", [], 'write "inniepussy shown from the front / from behind / from below" in place of the usual word; made for the video model'),
}
for stem, (name, words, note) in INFO.items():
    if os.path.exists(f"models/loras/minimax_h3_image/{stem}.safetensors"):
        meta[f"minimax_h3_image/{stem}.safetensors"] = {"name": name, "baseModel": "MiniMax H3 Image", "trainedWords": words, "note": note}
VIDEO_INFO = {
    "hm_nsfw_motion_v2_5": ("🔞 HM sex motion", ["hmmotion"], "strength 0.5-0.9; write one long flowing paragraph, not a list of words"),
    "hm_cumshot_v0_5": ("🔞 HM Cumshot", ["cumshot"], "strength 0.9; describe it in plain, blunt words"),
    "hm_squirt_v0_1": ("🔞 HM Squirt", ["hmsquirt"], "experimental; its author runs HM Pussy next to it at 0.4"),
    "hm_masturbation_v2": ("🔞 HM Masturbation", [], "describe the action in plain words"),
}
for stem, (name, words, note) in VIDEO_INFO.items():
    if os.path.exists(f"models/loras/minimax_h3/{stem}.safetensors"):
        meta[f"minimax_h3/{stem}.safetensors"] = {"name": name, "baseModel": "MiniMax H3", "trainedWords": words, "note": note + "; for the video model"}
os.makedirs(os.path.dirname(meta_path), exist_ok=True)
json.dump(meta, open(meta_path, "w"), indent=1)
PY

echo "== Tests on the GPU, through SwarmUI (loading the encoder takes a few minutes)"
curl -sf -o /dev/null -m 5 localhost:7801 || "$HOME/.local/bin/swarm-start"
python3 - <<'PY'
import json, os, shutil, subprocess, threading, time, urllib.request
B = "http://127.0.0.1:7801"
LIMIT = 900        # seconds one test may take before it is cancelled
def api(name, body, timeout=3600):
    req = urllib.request.Request(f"{B}/API/{name}", json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))
sid = api("GetNewSession", {}, 60)["session_id"]
print("   asking SwarmUI to look for the new files (up to 2 minutes)...", flush=True)
try: api("TriggerRefresh", {"session_id": sid}, 120)      # make SwarmUI see the new files
except Exception as e: print("   (SwarmUI didn't finish refreshing its lists:", type(e).__name__ + "; restart it with swarm-start if the tests can't find the files)", flush=True)
def names(subtype):
    return [f.get("name") or "" for f in api("ListModels", {"session_id": sid, "path": "", "depth": 8, "subtype": subtype, "allowRemote": False}, 120).get("files") or []]
model = next((n for n in names("Stable-Diffusion") if "minimax_h3_image" in n.lower()), None)
enc = next((n for n in names("Clip") if "heretic" in n.lower() and "minimax" in n.lower()), None)
loras = [n for n in names("LoRA") if n.lower().replace("\\", "/").startswith("minimax_h3_image/")]
vmodel = next((n for n in names("Stable-Diffusion") if "minimax_h3_fl2va" in n.lower()), None)
vloras = [n for n in names("LoRA") if n.lower().replace("\\", "/").startswith("minimax_h3/hm_")]
print("SwarmUI sees: image model", model, "| video model", vmodel, "| encoder", enc, "| LoRAs", len(loras), "+", len(vloras), "for video")
if not model and not vmodel: raise SystemExit("SwarmUI doesn't list a MiniMax H3 model; run add-minimax-image.sh or add-minimax-video.sh again.")
clean = lambda n: n[:-len(".safetensors")] if n.endswith(".safetensors") else n
def state():      # what the GPU and ComfyUI are doing, for the "still working" lines
    try: gpu = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader"], capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception: gpu = "?"
    try:
        q = json.load(urllib.request.urlopen("http://127.0.0.1:8188/api/queue", timeout=10))
        jobs = f"{len(q.get('queue_running') or [])} running, {len(q.get('queue_pending') or [])} waiting"
    except Exception: jobs = "ComfyUI didn't answer"
    return f"GPU {gpu} | ComfyUI jobs: {jobs}"
def gen(label, extra, video=False):
    """One test picture (or, with video, a 1-second 608x352 test video on the video model). Says every 20 s that it is still working; cancelled after LIMIT.
    True = picture made, False = SwarmUI refused or failed, None = no answer in time (says nothing about the file)."""
    res = {}
    def work():
        try:
            base = {"model": clean(vmodel), "width": 608, "height": 352, "text2videoframes": 22, "presets": ["Auto MiniMax H3 Video"]} if video else \
                   {"model": clean(model), "width": 1024, "height": 1024, "presets": ["Auto MiniMax H3 Image"]}
            res["r"] = api("GenerateText2Image", {"session_id": sid, "images": 1, "seed": 1, "prompt": "a red apple on a wooden table, photo",
                                                  "donotsave": True, **base, **extra}, LIMIT + 60)
        except Exception as e:
            res["e"] = f"{type(e).__name__}: {str(e)[:200]}"
    print(f"{label}: started", flush=True)
    th = threading.Thread(target=work, daemon=True); t = time.time(); th.start()
    while th.is_alive() and time.time() - t < LIMIT:
        th.join(20)
        if th.is_alive(): print(f"    {time.time() - t:>4.0f}s  still working… {state()}", flush=True)
    if th.is_alive():
        try: api("InterruptAll", {"session_id": sid, "other_sessions": False}, 30)
        except Exception: pass
        print(f"{label}: no answer after {LIMIT // 60} minutes; cancelled. Last state: {state()}", flush=True); return None
    if "e" in res:
        print(f"{label}: failed ({res['e']})", flush=True); return False
    r = res["r"]
    if r.get("error") or not r.get("images"):
        print(f"{label}: SwarmUI said: {str(r.get('error') or r)[:400]}", flush=True); return False
    print(f"{label}: ok in {time.time() - t:.0f}s", flush=True); return True
def aside(path, why):
    os.makedirs("models/_disabled", exist_ok=True)
    shutil.move(path, os.path.join("models/_disabled", os.path.basename(path)))
    print(f"   moved {os.path.basename(path)} to models/_disabled ({why}), so the page won't offer it")
# First the model as it is: if that fails the problem is elsewhere, and nothing is moved aside
if model:
    if not gen("MINIMAX H3 IMAGE, AS IT IS", {}):
        print("The image model itself didn't make a picture through SwarmUI, so its extras weren't tested. They stay installed. Send the lines above.")
    else:
        if enc and gen("UNCENSORED ENCODER TEST", {"qwenmodel": enc}) is False:
            aside("models/text_encoders/" + enc, "it didn't work")
        for lora in loras:      # each on a plain prompt: this shows it loads and runs, not what it draws
            if gen(f"LORA TEST {os.path.basename(lora)}", {"loras": [clean(lora)], "loraweights": ["1"]}) is False:
                aside("models/loras/" + lora, "it didn't work")
if vmodel and vloras:
    if not gen("MINIMAX H3 VIDEO, AS IT IS", {}, video=True):
        print("The video model itself didn't make a video through SwarmUI, so its LoRAs weren't tested. They stay installed. Send the lines above.")
    else:
        for lora in vloras:
            if gen(f"VIDEO LORA TEST {os.path.basename(lora)}", {"loras": [clean(lora)], "loraweights": ["0.7"]}, video=True) is False:
                aside("models/loras/" + lora, "it didn't work")
PY

echo
echo "Done. Run edit-restart, reload the edit page, then: 🐝 Swarm → Model → minimax_h3_image…"
echo "  Content: 🔓 Uncensored (the text encoder) · LoRAs: tap the ones you want"
echo "Video (when installed): in SwarmUI's own page (:8444), the LoRAs tab at the bottom has all of them;"
echo "  the anatomy ones work on the video model too (they were made for it)."
