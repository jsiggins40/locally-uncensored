#!/usr/bin/env bash
# Adds SwarmUI (mcmonkeyprojects, MIT) next to the edit page: a general image
# generation page with a model and LoRA browser, every setting, image history
# and the full ComfyUI editor. Served by Tailscale at https://<vm>.ts.net:8444
# (tailnet only, no login), for the phone and the laptop.
#   - Uses the ComfyUI and the models this VM already has (no second copy, and
#     the models stay loaded once for both pages). SwarmUI's own node pack is
#     linked into ComfyUI for that.
#   - Makes a preset for each installed model with the steps and CFG it was made
#     for, and links it to the model: choosing the model switches its preset on.
#   - Starts by itself after a boot (edit-restart brings it up).
# About 2 GB (.NET + SwarmUI); the first start builds it, which takes a few minutes.
# Pinned to the SwarmUI version this was written against; newest instead:
#   SWARM_REF=master bash <(curl ...)
#   bash <(curl -fsSL https://raw.githubusercontent.com/jsiggins40/locally-uncensored/rv6-mobile-editor/tools/rv6-mobile/add-swarmui.sh)
set -euo pipefail
SRC="https://raw.githubusercontent.com/jsiggins40/locally-uncensored/rv6-mobile-editor/tools/rv6-mobile"
C="$HOME/ComfyUI"; SW="$HOME/SwarmUI"
SWARM_REF="${SWARM_REF:-d347f3061ce9aa19e0d3950e5189415022de6659}"   # 12 Nov 2026
[ -d "$C/models" ] || { echo "No ComfyUI here: run setup.sh (or studio-setup.sh) first."; exit 1; }
mkdir -p "$HOME/.local/bin"

echo "== Packages"
sudo apt-get install -y -qq git wget curl tmux >/dev/null

echo "== SwarmUI"
[ -d "$SW/.git" ] || git clone -q https://github.com/mcmonkeyprojects/SwarmUI "$SW"
git -C "$SW" fetch -q origin "$SWARM_REF" 2>/dev/null || git -C "$SW" fetch -q origin
git -C "$SW" checkout -q "$SWARM_REF" 2>/dev/null || git -C "$SW" checkout -q FETCH_HEAD
echo "   version $(git -C "$SW" log -1 --format='%h from %cs')"

echo "== .NET (SwarmUI is built with it; installed in ~/.dotnet, a few minutes)"
if "$HOME/.dotnet/dotnet" --list-sdks 2>/dev/null | grep -q '^8\.' && "$HOME/.dotnet/dotnet" --list-sdks 2>/dev/null | grep -q '^10\.'; then
  echo "   already installed"
else
  (cd "$SW" && ./launchtools/linux-dotnet-install.sh "$HOME/.dotnet" >/dev/null 2>&1) || true
  "$HOME/.dotnet/dotnet" --list-sdks | grep -q '^8\.' || { echo "   .NET did not install. Try again; if it keeps failing, send the output of: cd ~/SwarmUI && ./launchtools/linux-dotnet-install.sh"; exit 1; }
fi

echo "== Settings: this VM's ComfyUI and its models"
mkdir -p "$SW/Data"
if [ ! -s "$SW/Data/Settings.fds" ]; then
  cat > "$SW/Data/Settings.fds" <<EOF
IsInstalled: true
InstallDate: $(date +%F)
InstallVersion: add-swarmui
LaunchMode: none
Paths:
    ModelRoot: $C/models
    SDModelFolder: checkpoints
    SDLoraFolder: loras
    SDVAEFolder: vae
    SDEmbeddingFolder: embeddings
    SDControlNetsFolder: controlnet;model_patches
    SDClipFolder: text_encoders;clip
    SDClipVisionFolder: clip_vision
Network:
    Host: localhost
    Port: 7801
    PortCanChange: false
EOF
  echo "   written"
else
  echo "   keeping the existing Settings.fds"
fi
if [ ! -s "$SW/Data/Backends.fds" ]; then
  cat > "$SW/Data/Backends.fds" <<EOF
0:
    type: comfyui_api
    title: ComfyUI on this VM (shared with the edit page)
    enabled: true
    settings:
        Address: http://127.0.0.1:8188
        AllowIdle: true
        OverQueue: 1
        EnableFrontendDev: false
EOF
  echo "   backend: the ComfyUI that is already running (127.0.0.1:8188)"
fi

echo "== SwarmUI's nodes for ComfyUI"
ln -sfn "$SW/src/BuiltinExtensions/ComfyUIBackend/ExtraNodes/SwarmComfyCommon" "$C/custom_nodes/SwarmComfyCommon"
"$C/venv/bin/python" -c "import cv2" 2>/dev/null || "$C/venv/bin/pip" install -q opencv-python-headless \
  || echo "   (couldn't install opencv: SwarmUI's nodes may not load)"

cat > "$HOME/.local/bin/swarm-start" <<'SH'
#!/usr/bin/env bash
# (Re)starts SwarmUI (add-swarmui.sh) on 127.0.0.1:7801 in a tmux session called
# "swarm" and publishes it on Tailscale at :8444. The first start builds it.
SW="$HOME/SwarmUI"
tmux kill-session -t swarm 2>/dev/null || true
tmux new -d -s swarm "cd $SW && ./launch-linux.sh --launch_mode none --host localhost --port 7801; echo; echo 'SwarmUI stopped'; sleep 600"
sudo tailscale serve --bg --https=8444 http://127.0.0.1:7801 >/dev/null 2>&1 || true
for _ in $(seq 300); do curl -sf -o /dev/null -m 3 localhost:7801 && exit 0; sleep 2; done
echo "SwarmUI did not start. See: tmux attach -t swarm"; exit 1
SH
chmod +x "$HOME/.local/bin/swarm-start"

echo "== Restarting ComfyUI with SwarmUI's nodes"
if command -v edit-restart >/dev/null; then
  sudo curl -fsSL "$SRC/edit-restart.sh" -o /usr/local/bin/edit-restart 2>/dev/null && sudo chmod +x /usr/local/bin/edit-restart || true
fi
tmux kill-session -t comfy 2>/dev/null || true
tmux new -d -s comfy "cd $C && venv/bin/python main.py --listen 127.0.0.1 --port 8188"
for _ in $(seq 120); do curl -sf -o /dev/null localhost:8188 && break; sleep 1; done
if curl -s localhost:8188/api/object_info/SwarmKSampler | grep -q SwarmKSampler; then echo "   ComfyUI has SwarmUI's nodes"
else echo "   WARNING: ComfyUI didn't load SwarmUI's nodes (see: tmux attach -t comfy). SwarmUI may not generate."; fi

echo "== Starting SwarmUI (the first start builds it: about 5 minutes)"
"$HOME/.local/bin/swarm-start" || { tmux capture-pane -pt swarm | tail -25; exit 1; }

echo "== Presets: each model's own steps and CFG, switched on when you choose the model"
python3 - "$C/models" <<'PY' || echo "   (couldn't set the presets; SwarmUI itself works. Send the lines above.)"
import json, os, re, sys, time, urllib.request
B = "http://127.0.0.1:7801/API/"
def api(name, body):
    req = urllib.request.Request(B + name, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=120))
sid = api("GetNewSession", {})["session_id"]
# (file name pattern, preset title, steps, CFG, what it is). First match wins.
# Titles: letters, digits, spaces and hyphens only (SwarmUI strips other characters).
RULES = [
    (r"minimax.?h3.?image", "Auto MiniMax H3 Image", 6, 1, "MiniMax H3 Image: 6 steps, CFG 1"),
    (r"minimax.?h3.?(fl2va|ref2va)", "Auto MiniMax H3 Video", 20, 1, "MiniMax H3 video: 20 steps, CFG 1 (6-8 steps with the Turbo LoRA)"),
    (r"krea.*turbo", "Auto Krea 2 Turbo", 8, 1, "Krea 2 Turbo: 8 steps, CFG 1"),
    (r"moss.?image", "Auto MOSS Image", 25, 2.5, "MOSS-Image: 25 steps, CFG 2.5"),
    (r"omni.*(q21|qwen.?image.?2\.1)|qwen.?image.?2\.1.*omni", "Auto Qwen 21 Omni", 35, 5, "Qwen-Image 2.1 Uncensored Omni: 35 steps, CFG 5 (its authors' settings)"),
    (r"noct.?q|noct.*qwen", "Auto Qwen 21 Noct Q", 25, 3, "Noct Q (Qwen-Image 2.1): 25 steps, CFG 3 (its author's settings)"),
    (r"qwen.?image.?2\.1.*turbo", "Auto Qwen 21 Turbo", 8, 1, "Qwen-Image 2.1 Turbo: 8 steps, CFG 1"),
    (r"qwen.?image.?2\.1", "Auto Qwen 21", 25, 1, "Qwen-Image 2.1: 25 steps, CFG 1"),
    (r"rapid.?aio", "Auto Rapid AIO", 8, 1, "Qwen-Image-Edit Rapid AIO: 8 steps, CFG 1"),
    (r"qwen.?image.?edit", "Auto Qwen Image Edit", 30, 1, "Qwen-Image-Edit: 30 steps, CFG 1 (SwarmUI: edit models need CFG 1)"),
    (r"qwen.?image.?\d{4}", "Auto Qwen Image 2512", 40, 4, "Qwen-Image 2512: 40 steps, CFG 4 (without a speed LoRA)"),
]
root = sys.argv[1]
models = []
for sub in ("diffusion_models", "checkpoints", "unet"):
    base = os.path.join(root, sub)
    for d, _, files in os.walk(base, followlinks=True):
        for f in files:
            if f.endswith((".safetensors", ".gguf")):
                models.append(os.path.relpath(os.path.join(d, f), base).replace(os.sep, "/"))
try: links = api("GetMyUserData", {"session_id": sid}).get("model_preset_links") or {}
except Exception: links = {}
mine = links.setdefault("Stable-Diffusion", {})
made = set()
for m in sorted(models):
    rule = next((r for r in RULES if re.search(r[0], m, re.I)), None)
    if not rule: continue
    _, title, steps, cfg, desc = rule
    if title not in made:
        body = {"session_id": sid, "title": title, "description": desc + ". Made by add-swarmui.sh; switch it off in the Presets tab to use your own settings.",
                "param_map": {"steps": str(steps), "cfgscale": str(cfg)}}
        r = api("AddNewPreset", body)
        if "preset_fail" in r: r = api("AddNewPreset", {**body, "is_edit": True, "editing": title})
        made.add(title)
    key = m[:-len(".safetensors")] if m.endswith(".safetensors") else m
    if not mine.get(key):          # a link you set yourself is kept
        mine[key] = [title]
    print(f"   {key}: {steps} steps, CFG {cfg}")
api("SetPresetLinks", {"session_id": sid, **links})
print(f"   {len(made)} presets for {sum(1 for v in mine.values() if v and v[0].startswith('Auto '))} models")
try:
    b = api("ListBackends", {"session_id": sid})
    for k, v in b.items():
        if isinstance(v, dict): print("   backend:", v.get("title") or v.get("type"), "->", v.get("status"))
except Exception as e:
    print("   (couldn't read the backend's status:", type(e).__name__ + ")")
PY

HOST=$(tailscale status --json 2>/dev/null | python3 -c 'import json,sys; print(json.load(sys.stdin)["Self"]["DNSName"].rstrip("."))' 2>/dev/null || true)
echo
echo "=============================================================="
echo " SWARMUI READY. On the phone or laptop (Tailscale connected) open:"
echo "   https://${HOST:-<your-vm>.ts.net}:8444/"
echo " Pick a model in the Models tab at the bottom: its Auto preset sets"
echo " the steps and CFG. Pictures are saved in ~/SwarmUI/Output."
echo " It starts by itself after a boot; by hand: swarm-start"
echo "=============================================================="
