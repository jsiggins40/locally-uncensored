#!/usr/bin/env bash
# Installs the RV6 mobile edit page into ComfyUI, restarts ComfyUI, runs one
# test edit on the GPU and prints the address to open on the phone.
#   curl -fsSL https://raw.githubusercontent.com/jsiggins40/locally-uncensored/rv6-mobile-editor/tools/rv6-mobile/install.sh | bash
set -euo pipefail

COMFY="$HOME/ComfyUI"
SRC="https://raw.githubusercontent.com/jsiggins40/locally-uncensored/rv6-mobile-editor/tools/rv6-mobile"
DEST="$COMFY/custom_nodes/rv6_mobile"

echo "== Downloading page"
mkdir -p "$DEST/web"
curl -fsSL "$SRC/__init__.py"   -o "$DEST/__init__.py"
curl -fsSL "$SRC/edit.html"     -o "$DEST/web/edit.html"
curl -fsSL "$SRC/workflow.json" -o "$DEST/web/workflow.json"

echo "== Restarting ComfyUI"
tmux kill-session -t comfy 2>/dev/null || true
tmux new -d -s comfy "cd $COMFY && venv/bin/python main.py --listen 127.0.0.1 --port 8188 --highvram"
for i in $(seq 1 90); do
  curl -sf -o /dev/null localhost:8188/extensions/rv6_mobile/edit.html && break
  sleep 1
done
if ! curl -sf -o /dev/null localhost:8188/extensions/rv6_mobile/edit.html; then
  echo "FAILED: ComfyUI is not serving the page. Last log lines:"
  tmux capture-pane -pt comfy | tail -20
  exit 1
fi

echo "== Test edit on the GPU"
cd "$COMFY"
venv/bin/python - <<'PY'
import json, time, urllib.request, urllib.error
from PIL import Image, ImageDraw
Image.new("RGB", (512, 512), (120, 120, 120)).save("input/rv6m_test_img.png")
m = Image.new("RGB", (512, 512), "black"); ImageDraw.Draw(m).rectangle([128, 128, 384, 384], fill="white")
m.save("input/rv6m_test_mask.png")
import os
if not os.path.exists("models/checkpoints/Realistic_Vision_V6.0_NV_B1_inpainting_fp16.safetensors"):
    print("TEST: skipped (Realistic Vision not installed; Describe edit is tested by add-edit-model.sh)")
    raise SystemExit(0)
p = json.load(open("custom_nodes/rv6_mobile/web/workflow.json"))
p["5"]["inputs"]["image"] = "rv6m_test_img.png"
p["10"]["inputs"]["image"] = "rv6m_test_mask.png"
p["3"]["inputs"]["text"] = "a red apple on a table, RAW photo"
p["4"]["inputs"]["text"] = "blurry"
req = urllib.request.Request("http://127.0.0.1:8188/api/prompt", json.dumps({"prompt": p}).encode(),
                             {"Content-Type": "application/json"})
try:
    pid = json.load(urllib.request.urlopen(req))["prompt_id"]
except urllib.error.HTTPError as e:
    print("TEST REJECTED:", e.read().decode()[:800]); raise SystemExit(1)
t = time.time()
while True:
    h = json.load(urllib.request.urlopen("http://127.0.0.1:8188/api/history/" + pid))
    if pid in h:
        print("TEST:", h[pid]["status"]["status_str"], f"in {time.time() - t:.1f}s (first run includes model loading)")
        if h[pid]["status"]["status_str"] != "success":
            print(json.dumps(h[pid]["status"])[:800]); raise SystemExit(1)
        break
    if time.time() - t > 300: print("TEST TIMED OUT"); raise SystemExit(1)
    time.sleep(1)
PY

echo "== Tap-to-select (Segment Anything, ~2.5 GB the first time)"
if curl -sf -m 600 -o /tmp/rv6m_seg.png -H "Content-Type: application/json" \
     -d '{"image":"rv6m_test_img.png","points":[[256,256]],"labels":[1]}' localhost:8188/rv6m/segment; then
  echo "SELECT: ok"
else
  echo "SELECT: not working (painting still works). Last log lines:"
  tmux capture-pane -pt comfy | grep -i "rv6_mobile\|error" | tail -8
fi

echo
read -rp "Also add the SDXL models Anteros XXXL and bigASP (about 13 GB)? [y/N] " a </dev/tty || a=n
if [[ "$a" =~ ^[Yy] ]]; then bash <(curl -fsSL "$SRC/add-models.sh"); fi

echo "== Phone access"
sudo tailscale serve --bg 8188 >/dev/null 2>&1 || true
HOST=$(tailscale status --json 2>/dev/null | python3 -c 'import json,sys; print(json.load(sys.stdin)["Self"]["DNSName"].rstrip("."))' 2>/dev/null || true)
if tailscale status 2>/dev/null | grep -q "logged out\|Logged out"; then
  echo "WARNING: Tailscale is logged out on this VM, run: sudo tailscale up"
fi
echo
echo "DONE. On your phone (Tailscale app connected), open:"
echo "  https://${HOST:-<your-vm>.ts.net}/extensions/rv6_mobile/edit.html"
