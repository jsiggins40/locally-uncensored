#!/usr/bin/env bash
# Gets the latest edit page, restarts ComfyUI, re-enables phone access and
# prints the edit page link. setup.sh installs it; by hand it's installed as
# /usr/local/bin/edit-restart by:
#   sudo curl -fsSL https://raw.githubusercontent.com/jsiggins40/locally-uncensored/rv6-mobile-editor/tools/rv6-mobile/edit-restart.sh -o /usr/local/bin/edit-restart && sudo chmod +x /usr/local/bin/edit-restart
SRC="https://raw.githubusercontent.com/jsiggins40/locally-uncensored/rv6-mobile-editor/tools/rv6-mobile"
DEST="$HOME/ComfyUI/custom_nodes/rv6_mobile"
# One at a time: when two run together (the boot-time auto-start and an
# installer, say) they trip over each other's downloads and ComfyUI restarts.
# The second waits for the first, at most 10 minutes. (The lock is held across the
# self-update below; tmux gets "9>&-" so the tmux server it may start doesn't keep
# the lock after this ends. v2: the first version's lock could stay held that way.)
if [ -z "${RV6M_LOCKED:-}" ] && command -v flock >/dev/null; then
  exec 9>"/tmp/rv6m-edit-restart-v2-$(id -u).lock"
  if ! flock -n 9; then
    echo "Another edit-restart is running; waiting for it to finish…"
    flock -w 600 9 || echo "  (still busy after 10 minutes: going ahead anyway)"
  fi
  export RV6M_LOCKED=1
fi
echo "Getting the latest page"
# GitHub's raw files are cached for up to 5 minutes by branch name; asking for
# the newest commit by its id always gets the newest files
sha=$(curl -fsSL --max-time 15 https://api.github.com/repos/jsiggins40/locally-uncensored/commits/rv6-mobile-editor 2>/dev/null \
      | python3 -c 'import json,sys; print(json.load(sys.stdin)["sha"])' 2>/dev/null)
[ -n "$sha" ] && SRC="https://raw.githubusercontent.com/jsiggins40/locally-uncensored/$sha/tools/rv6-mobile"
mkdir -p "$DEST/web"
ok=1
for f in __init__.py web/edit.html web/workflow.json web/lift.js; do
  curl -fsSL "$SRC/${f#web/}" -o "$DEST/$f.new" && mv "$DEST/$f.new" "$DEST/$f" || { ok=0; rm -f "$DEST/$f.new"; }
done
[ $ok = 1 ] || echo "  (couldn't download all of it; keeping the current page)"
echo "  page version: $(grep -o "PAGE_BUILD = '[^']*'" "$DEST/web/edit.html" | head -1 | cut -d"'" -f2)${sha:+ (commit ${sha:0:7})}"
# ✂️ Lift's models and runtime (about 50 MB, once): served from the VM, so lifting
# doesn't depend on outside sites. Without them the page loads them from the
# internet instead.
LIFT="$DEST/web/lift"; LIFT_V=1.1.1
if [ "$(cat "$LIFT/VERSION" 2>/dev/null)" != "$LIFT_V" ]; then
  echo "Getting the ✂️ Lift models (once, about 50 MB)"
  tmp=$(mktemp -d)
  GCS=https://storage.googleapis.com/mediapipe-models
  if curl -fsSL "https://registry.npmjs.org/@mediapipe/tasks-vision/-/tasks-vision-$LIFT_V.tgz" -o "$tmp/mp.tgz" \
     && tar -xzf "$tmp/mp.tgz" -C "$tmp" package/vision_bundle.mjs package/wasm \
     && curl -fsSL "$GCS/image_segmenter/deeplab_v3/float32/1/deeplab_v3.tflite" -o "$tmp/deeplab_v3.tflite" \
     && curl -fsSL "$GCS/interactive_segmenter/magic_touch/float32/1/magic_touch.tflite" -o "$tmp/magic_touch.tflite"; then
    # .js rather than .mjs: every server sends .js as JavaScript, which import() needs
    rm -rf "$LIFT.new" && mkdir -p "$LIFT.new/wasm" \
      && cp "$tmp/package/vision_bundle.mjs" "$LIFT.new/vision_bundle.js" \
      && cp "$tmp"/package/wasm/* "$LIFT.new/wasm/" \
      && cp "$tmp"/*.tflite "$LIFT.new/" \
      && echo "$LIFT_V" > "$LIFT.new/VERSION" \
      && rm -rf "$LIFT" && mv "$LIFT.new" "$LIFT" \
      || echo "  (couldn't install them; ✂️ Lift will load them from the internet)"
  else
    echo "  (couldn't download them; ✂️ Lift will load them from the internet)"
  fi
  rm -rf "$tmp"
fi
# 🧹 Remove from photo (✂️ Lift): LaMa (big-lama, Apache-2.0) fills in the
# background where a person was. About 200 MB, once; checked against its SHA-256.
LAMA="$HOME/ComfyUI/models/lama/big-lama.pt"
if [ ! -s "$LAMA" ]; then
  echo "Getting the ✂️ Lift remover (once, about 200 MB)"
  mkdir -p "$(dirname "$LAMA")"
  if curl -fsSL https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt -o "$LAMA.new" \
     && echo "344c77bbcb158f17dd143070d1e789f38a66c04202311ae3a258ef66667a9ea9  $LAMA.new" | sha256sum -c --quiet; then
    mv "$LAMA.new" "$LAMA"
  else
    rm -f "$LAMA.new"; echo "  (couldn't get it; 🧹 Remove from photo won't work until the next edit-restart)"
  fi
fi
# keep this command itself up to date too; when it changed, run the new copy
# right away (this run is still the old one)
if sudo curl -fsSL "$SRC/edit-restart.sh" -o /usr/local/bin/edit-restart.new 2>/dev/null; then
  if cmp -s /usr/local/bin/edit-restart.new /usr/local/bin/edit-restart; then
    sudo rm -f /usr/local/bin/edit-restart.new
  else
    sudo chmod +x /usr/local/bin/edit-restart.new && sudo mv /usr/local/bin/edit-restart.new /usr/local/bin/edit-restart
    [ -z "$RV6M_UPDATED" ] && { echo "  (edit-restart updated itself: running the new one)"; RV6M_UPDATED=1 exec /usr/local/bin/edit-restart; }
  fi
fi
# rv6: a short command for the installers and checks (rv6 add-swarmui, rv6 swarm-doctor...),
# so there is no long line to paste
sudo curl -fsSL "$SRC/rv6.sh" -o /usr/local/bin/rv6.new 2>/dev/null && sudo chmod +x /usr/local/bin/rv6.new \
  && sudo mv /usr/local/bin/rv6.new /usr/local/bin/rv6 || sudo rm -f /usr/local/bin/rv6.new
# ffmpeg: shrinks a reference video from the phone before MiniMax H3 reads it (once, small)
command -v ffmpeg >/dev/null 2>&1 || sudo apt-get install -y -qq ffmpeg >/dev/null 2>&1 || true
# HEIC photos from a computer whose browser can't open them are converted on the VM
"$HOME/ComfyUI/venv/bin/python" -c "import pillow_heif" 2>/dev/null \
  || "$HOME/ComfyUI/venv/bin/pip" install -q pillow-heif >/dev/null 2>&1 \
  || echo "  (couldn't install pillow-heif: HEIC photos won't convert)"
tmux kill-session -t comfy 2>/dev/null 9>&-
# ComfyUI's output also goes to ~/comfyui.log (the run before it: ~/comfyui.log.1), so the reason is
# still there after it crashes and its tmux window is gone. It holds no prompts.
[ -s "$HOME/comfyui.log" ] && mv -f "$HOME/comfyui.log" "$HOME/comfyui.log.1"
tmux new -d -s comfy "cd $HOME/ComfyUI && PYTHONUNBUFFERED=1 venv/bin/python main.py --listen 127.0.0.1 --port 8188 2>&1 | tee $HOME/comfyui.log" 9>&-
echo -n "Starting ComfyUI"
for _ in $(seq 90); do curl -sf -o /dev/null localhost:8188 && break; echo -n .; sleep 1; done; echo
curl -sf -o /dev/null localhost:8188 || { echo "ComfyUI did not start. See: tmux attach -t comfy"; exit 1; }
sudo tailscale serve --bg 8188 >/dev/null 2>&1
# The chat page (add-chat.sh), when installed: publish it on :8443 too (it has
# gone missing from Tailscale's settings on some VMs)
[ -x "$HOME/.local/bin/chat-start" ] && sudo tailscale serve --bg --https=8443 http://127.0.0.1:8080 >/dev/null 2>&1
# SwarmUI (add-swarmui.sh), when installed: publish it on :8444, and bring it up if it
# isn't running (after a boot). "9>&-": it must not keep this script's lock.
if [ -x "$HOME/.local/bin/swarm-start" ]; then
  sudo tailscale serve --bg --https=8444 http://127.0.0.1:7801 >/dev/null 2>&1
  curl -sf -o /dev/null -m 3 localhost:7801 || (nohup "$HOME/.local/bin/swarm-start" >/dev/null 2>&1 9>&- &)
fi
host=$(tailscale status --json 2>/dev/null | python3 -c 'import json,sys; print(json.load(sys.stdin)["Self"]["DNSName"].rstrip("."))' 2>/dev/null)
if [ -z "$host" ]; then echo "Tailscale is not logged in: sudo tailscale up --reset"; exit 1; fi
echo "OPEN: https://$host/extensions/rv6_mobile/edit.html"
