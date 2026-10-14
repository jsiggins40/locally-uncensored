#!/usr/bin/env bash
# Gets the latest edit page, restarts ComfyUI, re-enables phone access and
# prints the edit page link. setup.sh installs it; by hand it's installed as
# /usr/local/bin/edit-restart by:
#   sudo curl -fsSL https://raw.githubusercontent.com/jsiggins40/locally-uncensored/rv6-mobile-editor/tools/rv6-mobile/edit-restart.sh -o /usr/local/bin/edit-restart && sudo chmod +x /usr/local/bin/edit-restart
SRC="https://raw.githubusercontent.com/jsiggins40/locally-uncensored/rv6-mobile-editor/tools/rv6-mobile"
DEST="$HOME/ComfyUI/custom_nodes/rv6_mobile"
echo "Getting the latest page"
# GitHub's raw files are cached for up to 5 minutes by branch name; asking for
# the newest commit by its id always gets the newest files
sha=$(curl -fsSL --max-time 15 https://api.github.com/repos/jsiggins40/locally-uncensored/commits/rv6-mobile-editor 2>/dev/null \
      | python3 -c 'import json,sys; print(json.load(sys.stdin)["sha"])' 2>/dev/null)
[ -n "$sha" ] && SRC="https://raw.githubusercontent.com/jsiggins40/locally-uncensored/$sha/tools/rv6-mobile"
mkdir -p "$DEST/web"
ok=1
for f in __init__.py web/edit.html web/workflow.json; do
  curl -fsSL "$SRC/${f#web/}" -o "$DEST/$f.new" && mv "$DEST/$f.new" "$DEST/$f" || { ok=0; rm -f "$DEST/$f.new"; }
done
[ $ok = 1 ] || echo "  (couldn't download all of it; keeping the current page)"
echo "  page version: $(grep -o "PAGE_BUILD = '[^']*'" "$DEST/web/edit.html" | head -1 | cut -d"'" -f2)${sha:+ (commit ${sha:0:7})}"
# keep this command itself up to date too
sudo curl -fsSL "$SRC/edit-restart.sh" -o /usr/local/bin/edit-restart.new 2>/dev/null \
  && sudo chmod +x /usr/local/bin/edit-restart.new && sudo mv /usr/local/bin/edit-restart.new /usr/local/bin/edit-restart
tmux kill-session -t comfy 2>/dev/null
tmux new -d -s comfy "cd $HOME/ComfyUI && venv/bin/python main.py --listen 127.0.0.1 --port 8188"
echo -n "Starting ComfyUI"
for _ in $(seq 90); do curl -sf -o /dev/null localhost:8188 && break; echo -n .; sleep 1; done; echo
curl -sf -o /dev/null localhost:8188 || { echo "ComfyUI did not start. See: tmux attach -t comfy"; exit 1; }
sudo tailscale serve --bg 8188 >/dev/null 2>&1
host=$(tailscale status --json 2>/dev/null | python3 -c 'import json,sys; print(json.load(sys.stdin)["Self"]["DNSName"].rstrip("."))' 2>/dev/null)
if [ -z "$host" ]; then echo "Tailscale is not logged in: sudo tailscale up --reset"; exit 1; fi
echo "OPEN: https://$host/extensions/rv6_mobile/edit.html"
