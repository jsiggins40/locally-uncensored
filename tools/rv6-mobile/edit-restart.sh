#!/usr/bin/env bash
# Restarts ComfyUI, re-enables phone access and prints the edit page link.
# Installed as /usr/local/bin/edit-restart by:
#   sudo curl -fsSL https://raw.githubusercontent.com/jsiggins40/locally-uncensored/rv6-mobile-editor/tools/rv6-mobile/edit-restart.sh -o /usr/local/bin/edit-restart && sudo chmod +x /usr/local/bin/edit-restart
tmux kill-session -t comfy 2>/dev/null
tmux new -d -s comfy "cd $HOME/ComfyUI && venv/bin/python main.py --listen 127.0.0.1 --port 8188"
echo -n "Starting ComfyUI"
for _ in $(seq 90); do curl -sf -o /dev/null localhost:8188 && break; echo -n .; sleep 1; done; echo
curl -sf -o /dev/null localhost:8188 || { echo "ComfyUI did not start. See: tmux attach -t comfy"; exit 1; }
sudo tailscale serve --bg 8188 >/dev/null 2>&1
host=$(tailscale status --json 2>/dev/null | python3 -c 'import json,sys; print(json.load(sys.stdin)["Self"]["DNSName"].rstrip("."))' 2>/dev/null)
if [ -z "$host" ]; then echo "Tailscale is not logged in: sudo tailscale up --reset"; exit 1; fi
echo "OPEN: https://$host/extensions/rv6_mobile/edit.html"
