#!/usr/bin/env bash
# Starts the edit page (edit-restart) and the chat (chat-start) by themselves
# every time the VM boots, so starting the VM in the Nebius console is enough.
# Uses a systemd service: some images don't let normal users use cron.
# Only starts what is installed on this VM. Re-run any time; undo with:
#   sudo systemctl disable --now rv6m-autostart
#   bash <(curl -fsSL https://raw.githubusercontent.com/jsiggins40/locally-uncensored/rv6-mobile-editor/tools/rv6-mobile/add-autostart.sh)
set -euo pipefail
me=$(id -un)
cmds=()
[ -x /usr/local/bin/edit-restart ] && cmds+=("/usr/local/bin/edit-restart")
[ -x "$HOME/.local/bin/chat-start" ] && cmds+=("$HOME/.local/bin/chat-start")
if [ ${#cmds[@]} -eq 0 ]; then echo "Nothing to start yet: install the edit page or the chat first."; exit 1; fi
run=""
for c in "${cmds[@]}"; do run+="$c >> $HOME/autostart.log 2>&1; "; done

sudo tee /etc/systemd/system/rv6m-autostart.service >/dev/null <<EOF
[Unit]
Description=Start the edit page and/or the chat after boot (rv6-mobile)
After=network-online.target tailscaled.service
Wants=network-online.target

[Service]
Type=oneshot
RemainAfterExit=yes
User=$me
Environment=HOME=$HOME
Environment=PATH=/usr/local/bin:/usr/bin:/bin
ExecStartPre=/bin/sleep 20
ExecStart=/bin/bash -c ': > $HOME/autostart.log; $run'
KillMode=process
TimeoutStartSec=0

[Install]
WantedBy=multi-user.target
EOF
sudo systemctl daemon-reload
sudo systemctl enable rv6m-autostart >/dev/null 2>&1
echo "Auto-start is ON for $me: after every boot it runs: ${cmds[*]}"
echo "(log: ~/autostart.log)"
