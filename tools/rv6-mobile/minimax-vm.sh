#!/usr/bin/env bash
# One command for a fresh Ubuntu CUDA GPU VM (48 GB of GPU memory or more): the phone
# edit page with ✂️ Lift, SwarmUI, MiniMax H3 Image (pictures and photo edits),
# MiniMax H3 (video with sound) and their uncensored extras, reachable from the phone
# and the laptop over Tailscale, and started again by itself after every boot.
# It asks two things at the start (the MiniMax licence note, and a Tailscale login link
# to open on the phone), then runs by itself. About 110 GB and 1-2 hours, mostly
# downloads. Safe to run again: each part skips what it already has.
#   "ref"   also gets the reference-to-video model (21 GB)
#   "full"  gets the full 8-bit video models instead of the pruned ones
#   "all"   also gets the single-act video LoRAs
# Run it in tmux so a dropped connection doesn't stop it. Type these four lines:
#   U=https://raw.githubusercontent.com/jsiggins40
#   U=$U/locally-uncensored/rv6-mobile-editor
#   U=$U/tools/rv6-mobile
#   tmux new -s setup "bash <(curl -fsSL $U/minimax-vm.sh); bash"
# The VM's Tailscale name is "h200" (another one: VM_NAME=name in front of bash).
# Log: ~/minimax-vm.log
SRC="https://raw.githubusercontent.com/jsiggins40/locally-uncensored/rv6-mobile-editor/tools/rv6-mobile"
exec > >(tee -a "$HOME/minimax-vm.log") 2>&1
export PATH="/usr/local/cuda-12.8/bin:/usr/local/cuda/bin:$HOME/.local/bin:$PATH"
NAME="${VM_NAME:-h200}"

echo "== This VM"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || echo "   (no GPU driver answering yet: the setup step repairs it)"
free=$(df --output=avail -BG "$HOME" | tail -1 | tr -dc 0-9)
echo "   free disk: ${free} GB"
[ "${free:-0}" -ge 130 ] || echo "   WARNING: all of this needs about 110 GB; this disk may run out. Each part checks its own space."

if [ -z "${MINIMAX_LICENCE_OK:-}" ]; then
  echo
  echo "LICENCE NOTE: MiniMax's licence for MiniMax H3 and MiniMax H3 Image covers use worldwide"
  echo "EXCEPT in the European Union, the United Kingdom, South Korea and the United States."
  echo "Full text: https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE"
  read -rp "Type yes to install anyway, anything else to stop: " ok </dev/tty || ok=""
  [ "$ok" = yes ] || { echo "Stopped; nothing was installed."; exit 0; }
fi
export MINIMAX_LICENCE_OK=1

echo
echo "== Tailscale (the private link to the phone and laptop), named '$NAME'"
sudo apt-get update -qq && sudo apt-get install -y -qq curl tmux git >/dev/null
command -v tailscale >/dev/null || curl -fsSL https://tailscale.com/install.sh | sh
sudo systemctl enable --now tailscaled >/dev/null 2>&1 || true
if ! tailscale status >/dev/null 2>&1; then
  echo ">>> Open the login link below on your phone and approve it (same account as the phone's Tailscale app)."
  sudo tailscale up --hostname="$NAME" || echo "   Tailscale didn't come up; the rest still installs. Run again afterwards: sudo tailscale up --hostname=$NAME"
fi

done_=(); failed=()
step() {   # step "name" command...
  echo; echo "################ $1 ################"
  if "${@:2}"; then done_+=("$1"); else failed+=("$1"); fi
}
run() { bash <(curl -fsSL "$SRC/$1") "${@:2}"; }
video_args=(); nsfw_args=()
for a in "$@"; do case "$a" in ref|full) video_args+=("$a") ;; all) nsfw_args+=("$a") ;; esac; done

# setsid: no terminal, so setup.sh doesn't stop to ask for an Atlas key (seedream-setup.sh does that later)
step "ComfyUI, the edit page and ✂️ Lift" setsid -w bash -c "bash <(curl -fsSL $SRC/setup.sh) lift" </dev/null
step "SwarmUI" run add-swarmui.sh
step "MiniMax H3 Image (pictures and photo edits)" run add-minimax-image.sh
step "MiniMax H3 (video)" run add-minimax-video.sh "${video_args[@]}"
step "Uncensored extras: text encoder and LoRAs" run add-minimax-nsfw.sh "${nsfw_args[@]}"
step "Start by itself after every boot" run add-autostart.sh
step "Restart the page with everything" edit-restart

HOST=$(tailscale status --json 2>/dev/null | python3 -c 'import json,sys; print(json.load(sys.stdin)["Self"]["DNSName"].rstrip("."))' 2>/dev/null || true)
echo
echo "=============================================================="
echo " INSTALLED: ${#done_[@]}"; for d in "${done_[@]}"; do echo "   ✓ $d"; done
if [ ${#failed[@]} -gt 0 ]; then
  echo " FAILED (see ~/minimax-vm.log; run this again to retry just these):"; for f in "${failed[@]}"; do echo "   ✗ $f"; done
fi
echo " On the phone or the laptop, with Tailscale connected:"
echo "   Edit page (pictures, 🐝 Swarm): https://${HOST:-<this-vm>.ts.net}/extensions/rv6_mobile/edit.html"
echo "   SwarmUI (video):                https://${HOST:-<this-vm>.ts.net}:8444/"
echo " From now on, short commands work here: rv6 (lists them), rv6 swarm-doctor, edit-restart"
echo " Free disk now: $(df --output=avail -BG "$HOME" | tail -1 | tr -dc 0-9) GB"
echo "=============================================================="
