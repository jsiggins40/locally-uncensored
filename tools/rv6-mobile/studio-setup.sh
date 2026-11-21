#!/usr/bin/env bash
# Sets up a fresh GPU VM as the "studio" by hand: the same as studio-cloud-init.yaml
# (edit page with ✂️ Lift, MOSS, the chat with web search, auto-start after boot)
# for a VM that was created without it. Asks its two questions first (Tailscale
# login, optional Atlas key), then runs by itself for about an hour.
# Run it inside tmux, so a dropped connection doesn't stop it:
#   tmux new -s setup "bash <(curl -fsSL https://raw.githubusercontent.com/jsiggins40/locally-uncensored/rv6-mobile-editor/tools/rv6-mobile/studio-setup.sh); bash"
# Log: ~/studio-setup.log
SRC=https://raw.githubusercontent.com/jsiggins40/locally-uncensored/rv6-mobile-editor/tools/rv6-mobile
exec > >(tee -a "$HOME/studio-setup.log") 2>&1
# The newest CUDA compiler first: Blackwell GPUs (RTX PRO 6000) need 12.8+ for the chat engine
export PATH="/usr/local/cuda-12.8/bin:/usr/local/cuda/bin:$HOME/.local/bin:$PATH"
failed=()

free=$(df --output=avail -BG "$HOME" | tail -1 | tr -dc 0-9)
echo "== Free disk: ${free} GB"
[ "${free:-0}" -ge 120 ] || echo "   WARNING: everything needs about 100 GB; this may run out of room."

echo "== System packages"
sudo apt-get update -qq
sudo apt-get install -y -qq python3-venv tmux git curl mosh

echo "== Tailscale (the phone's private link), named 'studio'"
command -v tailscale >/dev/null || curl -fsSL https://tailscale.com/install.sh | sh
sudo systemctl enable --now tailscaled >/dev/null 2>&1 || true
if ! tailscale status >/dev/null 2>&1; then
  echo ">>> Open the login link below on your phone and approve it (same account as your phone's Tailscale app)."
  sudo tailscale up --hostname=studio || failed+=("Tailscale")
fi

echo "== Seedream (optional): your Atlas Cloud API key stays on this VM only"
if [ ! -s "$HOME/.atlascloud_key" ]; then
  read -rsp "Paste the Atlas key and press Enter (or just Enter to skip): " k </dev/tty || k=""
  echo
  k=$(printf '%s' "$k" | sed 's/\x1b\[[0-9;]*[A-Za-z~]//g; s/\[20[01]~//g' | tr -d '[:space:][:cntrl:]')
  if [ -n "$k" ]; then
    printf '%s' "$k" > "$HOME/.atlascloud_key"; chmod 600 "$HOME/.atlascloud_key"
    echo "   saved (ends in ...${k: -4})"
  else
    echo "   skipped (add it later with seedream-setup.sh)"
  fi
fi
echo
echo ">>> That's all the questions. The rest takes about an hour by itself."
echo ">>> You can leave: press Ctrl+B, then D. Come back with: tmux attach -t setup"
echo

v=$(nvcc --version 2>/dev/null | grep -o 'release [0-9]*\.[0-9]*' | cut -d' ' -f2)
if [ -z "$v" ] || [ "$(printf '%s\n12.8\n' "$v" | sort -V | head -1)" != 12.8 ]; then
  echo "== CUDA compiler 12.8 (CUDA ${v:-none} here)"
  sudo apt-get install -y -qq cuda-toolkit-12-8 >/dev/null || echo "   (couldn't install it; the chat step may fail)"
fi

# setsid: no terminal for setup.sh, so it doesn't stop to ask for the Atlas key again
echo "== Edit page + Lift"
setsid -w bash -c "bash <(curl -fsSL $SRC/setup.sh) lift" </dev/null || failed+=("edit page")
echo "== MOSS"
bash <(curl -fsSL "$SRC/add-moss.sh") || failed+=("MOSS")
echo "== Chat (Qwen 3.8 27B on the GPU next to the image models: Q6, 64K memory)"
CHAT_DEVICE=gpu bash <(curl -fsSL "$SRC/add-chat.sh") || failed+=("chat")
echo "== Web search for the chat"
bash <(curl -fsSL "$SRC/add-search.sh") || failed+=("search")
echo "== Auto-start after every boot"
bash <(curl -fsSL "$SRC/add-autostart.sh") || failed+=("auto-start")

HOST=$(tailscale status --json 2>/dev/null | python3 -c 'import json,sys; print(json.load(sys.stdin)["Self"]["DNSName"].rstrip("."))' 2>/dev/null || true)
echo
echo "=============================================================="
if [ ${#failed[@]} -eq 0 ]; then echo " STUDIO READY"; else echo " FINISHED, BUT THESE FAILED: ${failed[*]} (see ~/studio-setup.log)"; fi
echo " Edit page: https://${HOST:-studio.<tailnet>.ts.net}/extensions/rv6_mobile/edit.html"
echo " Chat:      https://${HOST:-studio.<tailnet>.ts.net}:8443/"
echo " After a restart in the Nebius console, both come back by themselves."
echo "=============================================================="
