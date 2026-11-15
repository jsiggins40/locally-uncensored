#!/usr/bin/env bash
# Installs every uncensored model and LoRA the edit page supports, one after
# another, on a VM that already has the edit page (setup.sh or studio-setup.sh):
#   1. Qwen-Image-Edit 2511 + the Scottzilla NSFW LoRA, ⚡ Rapid AIO (NSFW build),
#      ✍️ Qwen-Image 2512 (Create) and the 🔓 abliterated text encoder (setup.sh)
#   2. All the Qwen 2511 LoRAs for Describe edit, art styles included (add-qwen-loras.sh)
#   3. Qwen-Image 2.1: Official, 🔓 Noct Q, 🔞 Omni, the Heretic encoder (add-qwen21.sh)
#   4. Qwen-Image 2.1 ⚡ Turbo (add-qwen21-turbo.sh)
#   5. Krea 2 Turbo + the 🔓 realism LoRA, and Krea 2 photo editing (add-krea2*.sh)
#   6. MOSS-Image + its uncensored and edit LoRAs (add-moss.sh)
# About 300 GB and a few hours, mostly downloads. Each installer skips what it
# already has and tests its models on the GPU, so it's safe to run again.
# 🎬 Video (75 GB) and ⬆️ Upscale (34 GB) too:  ALL_EXTRAS=1 bash <(curl ...)
# Run it in tmux so a dropped connection doesn't stop it:
#   tmux new -s everything "bash <(curl -fsSL https://raw.githubusercontent.com/jsiggins40/locally-uncensored/rv6-mobile-editor/tools/rv6-mobile/add-everything.sh); bash"
# Log: ~/everything.log
SRC="https://raw.githubusercontent.com/jsiggins40/locally-uncensored/rv6-mobile-editor/tools/rv6-mobile"
exec > >(tee -a "$HOME/everything.log") 2>&1
export PATH="/usr/local/cuda-12.8/bin:/usr/local/cuda/bin:$HOME/.local/bin:$PATH"
[ -d "$HOME/ComfyUI" ] || { echo "No ComfyUI here: run setup.sh (or studio-setup.sh) first."; exit 1; }

free=$(df --output=avail -BG "$(readlink -f "$HOME/ComfyUI/models")" | tail -1 | tr -dc 0-9)
echo "== Free disk: ${free} GB"
if [ "${free:-0}" -lt 320 ]; then
  echo "   Everything needs about 300 GB (more with ALL_EXTRAS); this disk may run out."
  echo "   Each installer checks for its own space and skips itself if there isn't enough."
fi

done_=(); failed=()
step() {   # step "name" command...
  echo; echo "################ $1 ################"
  if "${@:2}"; then done_+=("$1"); else failed+=("$1"); fi
}
run() { bash <(curl -fsSL "$SRC/$1") "${@:2}"; }

# setsid: no terminal, so setup.sh doesn't stop to ask for an Atlas key (seedream-setup.sh does that)
step "Qwen 2511 + NSFW LoRA, Rapid AIO, Create 2512, abliterated encoder" \
  setsid -w bash -c "bash <(curl -fsSL $SRC/setup.sh)" </dev/null
step "Qwen 2511 LoRAs (all, styles too)" run add-qwen-loras.sh styles
step "Qwen 2.1: Official, Noct Q, Omni, Heretic encoder" run add-qwen21.sh
step "Qwen 2.1 Turbo" run add-qwen21-turbo.sh
step "Krea 2 Turbo + uncensored realism LoRA" run add-krea2.sh
step "Krea 2 photo editing" run add-krea2-edit.sh
step "MOSS-Image + uncensored and edit LoRAs" run add-moss.sh
if [ -n "${ALL_EXTRAS:-}" ]; then
  step "Video (Wan 2.2)" run add-video.sh
  step "Upscale (SeedVR2)" run add-upscaler.sh
fi
step "Restart the page with everything" edit-restart

HOST=$(tailscale status --json 2>/dev/null | python3 -c 'import json,sys; print(json.load(sys.stdin)["Self"]["DNSName"].rstrip("."))' 2>/dev/null || true)
echo
echo "=============================================================="
echo " INSTALLED: ${#done_[@]}"; for d in "${done_[@]}"; do echo "   ✓ $d"; done
if [ ${#failed[@]} -gt 0 ]; then
  echo " FAILED (see ~/everything.log; run again to retry just these):"; for f in "${failed[@]}"; do echo "   ✗ $f"; done
fi
echo " Edit page: https://${HOST:-<your-vm>.ts.net}/extensions/rv6_mobile/edit.html"
echo " Free disk now: $(df --output=avail -BG "$(readlink -f "$HOME/ComfyUI/models")" | tail -1 | tr -dc 0-9) GB"
echo "=============================================================="
