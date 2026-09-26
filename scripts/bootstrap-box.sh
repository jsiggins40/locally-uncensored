#!/usr/bin/env bash
#
# A whole box from nothing, in one command.
#
#   bash b.sh              everything
#   SKIP_COMFY=1 bash b.sh  Forge and Klein only
#
# Runs, in order: Forge Neo with Klein and its uncensored encoder; ComfyUI
# with the models symlinked across; the photo inbox; the edit form. Then
# prints every port and password in one place.
#
# One command rather than four because these get pasted on a phone keyboard,
# where a truncated URL quietly downloads a web page instead of a script -
# which has now happened twice.
#
# Wants: an A100 (or anything with ~24GB), 250GB of disk, and a Hugging Face
# token already stored. It checks all three before downloading anything.

set -uo pipefail

REF="${REF:-claude/previous-chat-link-k3xjlk}"
RAW="https://raw.githubusercontent.com/jsiggins40/locally-uncensored/$REF/scripts"
LOG="$HOME/bootstrap.log"
exec > >(tee -a "$LOG") 2>&1

say()  { printf '\n\n======== %s ========\n' "$*"; }
step() { printf '\n-- %s\n' "$*"; }
die()  { printf '\nSTOPPED: %s\n(see %s)\n' "$*" "$LOG"; exit 1; }

say "checks"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || die "no GPU"
sudo -n true 2>/dev/null || die "sudo wants a password; run 'sudo true' once, then re-run"

AVAIL=$(df -BG --output=avail "$HOME" | tail -1 | tr -dc '0-9')
echo "disk free: ${AVAIL}G"
if [ "${AVAIL:-0}" -lt 150 ]; then
  die "only ${AVAIL}G free. Klein and Qwen together need about 120G plus room
     to work. Resize the instance to 250G first - disk can only be grown,
     so pick generously."
fi

# Checked first because without it the gated uncensored encoder silently
# falls back to the stock one, and you find out three prompts into a session.
step "hugging face token"
export PATH="$HOME/.local/bin:$PATH"
command -v hf >/dev/null || pip install -q --break-system-packages "huggingface_hub[cli]" 2>/dev/null \
  || pip install -q "huggingface_hub[cli]" 2>/dev/null
if ! python3 -c 'from huggingface_hub import whoami; print("  as", whoami()["name"])' 2>/dev/null; then
  cat <<'EOF'

  No Hugging Face token. Stop here and run exactly this one line, then
  paste ONLY the hf_... key at the prompt:

    read -p "token: " K && mkdir -p ~/.cache/huggingface && echo -n "$K" > ~/.cache/huggingface/token && echo "stored ${#K} chars"

  Then start this script again.
EOF
  die "no token"
fi

fetch() {  # fetch <name>
  curl -fsSL "$RAW/$1" -o "$HOME/$1" || die "could not fetch $1"
  head -1 "$HOME/$1" | grep -q '^#!/usr/bin/env bash' \
    || { rm -f "$HOME/$1"; die "$1 came back as something else (truncated URL?)"; }
}

wait_for() {  # wait_for <tmux-session> <minutes>
  local s=$1 mins=$2 i=0
  while tmux has-session -t "=$s" 2>/dev/null; do
    sleep 10
    i=$((i + 1))
    [ $((i % 6)) -eq 0 ] && printf '   ... %s running, %d min\n' "$s" $((i / 6))
    [ "$i" -gt $((mins * 6)) ] && { echo "   (giving up on $s after $mins min)"; return 1; }
  done
  return 0
}

say "1/4  Forge Neo, Klein, uncensored encoder"
fetch setup-forge-neo-klein.sh
bash "$HOME/setup-forge-neo-klein.sh" || die "forge setup"
wait_for setup 60
grep -q 'text encoder is the STOCK one' "$HOME/forge-setup.log" 2>/dev/null \
  && echo "   !! the uncensored encoder did not download; prompts will be refused"

if [ "${SKIP_COMFY:-}" != "1" ]; then
  say "2/4  ComfyUI"
  fetch add-comfyui.sh
  bash "$HOME/add-comfyui.sh" || echo "   (comfy setup returned an error; continuing)"
  wait_for comfysetup 40
else
  say "2/4  ComfyUI - skipped"
fi

say "3/4  photo inbox"
fetch start-photo-inbox.sh
bash "$HOME/start-photo-inbox.sh" || echo "   (inbox failed; continuing)"

if [ "${SKIP_COMFY:-}" != "1" ] && tmux has-session -t =comfy 2>/dev/null; then
  say "4/4  edit form"
  fetch start-edit-form.sh
  bash "$HOME/start-edit-form.sh" || echo "   (edit form failed; continuing)"
else
  say "4/4  edit form - skipped (needs ComfyUI running)"
fi

# ------------------------------------------------------------------ report
say "everything, in one place"
show() {  # show <label> <port> <credentials-file> <tmux-session>
  local up="down"
  tmux has-session -t "=$4" 2>/dev/null && up="up"
  printf '\n  %-12s port %-6s [%s]\n' "$1" "$2" "$up"
  [ -s "$3" ] && sed 's/^/                /' "$3"
}
show "Forge"      7860 "$HOME/forge-credentials.txt"      forge
show "photo inbox" 7861 "$HOME/photo-inbox-credentials.txt" inbox
show "edit form"  7862 "$HOME/edit-form-credentials.txt"  editform
show "ComfyUI"    8188 "$HOME/comfy-credentials.txt"      comfy

cat <<EOF


  Forward those ports in the Thunder console, one URL each.

  Klein, in Forge on 7860:
    checkpoint     flux-2-klein-base-9b
    text encoder   flux2-klein-9b-uncensored-text-encoder   (NOT qwen_3_8b)
    vae            flux2-vae
    steps 30 · CFG 1.0 · Distilled CFG 3.5 · Euler/Simple

  Then, before generating anything: expand "Never OOM Integrated" in the
  Forge settings panel and switch it off. It tiles the VAE, which puts
  seams through every image, and it is on by default.

  Full log: $LOG
EOF
