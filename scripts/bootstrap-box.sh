#!/usr/bin/env bash
#
# A whole box from nothing, in one command.
#
#   bash b.sh                 everything
#   SKIP_COMFY=1 bash b.sh    Forge and Klein only
#   SKIP_FORGE=1 bash b.sh    no Forge/Klein (saves 60GB)
#   SKIP_QWEN=1 bash b.sh     no Qwen editor (saves 28GB)
#   SKIP_WAN=1 bash b.sh      no video (saves 45GB)
#   SKIP_WRITER=1 bash b.sh   no LLM (saves 25GB)
#   SKIP_TRAINER=1 bash b.sh  no LoRA training
#
# Runs, in order: Forge Neo with Klein and its uncensored encoder; ComfyUI
# with the models symlinked across; the Qwen editor; Wan video; the photo
# inbox; the edit form; the local LLM and its document tooling; the LoRA
# trainer. Then prints every port and password in one place.
#
# One command rather than four because these get pasted on a phone keyboard,
# where a truncated URL quietly downloads a web page instead of a script -
# which has now happened twice.
#
# Wants: an A100 (or anything with ~24GB), the disk the chosen parts need
# (230GB for all of it, 170GB with SKIP_FORGE=1), and a Hugging Face token
# already stored. It checks all three before downloading anything.

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
# Some images run as root with sudo not installed at all, where "sudo: not
# found" reads as too few privileges and means the opposite. Work out once
# how to become root, and only give up if neither route exists.
if [ "$(id -u)" = "0" ]; then
  SUDO=""
elif command -v sudo >/dev/null 2>&1 && sudo -n true 2>/dev/null; then
  SUDO="sudo -n"
else
  die "this needs root: run as root, or make sudo passwordless (run 'sudo true' once), then start again"
fi

# What is actually going to be pulled, so the number means something.
NEED=0
[ "${SKIP_FORGE:-}"   = "1" ] || NEED=$((NEED + 60))  # forge, klein, its encoder
[ "${SKIP_COMFY:-}"   = "1" ] || NEED=$((NEED + 15))
[ "${SKIP_QWEN:-}"    = "1" ] || NEED=$((NEED + 30))
[ "${SKIP_WAN:-}"     = "1" ] || NEED=$((NEED + 45))
[ "${SKIP_WRITER:-}"  = "1" ] || NEED=$((NEED + 25))
[ "${SKIP_TRAINER:-}" = "1" ] || NEED=$((NEED + 15))
NEED=$((NEED + 40))                            # room to actually work in
AVAIL=$(df -BG --output=avail "$HOME" | tail -1 | tr -dc '0-9')
echo "disk free: ${AVAIL}G, this run wants about ${NEED}G"
if [ "${AVAIL:-0}" -lt "$NEED" ]; then
  die "only ${AVAIL}G free, and this needs about ${NEED}G. Either resize the
     instance (disk can only be grown, so pick generously - 400G covers
     everything) or leave parts out:
       SKIP_FORGE=1    no Forge/Klein, -60G
       SKIP_WAN=1      no video, -45G
       SKIP_TRAINER=1  no LoRA training, -15G now and -60G later
       SKIP_WRITER=1   no LLM, -25G"
fi

# Checked first because every weight below comes from Hugging Face, and the
# gated uncensored encoder is worse than a failure without it: it silently
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

if [ "${SKIP_FORGE:-}" != "1" ]; then
  say "1/8  Forge Neo, Klein, uncensored encoder"
  fetch setup-forge-neo-klein.sh
  bash "$HOME/setup-forge-neo-klein.sh" || die "forge setup"
  wait_for setup 60
  grep -q 'text encoder is the STOCK one' "$HOME/forge-setup.log" 2>/dev/null \
    && echo "   !! the uncensored encoder did not download; prompts will be refused"
else
  say "1/8  Forge Neo and Klein - skipped"
fi

if [ "${SKIP_COMFY:-}" != "1" ]; then
  say "2/8  ComfyUI"
  fetch add-comfyui.sh
  bash "$HOME/add-comfyui.sh" || echo "   (comfy setup returned an error; continuing)"
  wait_for comfysetup 40
else
  say "2/8  ComfyUI - skipped"
fi

if [ "${SKIP_COMFY:-}" != "1" ] && [ "${SKIP_QWEN:-}" != "1" ]; then
  say "3/8  Qwen image editor"
  # One merged file carrying model, encoder, VAE and the Lightning
  # accelerators. Forge Neo could not drive it; ComfyUI is what it is built
  # for. v19 is the version its author rates best for edit consistency.
  CK="$HOME/ComfyUI/models/checkpoints"
  AIO="$CK/qwen-image-edit-rapid-aio-nsfw-v19.safetensors"
  if [ -s "$AIO" ]; then
    echo "  already have it"
  else
    mkdir -p "$CK"
    if hf download Phr00t/Qwen-Image-Edit-Rapid-AIO \
         v19/Qwen-Rapid-AIO-NSFW-v19.safetensors --local-dir "$CK"; then
      # ComfyUI reads the checkpoints folder flat, and the name says what
      # the file is - the upstream one does not.
      mv "$CK/v19/Qwen-Rapid-AIO-NSFW-v19.safetensors" "$AIO" \
        && rmdir "$CK/v19" 2>/dev/null
      echo "  -> $(basename "$AIO") ($(du -h "$AIO" | cut -f1))"
    else
      echo "  !! could not fetch it; Klein is unaffected"
    fi
  fi

  # An NSFW LoRA for the Plus-series edit models, applied on top rather
  # than merged in - so its strength is a dial instead of a decision
  # someone else made.
  LORAS="$HOME/ComfyUI/models/loras"
  mkdir -p "$LORAS"
  if ls "$LORAS"/*nsfw*.safetensors >/dev/null 2>&1; then
    echo "  nsfw lora already there"
  elif hf download ScottzillaSystems/qwen-image-edit-plus-nsfw-lora \
         --local-dir "$LORAS" --include "*.safetensors" >/dev/null 2>&1; then
    # Downloads can arrive nested; ComfyUI reads the folder flat.
    find "$LORAS" -mindepth 2 -name '*.safetensors' -exec mv -n {} "$LORAS"/ \; 2>/dev/null
    find "$LORAS" -mindepth 1 -type d -empty -delete 2>/dev/null
    echo "  + $(ls "$LORAS" | tr '\n' ' ')"
  else
    echo "  !! could not fetch the nsfw lora"
  fi
fi

if [ "${SKIP_COMFY:-}" != "1" ] && [ "${SKIP_WAN:-}" != "1" ]; then
  say "4/8  Wan 2.2 video"
  fetch add-wan-video.sh
  bash "$HOME/add-wan-video.sh" || echo "   (wan setup returned an error; continuing)"
  wait_for wansetup 60
else
  say "4/8  Wan video - skipped"
fi

say "5/8  photo inbox"
fetch start-photo-inbox.sh
bash "$HOME/start-photo-inbox.sh" || echo "   (inbox failed; continuing)"

if [ "${SKIP_COMFY:-}" != "1" ] && tmux has-session -t =comfy 2>/dev/null; then
  say "6/8  edit form"
  fetch start-edit-form.sh
  bash "$HOME/start-edit-form.sh" || echo "   (edit form failed; continuing)"
else
  say "6/8  edit form - skipped (needs ComfyUI running)"
fi

if [ "${SKIP_WRITER:-}" != "1" ]; then
  say "7/8  the local model, and what turns its text into documents"
  fetch add-writer.sh
  bash "$HOME/add-writer.sh" || echo "   (writer setup returned an error; continuing)"
  fetch start-writer-form.sh
  bash "$HOME/start-writer-form.sh" || echo "   (writer form failed; continuing)"
else
  say "7/8  LLM - skipped"
fi

if [ "${SKIP_TRAINER:-}" != "1" ]; then
  # Last, deliberately: it is the one that can run the disk down, and by
  # here everything else is already serving.
  say "8/8  LoRA trainer"
  fetch add-trainer.sh
  bash "$HOME/add-trainer.sh" || echo "   (trainer setup returned an error; continuing)"
  wait_for trainsetup 40
  fetch start-train-form.sh
  bash "$HOME/start-train-form.sh" || echo "   (train form failed; continuing)"
else
  say "8/8  LoRA trainer - skipped"
fi

# ------------------------------------------------------------------ report
say "everything, in one place"
show() {  # show <label> <port> <credentials-file> <tmux-session>
  local up="down"
  tmux has-session -t "=$4" 2>/dev/null && up="up"
  printf '\n  %-12s port %-6s [%s]\n' "$1" "$2" "$up"
  [ -s "$3" ] && sed 's/^/                /' "$3"
}
[ "${SKIP_FORGE:-}" = "1" ] \
  || show "Forge" 7860 "$HOME/forge-credentials.txt" forge
show "photo inbox" 7861 "$HOME/photo-inbox-credentials.txt" inbox
show "edit form"  7862 "$HOME/edit-form-credentials.txt"  editform
show "ComfyUI"    8188 "$HOME/comfy-credentials.txt"      comfy
show "write"      7863 "$HOME/writer-credentials.txt"    writer
show "train"      7864 "$HOME/train-credentials.txt"     trainform
printf '\n  %-12s %s\n' "ollama" "$(tmux has-session -t =ollama 2>/dev/null && echo 'up on 11434' || echo 'down')"

cat <<EOF


  Forward those ports in the Thunder console, one URL each. 11434 is the
  model's own API and stays private - the writing form on 7863 is the way
  in to it.
EOF

if [ "${SKIP_FORGE:-}" != "1" ]; then
cat <<'EOF'

  Klein, in Forge on 7860:
    checkpoint     flux-2-klein-base-9b
    text encoder   flux2-klein-9b-uncensored-text-encoder   (NOT qwen_3_8b)
    vae            flux2-vae
    steps 30 · CFG 1.0 · Distilled CFG 3.5 · Euler/Simple

  Then, before generating anything: expand "Never OOM Integrated" in the
  Forge settings panel and switch it off. It tiles the VAE, which puts
  seams through every image, and it is on by default.
EOF
fi

echo
echo "  Full log: $LOG"
