#!/usr/bin/env bash
#
# Train a LoRA on one person's face, on the box you already have.
#
#   bash t.sh
#
# Prompt tricks get you a face that is nearly right. A LoRA trained on
# twenty photos of one person gets you that person, across poses and
# lighting and in pictures with someone else in them. It is the only thing
# that actually fixes identity rather than nudging it.
#
# ai-toolkit does the training. It wants its own environment - it pins
# torch versions that would fight with ComfyUI's - so it gets one.
#
# Needs an hour or two of GPU time per LoRA, about 80GB of disk for the
# trainer and the unquantised base model, and at least 32GB of VRAM.

set -uo pipefail

DIR="${TRAINER_DIR:-$HOME/ai-toolkit}"
LOG="$HOME/trainer-setup.log"
REPO="https://github.com/ostris/ai-toolkit.git"
NEED_GB=80
exec > >(tee -a "$LOG") 2>&1

say() { printf '\n=== %s ===\n' "$*"; }
die() { printf '\nFAILED: %s\n(see %s)\n' "$*" "$LOG"; exit 1; }

say "the card"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || die "no GPU"
VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
VRAM=${VRAM:-0}

# Qwen-Image-Edit is a 20B model. What fits decides how it has to be
# squeezed, and squeezing it harder costs quality - so pick the lightest
# squeeze that fits rather than the one that always works.
if [ "$VRAM" -ge 48000 ]; then
  PROFILE=8bit
elif [ "$VRAM" -ge 30000 ]; then
  PROFILE=3bit
else
  PROFILE=3bit
  echo "  !! ${VRAM}MB is below what this needs. It is set up anyway, but"
  echo "     expect it to run out of memory. 32GB is the realistic floor."
fi
echo "  ${VRAM}MB -> $PROFILE profile"

AVAIL=$(df -BG --output=avail "$HOME" | tail -1 | tr -dc '0-9')
echo "  disk free: ${AVAIL}G, want ${NEED_GB}G"
[ "${AVAIL:-0}" -ge "$NEED_GB" ] || die "not enough disk - the base model alone is about 60G"

if [ -z "${TMUX:-}" ] && command -v tmux >/dev/null; then
  self=$(readlink -f "$0")
  say "re-running inside tmux session 'trainsetup'"
  tmux kill-session -t =trainsetup 2>/dev/null
  tmux new-session -d -s trainsetup "TRAINER_DIR='$DIR' bash '$self'"
  printf '\nDetached. watch: tail -n 40 %s\n\n' "$LOG"
  exit 0
fi

say "ai-toolkit"
if [ -d "$DIR/.git" ]; then
  echo "  already cloned; updating"
  git -C "$DIR" pull --ff-only 2>&1 | tail -2
else
  git clone --depth 1 "$REPO" "$DIR" || die "clone failed"
fi
cd "$DIR" || die "cd $DIR"

say "its own python"
if [ ! -x "$DIR/venv/bin/python" ]; then
  python3 -m venv venv || die "could not make a venv"
fi
./venv/bin/pip install -q --upgrade pip setuptools wheel 2>&1 | tail -2
# Plain PyPI rather than a pinned CUDA index: the pinned one has repeatedly
# resolved to a torch too old for what these repos annotate.
echo "  torch (this is the slow part)"
./venv/bin/pip install -q torch torchvision 2>&1 | tail -3
./venv/bin/python -c 'import torch;print("  torch",torch.__version__,"cuda",torch.cuda.is_available())' \
  || die "torch did not install"
echo "  the rest"
./venv/bin/pip install -q -r requirements.txt 2>&1 | tail -5
./venv/bin/python -c 'import diffusers, transformers, peft; print("  diffusers", diffusers.__version__)' \
  || die "requirements did not install"

# adamw8bit needs bitsandbytes and it is not always pulled in.
./venv/bin/python -c 'import bitsandbytes' 2>/dev/null \
  || ./venv/bin/pip install -q bitsandbytes 2>&1 | tail -2

say "which base model"
# 2511 if it is published, 2509 otherwise. Both are the "plus" architecture,
# which is the one that takes more than one reference image.
BASE=""
for cand in Qwen/Qwen-Image-Edit-2511 Qwen/Qwen-Image-Edit-2509; do
  if ./venv/bin/python - "$cand" <<'PY'
import sys
from huggingface_hub import list_repo_files
try:
    list_repo_files(sys.argv[1])
except Exception:
    sys.exit(1)
PY
  then BASE="$cand"; break; fi
  echo "  - $cand is not reachable"
done
[ -n "$BASE" ] || die "neither Qwen edit repo could be reached (hugging face token set?)"
echo "  $BASE"

mkdir -p "$HOME/training/datasets" "$HOME/training/output"
printf '%s\n%s\n' "$BASE" "$PROFILE" > "$HOME/.trainer-profile"

say "done"
cat <<EOF

  trainer    $DIR
  base       $BASE
  profile    $PROFILE  (${VRAM}MB of VRAM)
  datasets   $HOME/training/datasets
  output     $HOME/training/output

The base model itself downloads on the first training run, not now - it
is about 60G and there is no point fetching it until you have photos.

Now start the training form:

  curl -fsSL https://raw.githubusercontent.com/jsiggins40/locally-uncensored/claude/previous-chat-link-k3xjlk/scripts/start-train-form.sh -o ~/t2.sh && bash ~/t2.sh

Full log: $LOG
EOF
