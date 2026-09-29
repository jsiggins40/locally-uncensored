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

# Without tmux this runs in the foreground and dies with the terminal,
# and the "detached, you can close this" habit becomes a trap. It is a
# few hundred kilobytes; install it rather than silently not detaching.
command -v tmux >/dev/null || {
  echo "installing tmux so this can detach"
  if [ "$(id -u)" = "0" ]; then AS=""; else AS="sudo -n"; fi
  $AS apt-get update -qq 2>/dev/null
  $AS apt-get install -y -qq tmux 2>/dev/null || true
}

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
# torchaudio too: ai-toolkit imports it on startup and does not ask for it.
./venv/bin/pip install -q torch torchvision torchaudio 2>&1 | tail -3
./venv/bin/python -c 'import torch;print("  torch",torch.__version__,"cuda",torch.cuda.is_available())' \
  || die "torch did not install"
echo "  the rest"
./venv/bin/pip install -q -r requirements.txt 2>&1 | tail -5
./venv/bin/python -c 'import diffusers, transformers, peft; print("  diffusers", diffusers.__version__)' \
  || die "requirements did not install"

# opencv-python links against a desktop graphics library that a headless
# box has no reason to carry, and the import blows up on libGL.so.1 the
# first time a training run touches it - which is well after you have
# uploaded photos and pressed go. Settle it now.
if ! ./venv/bin/python -c 'import cv2' 2>/dev/null; then
  echo "  opencv needs libGL; installing it"
  if [ "$(id -u)" = "0" ]; then SUDO=""
  elif command -v sudo >/dev/null 2>&1 && sudo -n true 2>/dev/null; then SUDO="sudo -n"
  else SUDO="false"; fi
  $SUDO apt-get install -y -qq libgl1 libglib2.0-0 2>/dev/null \
    || $SUDO apt-get install -y -qq libgl1-mesa-glx libglib2.0-0 2>/dev/null
  if ! ./venv/bin/python -c 'import cv2' 2>/dev/null; then
    # The headless build of the same library wants none of it, and
    # nothing here draws to a screen anyway.
    echo "  no luck; using the headless opencv instead"
    ./venv/bin/pip uninstall -y -q opencv-python opencv-python-headless 2>/dev/null
    ./venv/bin/pip install -q opencv-python-headless 2>&1 | tail -2
  fi
fi
./venv/bin/python -c 'import cv2; print("  cv2", cv2.__version__)' \
  || die "opencv will not import, and the trainer needs it"

say "can the trainer start"
# Its requirements file is not the whole story - the import chain reaches
# torchaudio and opencv, neither of which is listed. Rather than name them
# one at a time as each new one surfaces, do the import the trainer does
# at startup and install whatever it complains about. Ninety minutes into
# a run is a bad time to find the next one.
for _ in 1 2 3 4 5 6; do
  MISSING=$(./venv/bin/python -c "from jobs import ExtensionJob" 2>&1 \
            | sed -n "s/.*No module named '\([^']*\)'.*/\1/p" | head -1)
  [ -z "$MISSING" ] && break
  # A few import names differ from what you install to get them.
  case "$MISSING" in
    cv2)     PKG=opencv-python-headless ;;
    PIL)     PKG=pillow ;;
    yaml)    PKG=pyyaml ;;
    skimage) PKG=scikit-image ;;
    *)       PKG="$MISSING" ;;
  esac
  echo "  missing $MISSING; installing $PKG"
  ./venv/bin/pip install -q "$PKG" 2>&1 | tail -1
done
./venv/bin/python -c "from jobs import ExtensionJob" 2>/dev/null \
  && echo "  the trainer imports cleanly" \
  || die "the trainer still will not start:
$(./venv/bin/python -c 'from jobs import ExtensionJob' 2>&1 | tail -4)"

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
