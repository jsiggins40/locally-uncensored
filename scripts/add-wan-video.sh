#!/usr/bin/env bash
#
# Wan 2.2 image-to-video, into the ComfyUI that is already here.
#
#   bash w.sh
#
# Animating a still you already approved beats text-to-video: the image
# costs seconds to iterate on, the video costs minutes. Get the frame right
# cheaply, then spend the expensive part once.
#
# Wan 2.2 14B is a two-expert model - a high-noise pass and a low-noise one,
# two files, run in sequence over one latent. Both are downloaded; the form
# wires them.
#
# Needs ComfyUI installed (add-comfyui.sh) and about 45GB free.

set -uo pipefail

COMFY="${COMFY_DIR:-$HOME/ComfyUI}"
LOG="$HOME/wan-setup.log"
NEED_GB=45
exec > >(tee -a "$LOG") 2>&1

say() { printf '\n=== %s ===\n' "$*"; }
die() { printf '\nFAILED: %s\n(see %s)\n' "$*" "$LOG"; exit 1; }

[ -d "$COMFY/models" ] || die "no ComfyUI at $COMFY (run add-comfyui.sh first)"

AVAIL=$(df -BG --output=avail "$HOME" | tail -1 | tr -dc '0-9')
echo "free: ${AVAIL}G, want ${NEED_GB}G"
[ "${AVAIL:-0}" -ge "$NEED_GB" ] || die "not enough disk"

if [ -z "${TMUX:-}" ] && command -v tmux >/dev/null; then
  self=$(readlink -f "$0")
  say "re-running inside tmux session 'wansetup'"
  tmux kill-session -t =wansetup 2>/dev/null
  tmux new-session -d -s wansetup "COMFY_DIR='$COMFY' bash '$self'"
  printf '\nDetached. watch: tail -f %s\n\n' "$LOG"
  exit 0
fi

export PATH="$HOME/.local/bin:$PATH"
cd "$COMFY" || die "cd $COMFY"
mkdir -p models/diffusion_models models/text_encoders models/vae models/loras

say "resolving"
python3 -u - "$COMFY" <<'PY' || die "download"
import sys, os
from huggingface_hub import list_repo_files, hf_hub_download

comfy = sys.argv[1]

def grab(repo, dest, must=(), avoid=(), prefer=(), label=""):
    """Download the best match from repo into comfy/models/<dest>."""
    try:
        files = list_repo_files(repo)
    except Exception as e:
        print(f"  - {repo}: cannot list ({type(e).__name__}: {e})")
        return None
    c = [f for f in files if f.endswith(".safetensors")
         and all(m in f.lower() for m in must)
         and not any(a in f.lower() for a in avoid)]
    if not c:
        print(f"  - {repo}: nothing matching {must}")
        print(f"      holds: {[f for f in files if f.endswith('.safetensors')][:6]}")
        return None
    def score(f):
        low = f.lower()
        for i, p in enumerate(prefer):
            if p in low:
                return (i, len(f))
        return (len(prefer), len(f))
    c.sort(key=score)
    name = c[0]
    out = os.path.join(comfy, "models", dest, os.path.basename(name))
    if os.path.exists(out):
        print(f"  = {label or dest}: already have {os.path.basename(name)}")
        return out
    print(f"  + {label or dest}: {name}")
    p = hf_hub_download(repo_id=repo, filename=name,
                        local_dir=os.path.join(comfy, "models", dest))
    if os.path.abspath(p) != os.path.abspath(out):
        os.replace(p, out)   # flatten split_files/... into the folder itself
    print(f"    -> {out} ({os.path.getsize(out)/2**30:.1f} GiB)")
    return out

REPO = "Comfy-Org/Wan_2.2_ComfyUI_Repackaged"
ok = True

# The two experts. fp8 rather than bf16: half the disk, and on a card this
# size the quality difference is not what limits these clips.
for tag in ("high_noise", "low_noise"):
    if not grab(REPO, "diffusion_models",
                must=("i2v", "14b", tag), avoid=("bf16",),
                prefer=("fp8_scaled", "fp8"), label=f"{tag} expert"):
        ok = False

if not grab(REPO, "text_encoders", must=("umt5",), avoid=(),
            prefer=("fp8_scaled", "fp8"), label="text encoder"):
    ok = False

# Wan 2.2 14B reuses the 2.1 VAE; only the 5B model has its own.
if not grab(REPO, "vae", must=("vae",), avoid=("2.2",),
            prefer=("wan_2.1", "wan2.1"), label="vae"):
    ok = False

sys.exit(0 if ok else 1)
PY

# Uncensored LoRAs live mostly on Civitai, which needs an API key; these are
# the ones reachable without one. Missing them is not fatal - the base model
# animates fine, it just will not do what it has not been taught.
say "uncensored loras (best effort)"
for repo in "Remade-AI/Wan2.2-NSFW-LoRA" "svjack/Wan2_2_NSFW_LoRA"; do
  if hf download "$repo" --local-dir models/loras --include "*.safetensors" >/dev/null 2>&1; then
    echo "  + $repo"
  else
    echo "  - $repo unavailable"
  fi
done
find models/loras -mindepth 2 -name '*.safetensors' -exec mv -n {} models/loras/ \; 2>/dev/null
find models/loras -mindepth 1 -type d -empty -delete 2>/dev/null

say "what is installed"
for d in diffusion_models text_encoders vae loras; do
  printf '  %-18s %s\n' "$d" "$(find "models/$d" -maxdepth 1 -name '*.safetensors' | wc -l)"
done
find models/diffusion_models -maxdepth 1 -name '*wan*' -printf '    %f\n' 2>/dev/null

cat <<'EOF'

=== done ===

Restart ComfyUI so it sees the new files, then restart the form:

  tmux kill-session -t =comfy
  cd ~/ComfyUI && tmux new -d -s comfy "./venv/bin/python main.py --listen 127.0.0.1 --port 8288 2>&1 | tee -a ~/comfy-run.log"

then, once it answers:

  bash ~/e.sh

The form gains a Video mode: pick an image, describe the motion, set a
length. Start at 49 frames (about 3 seconds at 16fps) - identity holds
over short clips and drifts over long ones.
EOF
