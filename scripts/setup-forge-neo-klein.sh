#!/usr/bin/env bash
#
# Forge Neo + uncensored FLUX.2 Klein 9B, one command.
#
# Written for a Thunder Compute A100 box driven from an iPhone, where pasting
# more than one line at a time corrupts the input. Everything below runs
# unattended and logs to ~/forge-setup.log.
#
#   curl -fsSL <raw url of this file> -o s.sh && bash s.sh
#
# Downloaded rather than piped straight into bash, so that sudo and the
# embedded heredoc keep their own stdin.
#
# It is idempotent: re-running skips what is already there.

set -uo pipefail

LOG="$HOME/forge-setup.log"
exec > >(tee -a "$LOG") 2>&1

FORGE_DIR="$HOME/sd-webui-forge-neo"
STAGE="$HOME/klein-stage"

say() { printf '\n=== %s ===\n' "$*"; }
die() { printf '\nFAILED: %s\n(see %s)\n' "$*" "$LOG"; exit 1; }

say "started $(date -u +%FT%TZ)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || die "no GPU visible"
sudo -n true 2>/dev/null || die "sudo wants a password; run 'sudo true' once first, then re-run this"

# Tens of GB of weights outlast a phone's SSH session, so get out of the
# foreground before downloading anything. tmux has to exist before it can
# hold us, hence the install above the re-exec.
if [ -z "${TMUX:-}" ] && [ "${NO_TMUX:-}" != "1" ]; then
  command -v tmux >/dev/null || { sudo apt-get update -qq; sudo apt-get install -y -qq tmux; }
  if command -v tmux >/dev/null; then
    self=$(readlink -f "$0")
    say "re-running inside tmux session 'setup' (attach: tmux attach -t setup)"
    tmux kill-session -t setup 2>/dev/null
    tmux new-session -d -s setup "bash '$self'"
    printf '\nDetached. It keeps running if your connection drops.\n'
    printf '  watch:  tmux attach -t setup      (detach again: Ctrl-B then D)\n'
    printf '  or:     tail -f %s\n\n' "$LOG"
    exit 0
  fi
  echo "(tmux unavailable, continuing in the foreground)"
fi

# ---------------------------------------------------------------- packages
say "system packages"
sudo apt-get update -qq || die "apt update"
sudo apt-get install -y -qq tmux git curl python3-pip python3-venv || die "apt install"

say "huggingface_hub"
# Ubuntu 24.04 marks the system python as externally managed; older ones do not.
pip install -q --break-system-packages "huggingface_hub[cli]" 2>/dev/null \
  || pip install -q "huggingface_hub[cli]" \
  || die "could not install huggingface_hub"
export PATH="$HOME/.local/bin:$PATH"

# ------------------------------------------------------------------- forge
say "Forge Neo"
if [ -d "$FORGE_DIR/.git" ]; then
  echo "already cloned at $FORGE_DIR"
else
  # The Neo work lives on a branch of forge-classic, not on its default branch.
  git clone --depth 1 --branch neo \
    https://github.com/Haoming02/sd-webui-forge-classic "$FORGE_DIR" \
    || die "clone"
fi
cd "$FORGE_DIR" || die "cd $FORGE_DIR"

say "model folders Forge expects"
ls models/ 2>/dev/null || echo "(models/ not present yet)"

# ------------------------------------------------------------------ models
# Filenames in these repos move around, so ask the registry what is actually
# there rather than hardcoding a path that 404s six weeks later.
say "resolving model files"
mkdir -p "$STAGE"

python3 - "$STAGE" <<'PY' || die "model download"
import sys, os
from huggingface_hub import list_repo_files, hf_hub_download

stage = sys.argv[1]

def pick(repo, subdir, prefer=(), avoid=()):
    """Newest-looking .safetensors under subdir, ranked by prefer/avoid hints."""
    try:
        files = list_repo_files(repo)
    except Exception as e:
        print(f"  ! cannot list {repo}: {e}")
        return None
    cands = [f for f in files
             if f.endswith(".safetensors") and (not subdir or subdir in f)]
    if not cands:
        print(f"  ! nothing matching in {repo} ({subdir or 'any'})")
        print(f"    saw: {files[:20]}")
        return None
    def score(f):
        low = f.lower()
        s = 0
        for i, p in enumerate(prefer):
            if p in low:
                s -= (len(prefer) - i) * 10
        for a in avoid:
            if a in low:
                s += 50
        return (s, len(f))
    cands.sort(key=score)
    print(f"  {repo}: {len(cands)} candidate(s), taking {cands[0]}")
    for c in cands[1:6]:
        print(f"      (also available: {c})")
    return cands[0]

jobs = [
    # The diffusion weights themselves are not the censored part; take the
    # plain base checkpoint and skip the community quants that Forge Neo
    # refuses to load (upstream issue #1226).
    ("diffusion", "Comfy-Org/flux2-klein-9B", "diffusion_models",
     ("base", "fp8"), ("nunchaku", "lightning", "lighting", "int8", "fp4")),

    ("vae", "Comfy-Org/vae-text-encorder-for-flux-klein-9b", "vae",
     ("flux2",), ()),

    # This is the uncensored piece. Klein's refusals live in the Qwen3 text
    # encoder, not in the transformer, so swapping this one file is what
    # actually lifts them.
    ("text_encoder", "ponpoke/flux2-klein-9b-uncensored-text-encoder", "",
     ("fp8mixed", "fp8", "safetensors"), ("fp4", "gguf")),
]

resolved = {}
for kind, repo, subdir, prefer, avoid in jobs:
    print(f"\n{kind}:")
    name = pick(repo, subdir, prefer, avoid)
    if not name:
        sys.exit(f"could not resolve {kind} from {repo}")
    path = hf_hub_download(repo_id=repo, filename=name, local_dir=stage)
    resolved[kind] = path
    print(f"  -> {path} ({os.path.getsize(path)/2**30:.1f} GiB)")

with open(os.path.join(stage, "resolved.env"), "w") as fh:
    for k, v in resolved.items():
        fh.write(f"{k.upper()}={v}\n")
print("\nall three resolved")
PY

# shellcheck disable=SC1090
. "$STAGE/resolved.env" || die "resolved.env missing"

# --------------------------------------------------------------- placement
# A1111/Forge layout. mkdir -p is harmless when they already exist, and the
# ls above is in the log if any of these turn out to be wrong.
say "placing files"
mkdir -p models/Stable-diffusion models/VAE models/text_encoder
cp -n "$DIFFUSION"    models/Stable-diffusion/ && echo "  checkpoint  -> models/Stable-diffusion/"
cp -n "$VAE"          models/VAE/              && echo "  vae         -> models/VAE/"
cp -n "$TEXT_ENCODER" models/text_encoder/     && echo "  uncensored TE -> models/text_encoder/"

say "on disk"
find models -name '*.safetensors' -printf '%p  %sB\n' 2>/dev/null || find models -name '*.safetensors'

# ------------------------------------------------------------------ launch
# Port 7860 is forwarded publicly by Thunder, so it does not go up bare.
PASSFILE="$HOME/forge-credentials.txt"
if [ ! -f "$PASSFILE" ]; then
  printf 'user: forge\npass: %s\n' "$(openssl rand -base64 18)" > "$PASSFILE"
  chmod 600 "$PASSFILE"
fi
FORGE_USER=$(awk '/^user:/{print $2}' "$PASSFILE")
FORGE_PASS=$(awk '/^pass:/{print $2}' "$PASSFILE")

say "launching under tmux"
tmux kill-session -t forge 2>/dev/null
tmux new-session -d -s forge \
  "cd $FORGE_DIR && ./webui.sh --listen --port 7860 --api \
     --gradio-auth $FORGE_USER:$FORGE_PASS 2>&1 | tee -a $HOME/forge-run.log"

cat <<EOF

=== done ===

Forge is starting in tmux session "forge". First boot builds its venv and
downloads torch, so give it 5-15 minutes before the page answers.

  login      $FORGE_USER / $FORGE_PASS     (also in $PASSFILE)
  watch it   tmux attach -t forge          (detach: Ctrl-B then D)
  run log    tail -f $HOME/forge-run.log
  setup log  $LOG

Then in the UI: Settings -> Stable Diffusion -> enable Klein img2img,
and select the Klein checkpoint in the top-left dropdown.

EOF
