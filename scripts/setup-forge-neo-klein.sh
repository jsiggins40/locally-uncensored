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

# The uncensored text encoder sits behind a gated repo, so without a token the
# run still finishes - just with the stock encoder, which refuses. Better to
# say so here than to have it discovered three prompts into a session.
say "hugging face auth"
if python3 -c 'from huggingface_hub import whoami; print("  as", whoami()["name"])' 2>/dev/null; then
  :
else
  cat <<'EOF'
  !! no Hugging Face token found.
     The uncensored text encoder is gated; without a token this run falls
     back to the stock one and prompts will still be refused.

     Stop now, run exactly this one line, paste ONLY the hf_... key:

       read -p "token: " K && mkdir -p ~/.cache/huggingface && echo -n "$K" > ~/.cache/huggingface/token && echo "stored ${#K} chars"

     then start this script again. Continuing anyway in 10s.
EOF
  sleep 10
fi

# ------------------------------------------------------------------- forge
say "Forge Neo"
# Thunder's Forge Neo template ships one already, at ~/ForgeNeo with a
# start-forgeneo helper. Cloning a second copy beside it wastes disk on a
# 100GB box and leaves two model trees to wonder about later.
if [ -d "$HOME/ForgeNeo" ]; then
  FORGE_DIR="$HOME/ForgeNeo"
  echo "using the image's own install at $FORGE_DIR"
elif [ -d "$FORGE_DIR/.git" ]; then
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

python3 -u - "$STAGE" <<'PY' || die "model download"
import sys, os
from huggingface_hub import list_repo_files, hf_hub_download

stage = sys.argv[1]

def pick(repo, subdir, must=(), prefer=(), avoid=()):
    """Best .safetensors in repo under subdir, ranked by prefer/avoid hints."""
    try:
        files = list_repo_files(repo)
    except Exception as e:
        print(f"  - {repo}: cannot list ({type(e).__name__}: {e})")
        return None
    cands = [f for f in files
             if f.endswith(".safetensors")
             and (not subdir or subdir in f)
             and all(m in f.lower() for m in must)]
    if not cands:
        print(f"  - {repo}: nothing matching {subdir or 'any'} {must or ''}")
        print(f"      holds: {files[:8]}")
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
    print(f"  + {repo}: taking {cands[0]}")
    for c in cands[1:5]:
        print(f"      (also there: {c})")
    return cands[0]

# Several sources per role, tried in order. Comfy-Org/flux2-klein-9B is named
# for the model but ships only its VAE and text encoders - no transformer - so
# the checkpoint has to come from somewhere else, and one dead repo should not
# take the whole install down with it.
QUANTS = ("nunchaku", "lightning", "lighting", "int8", "fp4", "gguf")

JOBS = [
    ("diffusion", [
        # Uncensored full-precision base checkpoint, ungated. First choice on
        # an 80GB card: no license gate to clear from a headless box.
        ("darknight9121/FLUX.2-klein-base-9B-bucket-uncensored", ""),
        # Community fp8 mixed, also ungated.
        ("silveroxides/FLUX.2-dev-fp8_scaled", ""),
        # Official, but gated - needs the licence accepted and HF_TOKEN set.
        ("black-forest-labs/FLUX.2-klein-9b-fp8", ""),
        ("black-forest-labs/FLUX.2-klein-9B", ""),
    ], ("klein", "9b"), ("uncensored", "base", "fp8"), QUANTS, True),

    ("vae", [
        ("Comfy-Org/flux2-klein-9B", "vae"),
        ("Comfy-Org/vae-text-encorder-for-flux-klein-9b", "vae"),
    ], (), ("flux2",), (), True),

    # The uncensored piece. Klein's refusals live in the Qwen3 text encoder,
    # not in the transformer, so this one file is what actually lifts them.
    # Falling back to the stock encoder leaves a working install that still
    # refuses - hence the warning rather than a silent substitution.
    ("text_encoder", [
        # Own mirror first: the upstream repo is gated behind a manual
        # approval that can sit pending indefinitely, and a copy under your
        # own account needs no one's permission but your token.
        ("jsiggins40/flux2-klein-9b-uncensored-text-encoder-bucket", ""),
        ("ponpoke/flux2-klein-9b-uncensored-text-encoder", ""),
        ("Comfy-Org/flux2-klein-9B", "text_encoders"),
    ], (), ("fp8mixed", "fp8"), ("fp4", "gguf"), False),
]

resolved, warnings = {}, []
for kind, sources, must, prefer, avoid, required in JOBS:
    print(f"\n{kind}:")
    for idx, (repo, subdir) in enumerate(sources):
        name = pick(repo, subdir, must, prefer, avoid)
        if not name:
            continue
        try:
            path = hf_hub_download(repo_id=repo, filename=name, local_dir=stage)
        except Exception as e:
            print(f"  - {repo}: download failed ({type(e).__name__}: {e})")
            continue
        # Some repos serve the weights as a bare model.safetensors, which tells
        # you nothing once it is sitting in a dropdown next to the stock one.
        leaf = name.rsplit("/", 1)[-1]
        if leaf in ("model.safetensors", "diffusion_pytorch_model.safetensors"):
            leaf = repo.rsplit("/", 1)[-1].lower().replace(".", "-") + ".safetensors"
        resolved[kind] = (path, leaf)
        print(f"  -> {path} ({os.path.getsize(path)/2**30:.1f} GiB)")
        if leaf != name.rsplit("/", 1)[-1]:
            print(f"     will be placed as {leaf}")
        if kind == "text_encoder" and "uncensored" not in repo.lower():
            warnings.append(
                "text encoder is the STOCK one - the uncensored source failed, "
                "so prompts will still be refused")
        break
    else:
        if required:
            sys.exit(f"could not resolve {kind} from any of "
                     f"{[r for r, _ in sources]}")
        warnings.append(f"{kind} unresolved, skipped")

for w in warnings:
    print(f"\n!! {w}")

with open(os.path.join(stage, "resolved.env"), "w") as fh:
    for k, (path, leaf) in resolved.items():
        fh.write(f"{k.upper()}={path}\n")
        fh.write(f"{k.upper()}_AS={leaf}\n")
print(f"\nresolved: {', '.join(sorted(resolved))}")
PY

# shellcheck disable=SC1090
. "$STAGE/resolved.env" || die "resolved.env missing"

# --------------------------------------------------------------- placement
# A1111/Forge layout. mkdir -p is harmless when they already exist, and the
# ls above is in the log if any of these turn out to be wrong.
say "placing files"
mkdir -p models/Stable-diffusion models/VAE models/text_encoder
place() {  # place <src> <dest-dir> <dest-name> <label>
  [ -n "${1:-}" ] || { echo "  $4: nothing resolved, skipped"; return; }
  [ -e "$2/$3" ] && { echo "  $4: already there"; return; }
  # Hardlink, not copy: staging and models sit on the same filesystem, and a
  # 35GB checkpoint stored twice is how a 100GB box runs out of room halfway
  # through installing torch. Falls back to a copy across filesystems.
  if ln "$1" "$2/$3" 2>/dev/null; then
    echo "  $4 -> $2/$3 (linked)"
  elif cp "$1" "$2/$3"; then
    echo "  $4 -> $2/$3 (copied)"
  else
    echo "  $4: could not place"
  fi
}
place "${DIFFUSION:-}"    models/Stable-diffusion "${DIFFUSION_AS:-}"    "checkpoint"
place "${VAE:-}"          models/VAE              "${VAE_AS:-}"          "vae"
place "${TEXT_ENCODER:-}" models/text_encoder     "${TEXT_ENCODER_AS:-}" "text encoder"

# -------------------------------------------------------------------- loras
# Klein LoRAs are size-specific: a 4B adapter will not load on 9B, and a
# FLUX.1 one will not load at all. Everything here is 9B, from Hugging Face,
# so no second account or API key enters the picture.
say "loras"
mkdir -p models/Lora
while read -r repo file; do
  [ -n "$repo" ] || continue
  if [ -s "models/Lora/$file" ]; then
    echo "  have $file"
    continue
  fi
  if hf download "$repo" "$file" --local-dir models/Lora >/dev/null 2>&1; then
    echo "  + $file  ($repo)"
  else
    echo "  - $file failed ($repo)"
  fi
done <<'LORAS'
dx8152/Flux2-Klein-9B-Enhanced-Details realistic.safetensors
thedeoxen/refcontrol-FLUX.2-klein-9B-reference-pose-lora refcontrol_v2_poses.safetensors
LORAS

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
