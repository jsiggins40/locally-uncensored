#!/usr/bin/env bash
#
# Add Qwen-Image-Edit 2511 to an existing Forge Neo install.
#
#   bash q.sh
#
# Separate from the Klein setup on purpose: this is ~30GB, and a rebuild
# should not be forced to carry it. Run it after the main script.
#
# Installs the STOCK encoder only. The abliterated one is a different job
# (it needs converting out of transformers format) and lives in
# add-qwen-abliterated-encoder.sh - run this first, get an edit working,
# then that.

set -uo pipefail

LOG="$HOME/qwen-edit-setup.log"
exec > >(tee -a "$LOG") 2>&1

STAGE="$HOME/qwen-stage"
NEED_GB=32

say() { printf '\n=== %s ===\n' "$*"; }
die() { printf '\nFAILED: %s\n(see %s)\n' "$*" "$LOG"; exit 1; }

# Match the main script's choice so both write into the same tree.
# Which install to target. Two can exist side by side - the image ships
# ~/ForgeNeo, and an earlier version of the setup script cloned
# ~/sd-webui-forge-neo - and models dropped into the one that is not running
# are invisible to the one that is. FORGE_DIR=... overrides the guess.
if [ -n "${FORGE_DIR:-}" ]; then
  [ -d "$FORGE_DIR" ] || die "FORGE_DIR=$FORGE_DIR does not exist"
elif [ -d "$HOME/ForgeNeo" ]; then
  FORGE_DIR="$HOME/ForgeNeo"
elif [ -d "$HOME/sd-webui-forge-neo" ]; then
  FORGE_DIR="$HOME/sd-webui-forge-neo"
else
  die "no Forge Neo install found; run the Klein setup script first"
fi
say "Forge at $FORGE_DIR"

# 30GB of weights onto a 100GB box that already holds Klein is close enough
# to the edge to be worth refusing rather than discovering at 90%.
AVAIL=$(df -BG --output=avail "$HOME" | tail -1 | tr -dc '0-9')
echo "free: ${AVAIL}G, want ${NEED_GB}G"
[ "${AVAIL:-0}" -ge "$NEED_GB" ] || die "not enough disk. Free some space (rm -rf ~/klein-stage frees ~33G once the Klein install is placed)"

if [ -z "${TMUX:-}" ] && command -v tmux >/dev/null; then
  self=$(readlink -f "$0")
  say "re-running inside tmux session 'qwen'"
  tmux kill-session -t qwen 2>/dev/null
  tmux new-session -d -s qwen "bash '$self'"
  printf '\nDetached. watch: tmux attach -t qwen   or: tail -f %s\n\n' "$LOG"
  exit 0
fi

export PATH="$HOME/.local/bin:$PATH"
python3 -c 'from huggingface_hub import whoami; print("  hf as", whoami()["name"])' 2>/dev/null \
  || echo "  (no hf token; public repos only, which is all this needs)"

cd "$FORGE_DIR" || die "cd $FORGE_DIR"
mkdir -p "$STAGE" models/Stable-diffusion models/text_encoder models/VAE models/Lora

say "resolving"
python3 -u - "$STAGE" <<'PY' || die "download"
import sys, os
from huggingface_hub import list_repo_files, hf_hub_download

stage = sys.argv[1]

def pick(repo, must=(), avoid=(), prefer=()):
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
        print(f"      holds: {files[:8]}")
        return None
    # Sorting by name length alone once picked qwen_image_edit_2511_bf16
    # (38GiB) over _fp8mixed (20GiB) purely because the name was shorter, and
    # filled a 100GB disk. Precision is a choice, so state it.
    def score(f):
        low = f.lower()
        for i, p in enumerate(prefer):
            if p in low:
                return (i, len(f))
        return (len(prefer), len(f))
    c.sort(key=score)
    print(f"  + {repo}: {c[0]}")
    for alt in c[1:4]:
        print(f"      (also there: {alt})")
    return c[0]

JOBS = [
    # Forge Neo only treats a model as an EDIT model when its path contains
    # both "qwen" and "edit" - so the filename is load-bearing, not cosmetic.
    # QUALITY=bf16 takes the full-precision 38GiB file instead; worth it on an
    # 80GB card with the disk to spare, ruinous on a 100GB one.
    ("model", "Comfy-Org/Qwen-Image-Edit_ComfyUI",
     ("qwen", "edit", "2511"), ("gguf", "nunchaku"),
     ("bf16",) if os.environ.get("QUALITY") == "bf16" else ("fp8mixed", "fp8")),
    ("clip", "Comfy-Org/Qwen-Image_ComfyUI",
     ("qwen_2.5_vl", "fp8"), ("gguf",), ("fp8_scaled", "fp8")),
    ("vae", "Comfy-Org/Qwen-Image_ComfyUI",
     ("vae",), ("gguf",), ()),
    # Distillation LoRA: 4 steps instead of ~40, about 10x faster. On a 20B
    # edit model that is what makes it usable rather than a coffee break.
    # Unlike the merged "lightning" checkpoints that Forge Neo refuses in
    # issue #1226, this is a LoRA applied at runtime - a different mechanism.
    ("lora", "lightx2v/Qwen-Image-Edit-2511-Lightning",
     ("lightning",), ("gguf", "fp32"), ("4steps", "bf16")),
]
out = {}
for kind, repo, must, avoid, prefer in JOBS:
    print(f"\n{kind}:")
    name = pick(repo, must, avoid, prefer)
    if not name:
        sys.exit(f"could not resolve {kind}")
    try:
        p = hf_hub_download(repo_id=repo, filename=name, local_dir=stage)
    except Exception as e:
        print(f"  - download failed ({type(e).__name__}: {e})")
        sys.exit(f"could not download {kind}")
    leaf = name.rsplit("/", 1)[-1]
    out[kind] = (p, leaf)
    print(f"  -> {p} ({os.path.getsize(p)/2**30:.1f} GiB)")

with open(os.path.join(stage, "qwen.env"), "w") as fh:
    for k, (p, leaf) in out.items():
        fh.write(f"{k.upper()}={p}\n{k.upper()}_AS={leaf}\n")
PY

. "$STAGE/qwen.env" || die "qwen.env missing"

place() {  # <src> <dir> <name> <label>
  [ -n "${1:-}" ] || return 0
  [ -e "$2/$3" ] && { echo "  $4: already there"; return 0; }
  ln "$1" "$2/$3" 2>/dev/null && { echo "  $4 -> $2/$3 (linked)"; return 0; }
  cp "$1" "$2/$3" && echo "  $4 -> $2/$3 (copied)"
}

say "placing"
# Guard the naming rule rather than trusting the upstream filename.
MODEL_AS="${MODEL_AS:-}"
case "$(printf '%s' "$MODEL_AS" | tr 'A-Z' 'a-z')" in
  *qwen*edit*|*edit*qwen*) ;;
  *) echo "  renaming '$MODEL_AS' - Forge needs 'qwen' and 'edit' in the path"
     MODEL_AS="qwen_image_edit_2511.safetensors" ;;
esac

place "${MODEL:-}"     models/Stable-diffusion "$MODEL_AS"           "edit model"
place "${CLIP:-}"      models/text_encoder     "${CLIP_AS:-}"        "stock encoder"
place "${VAE:-}"       models/VAE              "${VAE_AS:-}"         "vae"
place "${LORA:-}"      models/Lora             "${LORA_AS:-}"        "lightning lora"

say "on disk"
ls -la models/Stable-diffusion models/text_encoder models/VAE | grep -i qwen || true
df -h "$HOME" | tail -1

cat <<'EOF'

=== done ===

In Forge: refresh the model list, then pick the qwen_image_edit checkpoint.
Set text encoder to qwen_2.5_vl_7b_fp8_scaled and VAE to qwen_image_vae.
Those are Qwen's - do NOT pair them with Klein's encoder or VAE.

The Lightning LoRA cuts this to 4 steps. It needs CFG 1.0 - leave CFG high
and every render looks burnt. Add <lora:Qwen-Image-Edit-2511-Lightning...:1>
to the prompt, set steps to 4, CFG to 1.0.

Get a plain edit working before adding anything else. Once it does, the
abliterated encoder is a separate script: add-qwen-abliterated-encoder.sh

Staging can be deleted once you are happy:  rm -rf ~/qwen-stage
EOF
