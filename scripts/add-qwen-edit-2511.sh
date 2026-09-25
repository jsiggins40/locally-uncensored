#!/usr/bin/env bash
#
# Add Qwen-Image-Edit 2511 to an existing Forge Neo install.
#
#   bash q.sh                 stock encoder (works, censored)
#   ABLITERATED=1 bash q.sh   also fetch the abliterated caption encoder
#
# Separate from the Klein setup on purpose: this is ~30GB, and a rebuild
# should not be forced to carry it. Run it after the main script.
#
# Note on ABLITERATED: the encoder it fetches is a CAPTION fine-tune that
# happens to be abliterated, not a drop-in replacement built for this
# pipeline the way Klein's was. Its own repo has an open "mat1 and mat2
# shapes" report. It is installed ALONGSIDE the stock encoder, never over
# it, so you always have a working baseline to compare against.

set -uo pipefail

LOG="$HOME/qwen-edit-setup.log"
exec > >(tee -a "$LOG") 2>&1

STAGE="$HOME/qwen-stage"
NEED_GB=35

say() { printf '\n=== %s ===\n' "$*"; }
die() { printf '\nFAILED: %s\n(see %s)\n' "$*" "$LOG"; exit 1; }

# Match the main script's choice so both write into the same tree.
if [ -d "$HOME/ForgeNeo" ]; then
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
  tmux new-session -d -s qwen "ABLITERATED=${ABLITERATED:-0} bash '$self'"
  printf '\nDetached. watch: tmux attach -t qwen   or: tail -f %s\n\n' "$LOG"
  exit 0
fi

export PATH="$HOME/.local/bin:$PATH"
python3 -c 'from huggingface_hub import whoami; print("  hf as", whoami()["name"])' 2>/dev/null \
  || echo "  (no hf token; public repos only, which is all this needs)"

cd "$FORGE_DIR" || die "cd $FORGE_DIR"
mkdir -p "$STAGE" models/Stable-diffusion models/text_encoder models/VAE

say "resolving"
python3 -u - "$STAGE" "${ABLITERATED:-0}" <<'PY' || die "download"
import sys, os
from huggingface_hub import list_repo_files, hf_hub_download

stage, abliterated = sys.argv[1], sys.argv[2] == "1"

def pick(repo, must=(), avoid=()):
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
    c.sort(key=len)
    print(f"  + {repo}: {c[0]}")
    return c[0]

JOBS = [
    # Forge Neo only treats a model as an EDIT model when its path contains
    # both "qwen" and "edit" - so the filename is load-bearing, not cosmetic.
    ("model", "Comfy-Org/Qwen-Image-Edit_ComfyUI",
     ("qwen", "edit", "2511"), ("gguf", "nunchaku")),
    ("clip", "Comfy-Org/Qwen-Image_ComfyUI",
     ("qwen_2.5_vl", "fp8"), ("gguf",)),
    ("vae", "Comfy-Org/Qwen-Image_ComfyUI",
     ("vae",), ("gguf",)),
]
if abliterated:
    JOBS.append(("clip_abl", "falacal/Qwen2.5-VL-7B-fp8-Abliterated-Caption-it",
                 (), ("gguf",)))

out = {}
for kind, repo, must, avoid in JOBS:
    print(f"\n{kind}:")
    name = pick(repo, must, avoid)
    if not name:
        if kind == "clip_abl":
            print("  !! abliterated encoder unavailable; stock still installed")
            continue
        sys.exit(f"could not resolve {kind}")
    try:
        p = hf_hub_download(repo_id=repo, filename=name, local_dir=stage)
    except Exception as e:
        print(f"  - download failed ({type(e).__name__}: {e})")
        if kind == "clip_abl":
            continue
        sys.exit(f"could not download {kind}")
    leaf = name.rsplit("/", 1)[-1]
    if kind == "clip_abl" and leaf in ("model.safetensors",):
        leaf = "qwen_2.5_vl_7b_abliterated_caption.safetensors"
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
place "${CLIP_ABL:-}"  models/text_encoder     "${CLIP_ABL_AS:-}"    "abliterated encoder"

say "on disk"
ls -la models/Stable-diffusion models/text_encoder models/VAE | grep -i qwen || true
df -h "$HOME" | tail -1

cat <<'EOF'

=== done ===

In Forge: refresh the model list, then pick the qwen_image_edit checkpoint.
Set text encoder to qwen_2.5_vl_7b_fp8_scaled and VAE to qwen_image_vae.
Those are Qwen's - do NOT pair them with Klein's encoder or VAE.

Get a plain edit working on the STOCK encoder first. Only then switch to
the abliterated one, if you installed it. If output turns to noise or you
see a "mat1 and mat2 shapes" error, that encoder is misaligned for this
pipeline - switch back; nothing else is broken.

Staging can be deleted once you are happy:  rm -rf ~/qwen-stage
EOF
