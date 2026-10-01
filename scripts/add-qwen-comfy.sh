#!/usr/bin/env bash
#
# The separate Qwen-Image-Edit, into ComfyUI, beside the merged one.
#
#   bash qc.sh              fp8 - about 20GB
#   QUALITY=bf16 bash qc.sh full precision - about 40GB
#
# The merged AIO is built for speed: four steps, one file, and it shows on
# faces. This is the model in its own right - transformer, text encoder and
# vae as three files - run at twenty steps. Slower per image and visibly
# better, which is the trade you want once an edit is worth keeping.
#
# Both can sit side by side; the edit form lists whatever is installed and
# the model dropdown says which is which.

set -uo pipefail

COMFY="${COMFY_DIR:-$HOME/ComfyUI}"
LOG="$HOME/qwen-comfy-setup.log"
QUALITY="${QUALITY:-fp8}"
exec > >(tee -a "$LOG") 2>&1

say() { printf '\n=== %s ===\n' "$*"; }
die() { printf '\nFAILED: %s\n(see %s)\n' "$*" "$LOG"; exit 1; }

[ -d "$COMFY/models" ] || die "no ComfyUI at $COMFY"
case "$QUALITY" in fp8|bf16) ;; *) die "QUALITY must be fp8 or bf16";; esac
NEED_GB=$([ "$QUALITY" = "bf16" ] && echo 55 || echo 30)

AVAIL=$(df -BG --output=avail "$HOME" | tail -1 | tr -dc '0-9')
echo "free: ${AVAIL}G, want ${NEED_GB}G for $QUALITY"
[ "${AVAIL:-0}" -ge "$NEED_GB" ] || die "not enough disk"

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
  say "re-running inside tmux session 'qwencomfy'"
  tmux kill-session -t =qwencomfy 2>/dev/null
  tmux new-session -d -s qwencomfy \
    "COMFY_DIR='$COMFY' QUALITY='$QUALITY' bash '$self'"
  printf '\nDetached. watch: tail -n 40 %s\n\n' "$LOG"
  exit 0
fi

export PATH="$HOME/.local/bin:$PATH"
cd "$COMFY" || die "cd $COMFY"
mkdir -p models/diffusion_models models/text_encoders models/vae

# On a box built only for this, nothing has installed the hub client
# yet - the full bootstrap does it on the way past. And a token that is
# missing or revoked shows up as a 401 partway through a forty gigabyte
# download, which is the worst moment to find out.
say "hugging face"
command -v hf >/dev/null || python3 -c 'import huggingface_hub' 2>/dev/null || {
  echo "  installing the hub client"
  pip install -q --break-system-packages "huggingface_hub[cli]" 2>/dev/null \
    || pip install -q --user "huggingface_hub[cli]" 2>/dev/null \
    || pip install -q "huggingface_hub[cli]" 2>/dev/null
}
python3 -c 'import huggingface_hub' 2>/dev/null \
  || die "could not install huggingface_hub"
python3 -c 'from huggingface_hub import whoami; print("  as", whoami()["name"])' \
  || echo "  !! no working token - public models will still download, gated ones will not"

say "resolving"
python3 -u - "$COMFY" "$QUALITY" <<'PY' || die "download"
import os, sys
from huggingface_hub import list_repo_files, hf_hub_download

comfy, quality = sys.argv[1], sys.argv[2]

def grab(repos, dest, must, avoid, prefer, label):
    """First repo that has a matching file wins.

    Repackaged repos get renamed and reorganised, so the filename is
    looked up rather than assumed - and when nothing matches, what the
    repo actually holds is printed, which is the thing you need to fix
    it by hand."""
    for repo in repos:
        try:
            files = list_repo_files(repo)
        except Exception as e:
            print("  - %s: cannot list (%s)" % (repo, type(e).__name__))
            continue
        c = [f for f in files if f.endswith(".safetensors")
             and all(m in f.lower() for m in must)
             and not any(a in f.lower() for a in avoid)]
        if not c:
            print("  - %s: nothing matching %s" % (repo, list(must)))
            continue
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
            print("  = %s: already have %s" % (label, os.path.basename(name)))
            return out
        print("  + %s: %s (%s)" % (label, name, repo))
        p = hf_hub_download(repo_id=repo, filename=name,
                            local_dir=os.path.join(comfy, "models", dest))
        if os.path.abspath(p) != os.path.abspath(out):
            os.replace(p, out)      # flatten split_files/... into the folder
            root = os.path.join(comfy, "models", dest)
            d = os.path.dirname(p)
            while d.startswith(root + os.sep):   # and take the husk with it
                try:
                    os.rmdir(d)
                except OSError:
                    break
                d = os.path.dirname(d)
        print("    -> %s (%.1f GiB)" % (out, os.path.getsize(out) / 2**30))
        return out
    print("  !! %s: not found in any of %s" % (label, repos))
    return None

EDIT_REPOS = ["Comfy-Org/Qwen-Image-Edit_ComfyUI",
              "Comfy-Org/Qwen-Image_ComfyUI"]
ok = True

# 2511 if it is published, otherwise 2509 - both are the "plus" style that
# takes more than one reference image.
model = grab(EDIT_REPOS, "diffusion_models",
             must=("qwen", "edit"),
             avoid=(("fp8",) if quality == "bf16" else ("bf16",)),
             prefer=(("2511", "2509", "bf16") if quality == "bf16"
                     else ("2511", "2509", "fp8_e4m3fn", "fp8")),
             label="edit model (%s)" % quality)
ok = ok and bool(model)

# The vision-language encoder. fp8 regardless: it is conditioning, not the
# thing being sampled, and full precision here buys almost nothing.
ok = grab(EDIT_REPOS, "text_encoders",
          must=("qwen", "2.5", "vl"), avoid=(),
          prefer=("fp8_scaled", "fp8"), label="text encoder") and ok

ok = grab(EDIT_REPOS, "vae", must=("qwen", "vae"), avoid=(),
          prefer=("qwen_image_vae",), label="vae") and ok

sys.exit(0 if ok else 1)
PY

say "what is installed"
for d in diffusion_models text_encoders vae; do
  printf '  %-18s\n' "$d"
  find "models/$d" -maxdepth 1 -name '*.safetensors' -printf '    %f  %s\n' \
    2>/dev/null | awk '{printf "    %-56s %.1f GiB\n", $1, $2/1073741824}'
done

cat <<EOF

=== done ===

Restart ComfyUI so it sees them, then the form:

  tmux kill-session -t =comfy
  COMFY_ARGS=--highvram bash ~/r.sh
  bash ~/e.sh

--highvram leaves the model in VRAM between runs, which is where most of
an edit's wall clock goes. Drop it if this box also runs Wan - the two
model sets do not fit on one card together.

It appears in the Model dropdown as "(separate, 20 steps)". Use 20 steps
and CFG 3.0 with it - the 4 steps and CFG 1 the merged model wants will
give you mush from this one.

Full log: $LOG
EOF
