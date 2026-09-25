#!/usr/bin/env bash
#
# ComfyUI beside Forge, sharing its models.
#
#   bash c.sh
#
# Forge Neo could not drive Qwen-Image-Edit: four configurations produced
# four different kinds of corruption, and the checkpoint is built, shipped
# and documented for ComfyUI. Nothing here replaces Forge - it stays on
# 7860 with Klein - this only adds a second front end on 8188.
#
# Models are symlinked, never copied. ~90GB of weights are already on this
# box and there is no reason to fetch or duplicate a byte of them.
#
# ComfyUI has no authentication of any kind, and the forwarded port is
# reachable from the internet, so nginx sits in front of it with a
# password. ComfyUI is bound to localhost and cannot be reached directly.

set -uo pipefail

COMFY="$HOME/ComfyUI"
FORGE="${FORGE_DIR:-$HOME/sd-webui-forge-neo}"
PUBLIC_PORT="${PORT:-8188}"
INTERNAL_PORT=8288
CREDS="$HOME/comfy-credentials.txt"
LOG="$HOME/comfy-setup.log"
NEED_GB=25

exec > >(tee -a "$LOG") 2>&1
say() { printf '\n=== %s ===\n' "$*"; }
die() { printf '\nFAILED: %s\n(see %s)\n' "$*" "$LOG"; exit 1; }

say "started $(date -u +%FT%TZ)"
[ -d "$FORGE/models" ] || die "no Forge models at $FORGE (set FORGE_DIR=...)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || die "no GPU"
sudo -n true 2>/dev/null || die "sudo wants a password; run 'sudo true' once, then re-run"

AVAIL=$(df -BG --output=avail "$HOME" | tail -1 | tr -dc '0-9')
echo "free: ${AVAIL}G, want ${NEED_GB}G (torch and deps; models are linked)"
[ "${AVAIL:-0}" -ge "$NEED_GB" ] || die "not enough disk"

if [ -z "${TMUX:-}" ] && command -v tmux >/dev/null; then
  self=$(readlink -f "$0")
  say "re-running inside tmux session 'comfysetup'"
  tmux kill-session -t comfysetup 2>/dev/null
  tmux new-session -d -s comfysetup \
    "FORGE_DIR='$FORGE' PORT='$PUBLIC_PORT' bash '$self'"
  printf '\nDetached. watch: tmux attach -t comfysetup   or: tail -f %s\n\n' "$LOG"
  exit 0
fi

# ------------------------------------------------------------------ comfy
say "ComfyUI"
if [ -d "$COMFY/.git" ]; then
  echo "already cloned"
else
  git clone --depth 1 https://github.com/comfyanonymous/ComfyUI "$COMFY" || die "clone"
fi
cd "$COMFY" || die "cd $COMFY"

say "python environment"
[ -d venv ] || python3 -m venv venv || die "venv"
./venv/bin/pip install -q --upgrade pip
# cu124 wheels match the driver on these A100 images; without the index pip
# resolves a CPU build and every render takes minutes instead of seconds.
./venv/bin/pip install -q torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu124 || die "torch"
./venv/bin/pip install -q -r requirements.txt || die "requirements"
./venv/bin/python -c 'import torch; print("  cuda:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")' \
  || die "torch cannot see the GPU"

# ----------------------------------------------------------------- models
# ComfyUI's tree differs from A1111's, and a merged checkpoint belongs
# somewhere different from a bare diffusion model. Link by what the file is,
# not by where Forge happened to keep it.
say "linking models"
mkdir -p models/checkpoints models/diffusion_models models/text_encoders \
         models/vae models/loras

link() {  # link <src> <dest-dir>
  [ -f "$1" ] || return 0
  local dst="$2/$(basename "$1")"
  [ -e "$dst" ] && { echo "  have $(basename "$1")"; return 0; }
  ln -s "$1" "$dst" && echo "  + $2/$(basename "$1")"
}

for f in "$FORGE"/models/Stable-diffusion/*.safetensors; do
  [ -f "$f" ] || continue
  b=$(basename "$f" | tr 'A-Z' 'a-z')
  case "$b" in
    *rapid-aio*|*aio*) link "$f" models/checkpoints ;;   # merged: ckpt loader
    *) link "$f" models/diffusion_models ;;              # bare unet
  esac
done
for f in "$FORGE"/models/text_encoder/*.safetensors; do link "$f" models/text_encoders; done
for f in "$FORGE"/models/VAE/*.safetensors;          do link "$f" models/vae; done
for f in "$FORGE"/models/Lora/*.safetensors;         do link "$f" models/loras; done

say "what ComfyUI can see"
for d in checkpoints diffusion_models text_encoders vae loras; do
  n=$(find "models/$d" -maxdepth 1 -name '*.safetensors' | wc -l)
  printf '  %-18s %s\n' "$d" "$n"
done

# ------------------------------------------------------------------- auth
# ComfyUI has no login. On a public forwarded port that means anyone who
# finds it can drive the GPU and read every image on it.
say "password in front"
sudo apt-get update -qq
sudo apt-get install -y -qq nginx apache2-utils || die "nginx"

if [ ! -s "$CREDS" ]; then
  printf 'user: comfy\npass: %s\n' "$(openssl rand -base64 15 | tr -d '/+=')" > "$CREDS"
  chmod 600 "$CREDS"
fi
U=$(awk '/^user:/{print $2}' "$CREDS")
P=$(awk '/^pass:/{print $2}' "$CREDS")
sudo htpasswd -bc /etc/nginx/.comfy_htpasswd "$U" "$P" >/dev/null 2>&1 || die "htpasswd"

# ComfyUI reports progress over a WebSocket, so the proxy has to carry the
# upgrade through or the UI connects and then silently never updates.
sudo tee /etc/nginx/sites-available/comfy >/dev/null <<NGINX
server {
    listen $PUBLIC_PORT;
    client_max_body_size 64M;
    location / {
        auth_basic "comfy";
        auth_basic_user_file /etc/nginx/.comfy_htpasswd;
        proxy_pass http://127.0.0.1:$INTERNAL_PORT;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host \$host;
        proxy_read_timeout 3600s;
        proxy_send_timeout 3600s;
    }
}
NGINX
sudo ln -sf /etc/nginx/sites-available/comfy /etc/nginx/sites-enabled/comfy
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t || die "nginx config rejected"
sudo systemctl restart nginx || sudo nginx -s reload || die "nginx restart"

# ----------------------------------------------------------------- launch
say "launching"
tmux kill-session -t comfy 2>/dev/null
tmux new-session -d -s comfy \
  "cd $COMFY && ./venv/bin/python main.py --listen 127.0.0.1 --port $INTERNAL_PORT 2>&1 | tee -a $HOME/comfy-run.log"

for _ in $(seq 1 60); do
  sleep 2
  code=$(curl -sS -o /dev/null -w '%{http_code}' -u "$U:$P" \
         "http://127.0.0.1:$PUBLIC_PORT/" 2>/dev/null)
  [ "$code" = "200" ] && break
done

cat <<EOF

=== done ===

  through nginx   HTTP ${code:-none}   (200 means ready)
  login           $U / $P     (also in $CREDS)
  watch           tmux attach -t comfy      (detach: Ctrl-B then D)
  run log         $HOME/comfy-run.log

Next:
  1. Thunder console -> forward port $PUBLIC_PORT -> open that URL
  2. Sign in, then load Phr00t's workflow for the Rapid-AIO checkpoint
     (the repo ships one; drag its .json onto the canvas)
  3. Pick qwen-image-edit-rapid-aio-nsfw-v19 in the checkpoint loader

Forge is untouched on 7860. ComfyUI is bound to localhost - only nginx on
$PUBLIC_PORT can reach it, and only with the password above.
EOF
