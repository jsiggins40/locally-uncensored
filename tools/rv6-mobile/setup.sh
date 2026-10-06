#!/usr/bin/env bash
# One-shot setup of a fresh Ubuntu 24.04 CUDA GPU VM for the phone edit page:
# GPU driver check/fix, ComfyUI + PyTorch, the edit page, Qwen-Image-Edit (newest, 2511),
# the Rapid AIO NSFW edit model,
# the Qwen Edit Plus NSFW LoRA and Tailscale. Safe to re-run: finished steps
# are skipped. Run it inside tmux so a dropped connection doesn't stop it:
#   tmux new -s setup "bash <(curl -fsSL https://raw.githubusercontent.com/jsiggins40/locally-uncensored/rv6-mobile-editor/tools/rv6-mobile/setup.sh); bash"
set -euo pipefail

SRC="https://raw.githubusercontent.com/jsiggins40/locally-uncensored/rv6-mobile-editor/tools/rv6-mobile"
COMFY="$HOME/ComfyUI"
NSFW_LORA_REPO="ScottzillaSystems/qwen-image-edit-plus-nsfw-lora"

step() { echo; echo "===== $* ====="; }
# Drop arrow-key/paste escape codes and spaces that end up in a pasted key
clean_key() { printf '%s' "$1" | sed 's/\x1b\[[0-9;]*[A-Za-z~]//g; s/\[20[01]~//g' | tr -d '[:space:][:cntrl:]'; }

step "1/7 GPU driver"
if nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
else
  # The image ships a half-upgraded driver on some VMs (what happened on the
  # last one): finish it, overwriting the file the old 570 package still owns.
  echo "Driver not working, repairing (takes a few minutes)..."
  sudo apt-get update -qq
  sudo apt-get -o Dpkg::Options::="--force-overwrite" --fix-broken install -y || true
  sudo apt-get -o Dpkg::Options::="--force-overwrite" install -y nvidia-driver-580
  echo
  echo ">>> Driver installed. The VM will now REBOOT."
  echo ">>> Wait 1 minute, reconnect with: ssh hyper"
  echo ">>> then run the same setup command again; it continues from here."
  sleep 5; sudo reboot; exit 0
fi

step "2/7 System packages"
# Refresh the package list first: a fresh image's list is often stale and
# points at package versions Ubuntu has since replaced (404 Not Found)
sudo apt-get update -qq
sudo apt-get install -y -qq python3-venv git tmux wget curl >/dev/null

step "3/7 ComfyUI and PyTorch"
if [ ! -d "$COMFY" ]; then git clone -q https://github.com/Comfy-Org/ComfyUI.git "$COMFY"; fi
cd "$COMFY"
[ -d venv ] || python3 -m venv venv
if ! venv/bin/python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  venv/bin/pip install -q -U pip
  venv/bin/pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
fi
venv/bin/pip install -q -r requirements.txt
venv/bin/python -c "import torch; print('PyTorch sees:', torch.cuda.get_device_name(0))"

# Hyperstack GPU VMs come with a large local /ephemeral disk. Keep the models
# there so the faster bf16 Qwen (41 GB) fits; it is wiped on stop/hibernate,
# but a spot VM loses everything on reclaim anyway, and a reboot keeps it.
if [ ! -L models ] && mountpoint -q /ephemeral 2>/dev/null; then
  eph_free=$(df --output=avail -BG /ephemeral | tail -1 | tr -dc 0-9)
  if [ "${eph_free:-0}" -ge 150 ]; then
    echo "Putting models on /ephemeral (${eph_free} GB free)"
    sudo chown "$(id -un):$(id -gn)" /ephemeral
    if [ -d /ephemeral/models ]; then rm -rf models; else mv models /ephemeral/models; fi
    ln -s /ephemeral/models models
  fi
fi
if [ -L models ] && [ ! -e models ]; then
  # /ephemeral was wiped (VM stopped): start an empty models folder there again
  echo "/ephemeral was wiped; recreating the models folder"
  rm models; git checkout -q models; sudo chown "$(id -un):$(id -gn)" /ephemeral; mv models /ephemeral/models; ln -s /ephemeral/models models
fi

step "4/7 Edit page (restarts ComfyUI, tests tap-to-select)"
RV6M_NO_PROMPT=1 bash <(curl -fsSL "$SRC/install.sh") || echo "(page installer reported a problem above; continuing)"

step "5/7 Qwen-Image-Edit, newest version + its Lightning LoRA (20-45 GB download)"
free_gb=$(df --output=avail -BG "$(readlink -f "$COMFY/models")" | tail -1 | tr -dc 0-9)
have_edit=$(ls models/diffusion_models/ 2>/dev/null | grep -i qwen_image_edit || true)
if [ -n "$have_edit" ]; then
  echo "Already installed: $have_edit"
else
  # bf16 is faster on A100-class GPUs but needs ~55 GB free; fp8 needs ~35 GB
  if [ "${free_gb:-0}" -ge 75 ]; then prec=bf16; else prec=fp8; fi
  echo "Free disk: ${free_gb} GB, using $prec"
  QWEN_PRECISION=$prec bash <(curl -fsSL "$SRC/add-edit-model.sh")
fi

step "6/7 NSFW LoRA ($NSFW_LORA_REPO)"
venv/bin/python - "$NSFW_LORA_REPO" <<'PY' || echo "(LoRA install failed; if it says 401/403/gated, run: ~/ComfyUI/venv/bin/huggingface-cli login, then re-run setup)"
import json, os, shutil, sys
from huggingface_hub import HfApi, hf_hub_download
repo = sys.argv[1]
files = [f for f in HfApi().list_repo_files(repo) if f.endswith(".safetensors")]
meta_path = "custom_nodes/rv6_mobile/web/lora_info.json"
try: meta = json.load(open(meta_path))
except Exception: meta = {}
for f in files:
    name = os.path.basename(f); dest = os.path.join("models/loras", name)
    if not os.path.exists(dest):
        print("Downloading", f)
        shutil.move(hf_hub_download(repo, f, local_dir="models/_hf_tmp"), dest)
    meta[name] = {"name": "Qwen Edit Plus NSFW" + ("" if len(files) == 1 else f" ({name})"),
                  "baseModel": "Qwen-Image-Edit", "trainedWords": []}
    print("Installed:", dest)
shutil.rmtree("models/_hf_tmp", ignore_errors=True)
json.dump(meta, open(meta_path, "w"), indent=1)
PY

step "6b/7 ⚡ Rapid AIO NSFW edit model (about 28 GB)"
bash <(curl -fsSL "$SRC/add-aio-model.sh") || echo "(Rapid AIO install failed; Qwen 2511 + LoRA still works)"

step "6c/7 ✍️ Create: Qwen-Image 2512 text-to-image (about 41 GB)"
if ls models/diffusion_models/ 2>/dev/null | grep -qiE "^qwen_image_[0-9]{4}"; then
  echo "Already installed"
else
  bash <(curl -fsSL "$SRC/add-create-model.sh") || echo "(Create model install failed; editing works without it)"
fi

step "6d/7 🔓 Abliterated text encoder for Qwen (about 16 GB)"
bash <(curl -fsSL "$SRC/add-abliterated-encoder.sh") || echo "(Abliterated encoder install failed; the standard one still works)"

# Optional: Seedream 5.0 edits through Atlas Cloud (the key stays on the VM)
if [ ! -s "$HOME/.atlascloud_key" ]; then
  echo
  read -rp "Atlas Cloud API key for the Seedream button (Enter to skip): " akey </dev/tty || akey=""
  akey=$(clean_key "$akey"); [[ "$akey" =~ ^[A-Za-z0-9_.-]{20,}$ ]] || akey=""  # junk (stray keys) = skip
  if [ -n "$akey" ]; then
    printf '%s' "$akey" > "$HOME/.atlascloud_key"; chmod 600 "$HOME/.atlascloud_key"
    tmux kill-session -t comfy 2>/dev/null || true  # pick up the key
    tmux new -d -s comfy "cd $COMFY && venv/bin/python main.py --listen 127.0.0.1 --port 8188"
  fi
fi

step "7/7 Tailscale (private phone access)"
command -v tailscale >/dev/null || curl -fsSL https://tailscale.com/install.sh | sh
sudo systemctl enable --now tailscaled >/dev/null 2>&1 || true
ts_state=$(tailscale status 2>&1 || true)
if grep -qi "logged out\|NeedsLogin\|not logged in" <<<"$ts_state"; then
  echo "Tailscale needs to join your account."
  echo "Easiest: make an auth key at login.tailscale.com/admin/settings/keys (Generate auth key),"
  echo "copy it, and paste it below. Or just press Enter to get a login link instead."
  read -rp "Auth key (tskey-auth-...): " key </dev/tty || key=""
  key=$(clean_key "$key"); key=${key#"${key%%tskey-*}"}  # drop anything typed before the key itself
  if [[ "$key" == tskey-* ]] && sudo tailscale up --reset --auth-key="$key"; then :
  else
    [ -n "$key" ] && echo "That key didn't work; use the login link below instead (open it on your phone)."
    sudo tailscale up --reset
  fi
fi
sudo tailscale serve --bg 8188 >/dev/null
# `edit-restart`: gets the latest page and restarts ComfyUI in one command
sudo curl -fsSL "$SRC/edit-restart.sh" -o /usr/local/bin/edit-restart && sudo chmod +x /usr/local/bin/edit-restart || true
HOST=$(tailscale status --json | python3 -c 'import json,sys; print(json.load(sys.stdin)["Self"]["DNSName"].rstrip("."))')

echo
echo "=============================================================="
echo " ALL DONE. On your phone (Tailscale app connected) open:"
echo "   https://$HOST/extensions/rv6_mobile/edit.html"
echo " ✨ Describe edit: say what to change. Selected area: tap/paint a spot,"
echo " say what to do with it, and only that spot changes."
echo " Later: type  edit-restart  to get page updates or bring it back up."
echo "=============================================================="
