#!/usr/bin/env bash
# One-shot setup of a fresh Ubuntu 24.04 CUDA GPU VM for the phone edit page:
# GPU driver check/fix, ComfyUI + PyTorch, the edit page, Qwen-Image-Edit,
# the Qwen Edit Plus NSFW LoRA and Tailscale. Safe to re-run: finished steps
# are skipped. Run it inside tmux so a dropped connection doesn't stop it:
#   tmux new -s setup "bash <(curl -fsSL https://raw.githubusercontent.com/jsiggins40/locally-uncensored/rv6-mobile-editor/tools/rv6-mobile/setup.sh); bash"
set -euo pipefail

SRC="https://raw.githubusercontent.com/jsiggins40/locally-uncensored/rv6-mobile-editor/tools/rv6-mobile"
COMFY="$HOME/ComfyUI"
NSFW_LORA_REPO="ScottzillaSystems/qwen-image-edit-plus-nsfw-lora"

step() { echo; echo "===== $* ====="; }

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

step "4/7 Edit page (restarts ComfyUI, tests tap-to-select)"
RV6M_NO_PROMPT=1 bash <(curl -fsSL "$SRC/install.sh") || echo "(page installer reported a problem above; continuing)"

step "5/7 Qwen-Image-Edit (20-40 GB download)"
free_gb=$(df --output=avail -BG "$COMFY" | tail -1 | tr -dc 0-9)
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

step "7/7 Tailscale (private phone access)"
command -v tailscale >/dev/null || curl -fsSL https://tailscale.com/install.sh | sh
sudo systemctl enable --now tailscaled >/dev/null 2>&1 || true
ts_state=$(tailscale status 2>&1 || true)
if grep -qi "logged out\|NeedsLogin\|not logged in" <<<"$ts_state"; then
  echo "Tailscale needs to join your account."
  echo "Easiest: make an auth key at login.tailscale.com/admin/settings/keys (Generate auth key),"
  echo "copy it, and paste it below. Or just press Enter to get a login link instead."
  read -rp "Auth key (tskey-auth-...): " key </dev/tty || key=""
  if [ -n "$key" ]; then sudo tailscale up --reset --auth-key="$key"; else sudo tailscale up --reset; fi
fi
sudo tailscale serve --bg 8188 >/dev/null
HOST=$(tailscale status --json | python3 -c 'import json,sys; print(json.load(sys.stdin)["Self"]["DNSName"].rstrip("."))')

echo
echo "=============================================================="
echo " ALL DONE. On your phone (Tailscale app connected) open:"
echo "   https://$HOST/extensions/rv6_mobile/edit.html"
echo " Choose ✨ Describe edit, then pick the LoRA in the LoRA menu."
echo "=============================================================="
