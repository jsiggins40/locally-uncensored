#!/usr/bin/env bash
# Adds an NSFW LoRA for the page's 🩹 Fix engine (FLUX.1 Fill dev OneReward).
# Fix can't use the Qwen NSFW LoRA; Flux needs its own. Downloads
# Flux-Uncensored-V2 from Hugging Face (no key needed) and registers it so the
# page shows it in the "Flux LoRA" menu, switched on by default.
#   bash <(curl -fsSL https://raw.githubusercontent.com/jsiggins40/locally-uncensored/rv6-mobile-editor/tools/rv6-mobile/add-flux-lora.sh)
# To pick a different one from Civitai instead:
#   bash <(curl -fsSL .../add-models.sh) --lora "flux nsfw"
set -euo pipefail
cd "$HOME/ComfyUI"

venv/bin/python - <<'PY'
import json, os, shutil, sys
from huggingface_hub import HfApi, hf_hub_download

api = HfApi()
meta_path = "custom_nodes/rv6_mobile/web/lora_info.json"
# First one that downloads wins; later ones are fallbacks if a repo is gone or gated
sources = [("enhanceaiteam/Flux-Uncensored-V2", "flux_uncensored_v2.safetensors", "Flux Uncensored V2"),
           ("enhanceaiteam/Flux-uncensored", "flux_uncensored_v1.safetensors", "Flux Uncensored")]
for repo, name, label in sources:
    dest = os.path.join("models/loras", name)
    try:
        if not (os.path.exists(dest) and os.path.getsize(dest) > 0):
            files = [f for f in api.list_repo_files(repo) if f.endswith(".safetensors")]
            if not files:
                raise RuntimeError("no .safetensors file")
            print(f"== Downloading {files[0]} from {repo}")
            shutil.move(hf_hub_download(repo, files[0], local_dir="models/_hf_tmp"), dest)
        else:
            print("== Already have", dest)
    except Exception as e:
        print(f"   {repo}: {type(e).__name__}: {e}")
        continue
    try: meta = json.load(open(meta_path))
    except Exception: meta = {}
    meta[name] = {"name": label, "baseModel": "Flux.1 D", "trainedWords": []}
    json.dump(meta, open(meta_path, "w"), indent=1)
    shutil.rmtree("models/_hf_tmp", ignore_errors=True)
    print("Installed:", dest)
    break
else:
    shutil.rmtree("models/_hf_tmp", ignore_errors=True)
    sys.exit("Could not download a Flux NSFW LoRA from Hugging Face. Pick one from Civitai instead:\n"
             "  bash <(curl -fsSL https://raw.githubusercontent.com/jsiggins40/locally-uncensored/rv6-mobile-editor/tools/rv6-mobile/add-models.sh) --lora \"flux nsfw\"")
PY
curl -fsS -o /dev/null -X POST localhost:8188/api/refresh 2>/dev/null || true
echo
echo "Done. Reload the edit page: Selected area -> 🩹 Fix now shows a Flux LoRA menu."
