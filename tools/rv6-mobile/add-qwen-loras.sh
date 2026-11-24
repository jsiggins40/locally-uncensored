#!/usr/bin/env bash
# Downloads LoRAs made for Qwen-Image-Edit-2511 from Hugging Face (the ones
# prithivMLmods' "Qwen-Image-Edit-2511-LoRAs-Fast-Lazy-Load" project uses) into
# models/loras/qwen2511/, and names them for the edit page (lora_info.json).
# They show up under "Qwen 2511 + LoRA" in Describe edit. Re-run any time.
#   bash <(curl -fsSL https://raw.githubusercontent.com/jsiggins40/locally-uncensored/rv6-mobile-editor/tools/rv6-mobile/add-qwen-loras.sh)
#   ... add-qwen-loras.sh) styles     also the art-style ones (anime, comic, 3D...)
set -euo pipefail
cd "$HOME/ComfyUI"
venv/bin/python - "$@" <<'PY'
import json, os, re, sys
from huggingface_hub import HfApi, hf_hub_download

# (repo, file, name on the page, what to type)
PHOTO = [
    ("prithivMLmods/Qwen-Image-Edit-2511-Ultra-Realistic-Portrait", "URP_20.safetensors", "Ultra-Realistic Portrait", "Ultra-realistic portrait."),
    ("prithivMLmods/Qwen-Image-Edit-2511-Hyper-Realistic-Portrait", "HRP_20.safetensors", "Hyper-Realistic Portrait", "Transform into a hyper-realistic face portrait."),
    ("fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA", "qwen-image-edit-2511-multiple-angles-lora.safetensors", "Multiple Angles", "Front-right quarter view."),
    ("prithivMLmods/Qwen-Image-Edit-2511-Unblur-Upscale", "Qwen-Image-Edit-Unblur-Upscale_15.safetensors", "Unblur", "Unblur and upscale."),
    ("lilylilith/QIE-2511-MP-AnyLight", "QIE-2511-AnyLight_.safetensors", "Any Light (2 photos)", "Apply the lighting from image 2 to image 1."),
    ("prithivMLmods/QIE-2511-Studio-DeLight", "QIE-2511-Studio-DeLight-5000.safetensors", "Studio DeLight", "Neutral uniform lighting. Preserve identity and composition."),
    ("lrzjason/Anything2Real_2601", "anything2real_2601.safetensors", "Anything2Real", "Change the picture to realistic photograph."),
    ("prithivMLmods/QIE-2511-Cinematic-FlatLog-Control", "QIE-2511-Cinematic-FlatLog-Control-3200.safetensors", "Cinematic Flat Log", "Transform into a cinematic flat log."),
]
STYLES = [
    ("prithivMLmods/Qwen-Image-Edit-2511-Anime", "Qwen-Image-Edit-2511-Anime-2000.safetensors", "Anime", "Transform into anime."),
    ("prithivMLmods/Qwen-Image-Edit-2511-Polaroid-Photo", "Qwen-Image-Edit-2511-Polaroid-Photo.safetensors", "Polaroid", "cinematic polaroid with soft grain, subtle vignette, white frame."),
    ("prithivMLmods/Qwen-Image-Edit-2511-Midnight-Noir-Eyes-Spotlight", "Qwen-Image-Edit-2511-Midnight-Noir-Eyes-Spotlight.safetensors", "Midnight Noir", "Transform into Midnight Noir Eyes Spotlight."),
    ("prithivMLmods/Qwen-Image-Edit-2511-Pixar-Inspired-3D", "PI3_20.safetensors", "Pixar-style 3D", "Transform it into Pixar-inspired 3D."),
    ("prithivMLmods/Qwen-Image-Edit-2511-Noir-Comic-Book-Panel", "Noir-Comic-Book-Panel_20.safetensors", "Noir Comic", "Transform into a noir comic book style."),
    ("starsfriday/Qwen-Image-Edit-2511-Upscale2K", "qwen_image_edit_2511_upscale.safetensors", "Upscale 2K", "Upscale this picture to 4K resolution."),
]
todo = PHOTO + (STYLES if "styles" in sys.argv[1:] else [])
dest = os.path.join("models", "loras", "qwen2511")
meta_path = os.path.join("custom_nodes", "rv6_mobile", "web", "lora_info.json")
num = lambda f: [int(n) for n in re.findall(r"\d+", f)]

try: meta = json.load(open(meta_path))
except Exception: meta = {}
done, failed = [], []
for repo, fname, name, say in todo:
    print(f"\n{name}  ({repo})")
    try:
        try:
            path = hf_hub_download(repo, fname, local_dir=dest)
        except Exception as e:
            # The file was renamed upstream: take the repo's newest .safetensors
            files = [f for f in HfApi().list_repo_files(repo) if f.endswith(".safetensors")]
            if not files: raise e
            fname = sorted(files, key=num)[-1]
            print("  (file is now called", fname + ")")
            path = hf_hub_download(repo, fname, local_dir=dest)
    except Exception as e:
        print("  FAILED:", str(e).splitlines()[0][:200]); failed.append(name); continue
    rel = os.path.relpath(path, os.path.join("models", "loras"))
    meta[rel] = {"name": name, "baseModel": "Qwen Image Edit 2511", "trainedWords": []}
    print(f"  saved {rel} ({os.path.getsize(path) // 2**20} MB)  ·  type something like: {say}")
    done.append(name)
json.dump(meta, open(meta_path, "w"), indent=1)
print(f"\nDone: {len(done)} LoRA(s)" + (f"; failed: {', '.join(failed)}" if failed else "") +
      ". Reload the edit page: they're in the LoRA list under 'Qwen 2511 + LoRA' (Describe edit).")
PY
