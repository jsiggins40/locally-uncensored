#!/usr/bin/env bash
#
# Convert prithivMLmods/Qwen2.5-VL-7B-Abliterated-Caption-it into a single
# safetensors that Forge Neo can load as a text encoder, and install it
# beside the stock one.
#
#   bash a.sh
#
# Why this is its own script: the repo ships in transformers format - a
# dozen sharded model-0000N-of-0000M.safetensors plus an index - while Forge
# wants one file laid out like Comfy's qwen_2.5_vl_7b_fp8_scaled. Merging the
# shards is mechanical. Whether the KEY NAMES then line up is not, so the
# script diffs them against the stock encoder and says what it found instead
# of installing something that loads to noise.
#
# Run add-qwen-edit-2511.sh first and get a plain edit working on the stock
# encoder. Without that baseline you cannot tell a misaligned encoder from a
# misconfigured pipeline - they look identical from the UI.

set -uo pipefail

REPO="prithivMLmods/Qwen2.5-VL-7B-Abliterated-Caption-it"
OUT_NAME="qwen_2.5_vl_7b_abliterated_caption.safetensors"
STAGE="$HOME/qwen-abl-stage"
NEED_GB=20
LOG="$HOME/qwen-abl.log"
exec > >(tee -a "$LOG") 2>&1

say() { printf '\n=== %s ===\n' "$*"; }
die() { printf '\nFAILED: %s\n(see %s)\n' "$*" "$LOG"; exit 1; }

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
  die "no Forge Neo install found"
fi
say "Forge at $FORGE_DIR"

# The shards are ~16GB and the merged file is another ~16GB. Holding both on
# disk needs 33GB, which a box already carrying Klein and Qwen does not have.
# This machine has 64GB of RAM, so the merge is held in memory and the shards
# are deleted before the output is written - peak disk is one copy, not two.
AVAIL=$(df -BG --output=avail "$HOME" | tail -1 | tr -dc '0-9')
echo "free: ${AVAIL}G, want ${NEED_GB}G (one copy; merge is held in RAM)"
[ "${AVAIL:-0}" -ge "$NEED_GB" ] || die "not enough disk. Note ~/qwen-stage is hardlinked to the installed files, so deleting it frees nothing; remove an unused model instead"

if [ -z "${TMUX:-}" ] && command -v tmux >/dev/null; then
  self=$(readlink -f "$0")
  say "re-running inside tmux session 'qwenabl'"
  tmux kill-session -t qwenabl 2>/dev/null
  tmux new-session -d -s qwenabl "bash '$self'"
  printf '\nDetached. watch: tmux attach -t qwenabl   or: tail -f %s\n\n' "$LOG"
  exit 0
fi

export PATH="$HOME/.local/bin:$PATH"
python3 -c 'import torch, safetensors' 2>/dev/null \
  || pip install -q --break-system-packages torch safetensors 2>/dev/null \
  || pip install -q torch safetensors \
  || die "need torch and safetensors"

cd "$FORGE_DIR" || die "cd $FORGE_DIR"
mkdir -p "$STAGE" models/text_encoder

say "downloading $REPO"
hf download "$REPO" --local-dir "$STAGE" \
  --exclude "*.gguf" "*.pt" "*.bin" "original/*" \
  || die "download"

say "merging and comparing"
python3 -u - "$STAGE" "$FORGE_DIR/models/text_encoder/$OUT_NAME" \
              "$FORGE_DIR/models/text_encoder" <<'PY' || die "merge"
import sys, os, glob, json, collections
import torch
from safetensors.torch import load_file, save_file

stage, out_path, te_dir = sys.argv[1], sys.argv[2], sys.argv[3]

shards = sorted(glob.glob(os.path.join(stage, "**", "*.safetensors"), recursive=True))
shards = [s for s in shards if "abliterated_caption" not in os.path.basename(s)]
if not shards:
    sys.exit(f"no shards under {stage}")
print(f"  {len(shards)} shard(s)")

merged = {}
for s in shards:
    part = load_file(s)
    merged.update(part)
    print(f"    {os.path.basename(s)}: {len(part)} tensors")
print(f"  merged: {len(merged)} tensors")

def prefixes(keys, depth=2):
    c = collections.Counter(".".join(k.split(".")[:depth]) for k in keys)
    return c.most_common(8)

print("\n  its top-level layout:")
for p, n in prefixes(merged):
    print(f"    {p:<40} {n}")

# The merge is mechanical; the naming is the part that decides whether Forge
# can load this at all. Compare against the stock encoder sitting next to it.
stock = None
for cand in glob.glob(os.path.join(te_dir, "*.safetensors")):
    b = os.path.basename(cand).lower()
    if "qwen_2.5_vl" in b and "abliterated" not in b:
        stock = cand
        break

if stock:
    print(f"\n  comparing against {os.path.basename(stock)}")
    try:
        from safetensors import safe_open
        with safe_open(stock, framework="pt") as f:
            skeys = set(f.keys())
        mkeys = set(merged)
        shared = skeys & mkeys
        print(f"    stock keys: {len(skeys)}, ours: {len(mkeys)}, shared: {len(shared)}")
        print("\n    stock layout:")
        for p, n in prefixes(skeys):
            print(f"      {p:<40} {n}")
        if not shared:
            print("\n    !! NO shared key names. Forge will not load this as-is;")
            print("       the two use different naming and it needs remapping.")
        elif len(shared) < 0.5 * len(skeys):
            print("\n    !! fewer than half the stock keys are present.")
            print("       Expect it to load partially or produce noise.")
        else:
            print("\n    naming looks compatible.")
            missing = sorted(skeys - mkeys)[:10]
            if missing:
                print(f"    missing from ours: {missing}")
    except Exception as e:
        print(f"    could not compare: {type(e).__name__}: {e}")
else:
    print("\n  !! no stock qwen_2.5_vl encoder found to compare against.")
    print("     Run add-qwen-edit-2511.sh first - without it there is nothing")
    print("     to check the naming against, and no baseline to fall back to.")

merged = {k: (v.to(torch.bfloat16) if v.is_floating_point() else v)
          for k, v in merged.items()}

# Everything is in RAM now. Drop the shards before writing so the output
# never has to coexist with its own source on a nearly full disk.
freed = 0
for s_ in shards:
    try:
        freed += os.path.getsize(s_)
        os.remove(s_)
    except OSError:
        pass
print(f"\n  shards removed, {freed/2**30:.1f} GiB back before writing")

save_file(merged, out_path, metadata={"format": "pt"})
print(f"\n  -> {out_path} ({os.path.getsize(out_path)/2**30:.1f} GiB)")
PY

say "text encoders now installed"
ls -la models/text_encoder
df -h "$HOME" | tail -1

cat <<EOF

=== done ===

Written: models/text_encoder/$OUT_NAME

Read the comparison above before trying it. If it said the key names do not
line up, it will not work and no setting in the UI will fix that - tell me
what the two layouts were and the remapping can be written.

If it said compatible: in Forge select $OUT_NAME in place of
qwen_2.5_vl_7b_fp8_scaled, keep the Qwen VAE and the qwen_image_edit
checkpoint, and run the same edit you already had working. Compare it
against that known-good result. Noise, or a "mat1 and mat2 shapes" error,
means misaligned - switch the encoder back, nothing else is harmed.

Bear in mind this is a CAPTION fine-tune. Even loading cleanly, its
embedding space has moved from what Qwen-Image-Edit was trained against,
so prompt adherence may differ from stock in ways unrelated to censoring.

Shards can go once you have decided:  rm -rf $STAGE
EOF
