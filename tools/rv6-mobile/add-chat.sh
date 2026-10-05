#!/usr/bin/env bash
# Private uncensored chat on the VM, used from the phone like the edit page.
# Open WebUI is the ChatGPT-style page, served by Tailscale at
# https://<vm>.ts.net:8443 (tailnet only, no login needed). Models:
#   qwen  Qwen 3.8 27B Uncensored (OrcaRouter build, top of the Sep 2026
#         abliteration benchmarks; reads images). Runs on llama.cpp's
#         llama-server with --jinja: Ollama drops Qwen 3.8's own chat template,
#         which leaves thinking stuck at "xhigh" (very long waits); with the
#         real template it is set to "medium". ~23 GB at Q6_K, so it shares the
#         GPU with image editing.
#   8b    Llama 3.1 8B abliterated on Ollama: small and fast.
#   70b   Llama 3.1 70B lorablated on Ollama: only on request (43 GB).
# Safe to re-run: finished steps are skipped.
#   bash <(curl -fsSL https://raw.githubusercontent.com/jsiggins40/locally-uncensored/rv6-mobile-editor/tools/rv6-mobile/add-chat.sh)
#   glm   GLM-5.3-Flash Uncensored (320B MoE, 18B active; #3 open model in Sep
#         2026, the biggest uncensored one this VM can hold): core on the GPU,
#         experts in system RAM. Takes the whole machine (stops image editing)
#         and replaces Qwen in the chat.  CHAT_MODELS=glm bash <(curl ...)
# Pick models:  CHAT_MODELS="qwen 8b 70b" bash <(curl ...)
# Qwen on the CPU (default) keeps the whole GPU for images/video; slower
# replies (a few words a second), so it uses the smaller Q4 file and thinks
# "low". On the GPU, sharing it with image editing:  CHAT_DEVICE=gpu bash <(curl ...)
# Everything for Qwen (stops image editing; edit-restart brings it back),
# best quality Q8_0, 128K context:  CHAT_DEVICE=max bash <(curl ...)
set -euo pipefail
WANT="${CHAT_MODELS:-qwen}"
DEVICE="${CHAT_DEVICE:-cpu}"
if [[ " $WANT " == *" glm "* ]]; then DEVICE=max; fi  # GLM needs the GPU plus all RAM
has() { [[ " $WANT " == *" $1 "* ]]; }
ok=1
if mountpoint -q /ephemeral 2>/dev/null; then BIG=/ephemeral; else BIG="$HOME"; fi
LLM="$BIG/llm"   # Qwen GGUF files live here (big disk when there is one)
mkdir -p "$HOME/.local/bin"

# ---------------------------------------------------------------- Ollama
if has 8b || has 70b; then
  echo "== Ollama (Llama models)"
  command -v ollama >/dev/null || curl -fsSL https://ollama.com/install.sh | sh
  # Models on the big disk; a model leaves the GPU after 10 idle minutes
  sudo mkdir -p /etc/systemd/system/ollama.service.d
  {
    echo "[Service]"
    echo "Environment=OLLAMA_KEEP_ALIVE=10m"
    echo "Environment=OLLAMA_HOST=127.0.0.1:11434"
    if [ "$BIG" = /ephemeral ]; then
      sudo mkdir -p /ephemeral/ollama && sudo chown -R ollama:ollama /ephemeral/ollama
      echo "Environment=OLLAMA_MODELS=/ephemeral/ollama"
    fi
  } | sudo tee /etc/systemd/system/ollama.service.d/rv6m.conf >/dev/null
  sudo systemctl daemon-reload
  sudo systemctl enable ollama >/dev/null 2>&1 || true
  sudo systemctl restart ollama
  for i in $(seq 1 30); do curl -sf -o /dev/null localhost:11434/api/version && break; sleep 1; done
  curl -sf localhost:11434/api/version >/dev/null || { echo "Ollama did not start:"; sudo journalctl -u ollama -n 20 --no-pager; exit 1; }

  # Pull the first source/quant that exists, then give it a short, clear name
  pull_as() {
    local name="$1"; shift
    if ollama list | awk '{print $1}' | grep -qx "$name:latest"; then echo "   already have $name"; return 0; fi
    for src in "$@"; do
      echo "   trying $src"
      if ollama pull "$src"; then ollama cp "$src" "$name" && echo "   installed as $name"; return 0; fi
    done
    echo "   COULD NOT GET $name from any source"; return 1
  }
  if has 8b; then
    pull_as llama3.1-8b-abliterated \
      hf.co/mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated-GGUF:Q8_0 \
      hf.co/mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated-GGUF:Q6_K \
      hf.co/bartowski/Meta-Llama-3.1-8B-Instruct-abliterated-GGUF:Q8_0 \
      mannix/llama3.1-8b-abliterated || ok=0
  fi
  if has 70b; then
    pull_as llama3.1-70b-abliterated \
      hf.co/mlabonne/Llama-3.1-70B-Instruct-lorablated-GGUF:Q4_K_M \
      hf.co/bartowski/Llama-3.1-70B-Instruct-lorablated-GGUF:Q4_K_M \
      hf.co/mlabonne/Llama-3.1-70B-Instruct-lorablated-GGUF:Q4_K_S \
      hf.co/mlabonne/Llama-3.1-70B-Instruct-lorablated-GGUF:IQ4_XS || ok=0
  fi
  # The earlier Ollama build of Qwen 3.8 is replaced by the llama.cpp one below
  if has qwen && ollama list | awk '{print $1}' | grep -qx "qwen3.8-27b-abliterated:latest"; then
    ollama rm qwen3.8-27b-abliterated >/dev/null && echo "   removed the old Ollama Qwen 3.8 (thinking stuck at xhigh)"
  fi
fi

if ! has 8b && ! has 70b && systemctl is-active --quiet ollama 2>/dev/null; then
  sudo systemctl disable --now ollama >/dev/null 2>&1 && echo "== Ollama switched off (only Qwen runs now)"
fi

# ---------------------------------------------------- llama.cpp + Qwen 3.8
if has qwen || has glm; then
  echo "== llama.cpp ($DEVICE)"
  # Separate builds: a CPU one (no CUDA needed) and a GPU one
  BUILD="$HOME/llama.cpp/build-$([ "$DEVICE" = cpu ] && echo cpu || echo cuda)"
  SERVER="$BUILD/bin/llama-server"
  if [ ! -x "$SERVER" ] && [ "$DEVICE" = cpu ]; then
    # CPU only: no CUDA needed, quick build
    sudo apt-get install -y -qq git cmake build-essential >/dev/null
    [ -d "$HOME/llama.cpp" ] || git clone -q --depth 1 https://github.com/ggml-org/llama.cpp "$HOME/llama.cpp"
    echo "   building for the CPU (a few minutes)"
    cmake -S "$HOME/llama.cpp" -B "$BUILD" -DGGML_NATIVE=ON -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release >/dev/null
    cmake --build "$BUILD" --target llama-server -j"$(nproc)" >/dev/null
  fi
  if [ ! -x "$SERVER" ]; then
    sudo apt-get install -y -qq git cmake build-essential >/dev/null
    export PATH="/usr/local/cuda/bin:$PATH"
    if ! command -v nvcc >/dev/null; then
      echo "   installing the CUDA compiler (a few minutes)"
      sudo apt-get install -y -qq cuda-toolkit-12-8 >/dev/null 2>&1 || sudo apt-get install -y -qq nvidia-cuda-toolkit >/dev/null
      export PATH="/usr/local/cuda/bin:$PATH"
    fi
    host_cc=()
    # CUDA older than 12.4 can't use Ubuntu 24.04's gcc 13: build with gcc 12 then
    cuda_ver=$(nvcc --version | grep -o 'release [0-9.]*' | tr -dc 0-9. )
    if [ "$(printf '%s\n12.4\n' "$cuda_ver" | sort -V | head -1)" != "12.4" ]; then
      sudo apt-get install -y -qq g++-12 >/dev/null; host_cc=(-DCMAKE_CUDA_HOST_COMPILER=g++-12)
    fi
    [ -d "$HOME/llama.cpp" ] || git clone -q --depth 1 https://github.com/ggml-org/llama.cpp "$HOME/llama.cpp"
    echo "   building (5-10 minutes)"
    cmake -S "$HOME/llama.cpp" -B "$BUILD" -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=native \
      -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release "${host_cc[@]}" >/dev/null
    cmake --build "$BUILD" --target llama-server -j"$(nproc)" >/dev/null
  fi
  "$SERVER" --version 2>&1 | head -2

  if ! has glm; then
  echo "== Qwen 3.8 27B Uncensored ($(case $DEVICE in cpu) echo 'Q4, about 17 GB';; max) echo 'Q8, about 29 GB';; *) echo 'Q6, about 23 GB';; esac))"
  mkdir -p "$LLM"
  "$HOME/ComfyUI/venv/bin/python" - "$LLM" "$DEVICE" <<'PY' || ok=0
import os, sys
from huggingface_hub import HfApi, hf_hub_download
out, device = sys.argv[1], sys.argv[2]
# CPU speed is limited by memory reads, so the smaller Q4 first there
order = {"cpu": ("Q4_K_M", "Q4_K", "Q5_K_M", "Q5_K", "Q6_K"),
         "max": ("Q8_0", "Q6_K_L", "Q6_K", "Q5_K_M", "Q5_K", "Q4_K_M", "Q4_K")}.get(
         device, ("Q6_K", "Q6_K_L", "Q5_K_M", "Q5_K", "Q4_K_M", "Q4_K"))
api = HfApi()
# OrcaRouter's build (best compliance in the Sep 2026 benchmarks) via bartowski's
# open mirror; OrcaRouter's own repo is gated; huihui-ai's is the fallback.
sources = [("bartowski/orcarouter_Qwen3.8-27B-Uncensored-GGUF", "Qwen 3.8 27B Uncensored (OrcaRouter)"),
           ("orcarouter/Qwen3.8-27B-Uncensored-GGUF", "Qwen 3.8 27B Uncensored (OrcaRouter)"),
           ("huihui-ai/Huihui-Qwen3.8-27B-abliterated-GGUF", "Qwen 3.8 27B Abliterated (huihui)")]
for repo, label in sources:
    try:
        files = api.list_repo_files(repo)
    except Exception as e:
        print(f"   {repo}: {type(e).__name__}"); continue
    ggufs = [f for f in files if f.endswith(".gguf") and "mmproj" not in f.lower()]
    model = None
    for q in order:
        parts = sorted(f for f in ggufs if f"-{q}." in f or f"-{q}-" in f or f"_{q}." in f or f"/{q}/" in f)
        if parts: model = parts; break
    mm = sorted((f for f in files if "mmproj" in f.lower() and f.endswith(".gguf")),
                key=lambda f: ("f16" in f.lower() or "bf16" in f.lower()), reverse=True)
    if not model:
        print(f"   {repo}: no Q6/Q5/Q4 file"); continue
    try:
        paths = [hf_hub_download(repo, f, local_dir=out) for f in model]  # split files: all parts
        mmp = hf_hub_download(repo, mm[0], local_dir=out) if mm else ""
    except Exception as e:
        print(f"   {repo}: download failed ({type(e).__name__}: {e})"); continue
    with open(os.path.join(out, "qwen-chat.env"), "w") as f:
        f.write(f'MODEL="{paths[0]}"\nMMPROJ="{mmp}"\nLABEL="{label}"\n')
    print("   ready:", label, "|", os.path.basename(paths[0]), "| vision:", bool(mmp))
    break
else:
    sys.exit("   COULD NOT GET Qwen 3.8 from any source")
PY
  else
  # GLM-5.3-Flash: pick the best quant that fits. Its experts (most of the
  # file) live in system RAM, so the file must fit in RAM with room to spare.
  ram_gb=$(awk '/MemTotal/ {printf "%d", $2/1048576}' /proc/meminfo)
  echo "== GLM-5.3-Flash Uncensored (biggest quant that fits ${ram_gb} GB of RAM)"
  mkdir -p "$LLM"
  "$HOME/ComfyUI/venv/bin/python" - "$LLM" "$ram_gb" <<'PY' || ok=0
import os, re, shutil, sys
from huggingface_hub import HfApi, hf_hub_download
out, ram = sys.argv[1], float(sys.argv[2])
budget = ram * 0.85 * 1e9        # leave RAM for the OS, the chat page and context
api = HfApi()
# Best quality first; the first one that fits the budget is used
prefs = ["Q4_K_M", "UD-Q4_K_XL", "Q4_K_S", "IQ4_XS", "IQ4_NL", "UD-Q3_K_XL", "Q3_K_M", "IQ3_M",
         "UD-IQ3_XXS", "IQ3_XXS", "Q3_K_S", "UD-Q2_K_XL", "Q2_K", "IQ2_M", "IQ2_XS"]
sources = [("orcarouter/GLM-5.3-Flash-Uncensored-GGUF", "GLM-5.3-Flash Uncensored (OrcaRouter)"),
           ("huihui-ai/Huihui-GLM-5.3-Flash-abliterated-GGUF", "GLM-5.3-Flash Abliterated (huihui)")]
def quant_of(path):
    name = os.path.basename(path); folder = os.path.dirname(path)
    for q in sorted(prefs, key=len, reverse=True):  # longest first: UD-Q4_K_XL before Q4_K
        if re.search(r"(^|[-_./])" + re.escape(q) + r"([-_.]|$)", name) or os.path.basename(folder) == q:
            return q
    return None
for repo, label in sources:
    try:
        sib = api.model_info(repo, files_metadata=True).siblings
    except Exception as e:
        print(f"   {repo}: {type(e).__name__} (gated or missing)"); continue
    groups = {}
    for s in sib:
        if s.rfilename.endswith(".gguf") and "mmproj" not in s.rfilename.lower():
            q = quant_of(s.rfilename)
            if q: groups.setdefault(q, []).append(s)
    for q in prefs:
        if q in groups: print(f"   {repo}: {q} {sum(x.size or 0 for x in groups[q]) / 1e9:.0f} GB")
    pick = next((q for q in prefs if q in groups and sum(x.size or 0 for x in groups[q]) <= budget), None)
    if not pick:
        print(f"   {repo}: nothing fits {budget / 1e9:.0f} GB"); continue
    size = sum(x.size or 0 for x in groups[pick])
    if shutil.disk_usage(out).free < size + 10e9:
        sys.exit(f"   not enough disk for {size / 1e9:.0f} GB")
    print(f"== Downloading {pick} ({size / 1e9:.0f} GB) from {repo}; this takes a while")
    try:
        paths = sorted(hf_hub_download(repo, x.rfilename, local_dir=out) for x in groups[pick])
    except Exception as e:
        print(f"   download failed ({type(e).__name__}: {e})"); continue
    with open(os.path.join(out, "glm-chat.env"), "w") as f:
        f.write(f'MODEL="{paths[0]}"\nLABEL="{label} {pick}"\n')
    print("   ready:", label, pick)
    break
else:
    sys.exit("   COULD NOT GET GLM-5.3-Flash from any source")
PY
  fi
  # Runs llama-server with Qwen's own template (--jinja). GPU: everything on
  # the GPU, thinking "medium", 32K context. CPU: the GPU is hidden from it
  # entirely, all cores, thinking "low", 16K context.
  if has glm; then
    # Core layers on the GPU, all routed experts in system RAM (-ot exps=CPU);
    # GLM uses its own template defaults (no reasoning_effort switch)
    cat > "$HOME/.local/bin/qwen-start" <<SH
#!/usr/bin/env bash
. "$LLM/glm-chat.env"
exec "$SERVER" -m "\$MODEL" --jinja -ngl 99 -ot "exps=CPU" -t $(nproc) -c 65536 --host 127.0.0.1 --port 8081 \\
  --alias glm-5.3-flash-uncensored
SH
    chmod +x "$HOME/.local/bin/qwen-start"
  else
  if [ "$DEVICE" = cpu ]; then
    run='CUDA_VISIBLE_DEVICES= exec'; dev="-ngl 0 -t $(nproc) -c 16384"; effort=low
  else
    run=exec; dev="-ngl 99 -c $([ "$DEVICE" = max ] && echo 131072 || echo 32768)"; effort=medium
  fi
  cat > "$HOME/.local/bin/qwen-start" <<SH
#!/usr/bin/env bash
. "$LLM/qwen-chat.env"
mm=(); [ -n "\$MMPROJ" ] && mm=(--mmproj "\$MMPROJ")
$run "$SERVER" -m "\$MODEL" "\${mm[@]}" --jinja $dev --host 127.0.0.1 --port 8081 \\
  --alias qwen3.8-27b-uncensored --chat-template-kwargs '{"reasoning_effort":"$effort"}'
SH
  chmod +x "$HOME/.local/bin/qwen-start"
  fi
  if [ "$DEVICE" = max ] && tmux has-session -t comfy 2>/dev/null; then
    tmux kill-session -t comfy && echo "   image editing stopped to give the chat model the whole GPU (edit-restart brings it back)"
  fi
fi

# ------------------------------------------------------------- Open WebUI
echo "== Open WebUI (the chat page)"
command -v uv >/dev/null || [ -x "$HOME/.local/bin/uv" ] || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
command -v open-webui >/dev/null || uv tool install --python 3.11 open-webui  # it wants Python 3.11
mkdir -p "$HOME/open-webui-data"
# chat-start (re)starts everything; run it after a VM restart. The env vars
# are authoritative (ENABLE_PERSISTENT_CONFIG=False), so re-runs can change them.
cat > "$HOME/.local/bin/chat-start" <<'SH'
#!/usr/bin/env bash
export PATH="$HOME/.local/bin:$PATH"
env=(DATA_DIR="$HOME/open-webui-data" WEBUI_AUTH=False ENABLE_PERSISTENT_CONFIG=False)
if [ -x "$HOME/.local/bin/qwen-start" ]; then
  tmux kill-session -t qwenchat 2>/dev/null || true
  tmux new -d -s qwenchat "$HOME/.local/bin/qwen-start; echo; echo 'llama-server stopped'; sleep 600"
  env+=(ENABLE_OPENAI_API=True OPENAI_API_BASE_URL=http://127.0.0.1:8081/v1 OPENAI_API_KEY=local)
else
  env+=(ENABLE_OPENAI_API=False)
fi
if systemctl is-active --quiet ollama 2>/dev/null; then env+=(ENABLE_OLLAMA_API=True OLLAMA_BASE_URL=http://127.0.0.1:11434); else env+=(ENABLE_OLLAMA_API=False); fi
tmux kill-session -t chat 2>/dev/null || true
tmux new -d -s chat "env ${env[*]} open-webui serve --host 127.0.0.1 --port 8080"
SH
chmod +x "$HOME/.local/bin/chat-start"
"$HOME/.local/bin/chat-start"
echo "   starting (the first start takes a minute or two)..."
for i in $(seq 1 180); do curl -sf -o /dev/null localhost:8080/health && break; sleep 2; done
if ! curl -sf -o /dev/null localhost:8080/health; then
  echo "Open WebUI did not start. Last log lines:"; tmux capture-pane -pt chat | tail -20; exit 1
fi

echo "== Phone access and test replies"
sudo tailscale serve --bg --https=8443 http://127.0.0.1:8080 >/dev/null
HOST=$(tailscale status --json 2>/dev/null | python3 -c 'import json,sys; print(json.load(sys.stdin)["Self"]["DNSName"].rstrip("."))' 2>/dev/null || true)
if has qwen || has glm; then
  echo "   loading the model (GLM takes several minutes)..."
  for i in $(seq 1 400); do curl -sf -o /dev/null localhost:8081/health && break; sleep 3; done
  echo -n "CHAT TEST ($(has glm && echo GLM-5.3-Flash || echo Qwen 3.8)): "
  curl -s -m 900 localhost:8081/v1/chat/completions -H 'Content-Type: application/json' \
    -d '{"messages":[{"role":"user","content":"Reply with just the word ready."}],"max_tokens":20,"chat_template_kwargs":{"enable_thinking":false}}' \
    | python3 -c 'import json,sys
try: print(json.load(sys.stdin)["choices"][0]["message"]["content"].strip()[:60])
except Exception as e: print("no reply:", e)' || true
  curl -sf -o /dev/null localhost:8081/health || { echo "llama-server is not up. Last lines:"; tmux capture-pane -pt qwenchat | tail -15; }
fi
if systemctl is-active --quiet ollama 2>/dev/null && ollama list | grep -q "llama3.1-8b-abliterated"; then
  echo -n "CHAT TEST (Llama 8B): "
  curl -s localhost:11434/api/generate -d '{"model":"llama3.1-8b-abliterated","prompt":"Reply with just the word ready.","stream":false,"keep_alive":"1m"}' \
    | python3 -c 'import json,sys; print(json.load(sys.stdin).get("response","").strip()[:60])' || true
fi
echo
echo "=============================================================="
echo " CHAT READY. On your phone (Tailscale app connected) open:"
echo "   https://${HOST:-<your-vm>.ts.net}:8443/"
if has glm; then echo " Model: glm-5.3-flash-uncensored (image editing is stopped; edit-restart brings it back)."
else echo " Model: qwen3.8-27b-uncensored (tap the paperclip to show it a photo)."; fi
[ "$DEVICE" = cpu ] && echo " Running on the CPU: replies come at a few words a second."
echo " After a VM restart, run: chat-start"
echo "=============================================================="
[ "$ok" = 1 ] || echo "NOTE: a model above could not be downloaded; see the COULD NOT GET line."
