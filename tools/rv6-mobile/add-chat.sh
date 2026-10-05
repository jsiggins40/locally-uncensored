#!/usr/bin/env bash
# Private uncensored chat on the VM, used from the phone like the edit page:
# Ollama runs the models, Open WebUI is the ChatGPT-style page, Tailscale
# serves it at https://<vm>.ts.net:8443 (tailnet only, no login needed).
# Models: abliterated Llama 3.1 8B and 70B ("abliterated" = refusals removed).
# Safe to re-run: finished steps are skipped.
#   bash <(curl -fsSL https://raw.githubusercontent.com/jsiggins40/locally-uncensored/rv6-mobile-editor/tools/rv6-mobile/add-chat.sh)
# Only the 8B (6 GB):  CHAT_MODELS=8b bash <(curl ...)
set -euo pipefail
WANT="${CHAT_MODELS:-8b 70b}"

echo "== 1/4 Ollama"
command -v ollama >/dev/null || curl -fsSL https://ollama.com/install.sh | sh
# Keep models on the big /ephemeral disk when there is one (like the image
# models), and let a chat model leave the GPU after 10 idle minutes so image
# edits get the memory back.
sudo mkdir -p /etc/systemd/system/ollama.service.d
{
  echo "[Service]"
  echo "Environment=OLLAMA_KEEP_ALIVE=10m"
  echo "Environment=OLLAMA_HOST=127.0.0.1:11434"
  if mountpoint -q /ephemeral 2>/dev/null; then
    sudo mkdir -p /ephemeral/ollama && sudo chown -R ollama:ollama /ephemeral/ollama
    echo "Environment=OLLAMA_MODELS=/ephemeral/ollama"
  fi
} | sudo tee /etc/systemd/system/ollama.service.d/rv6m.conf >/dev/null
sudo systemctl daemon-reload
sudo systemctl enable ollama >/dev/null 2>&1 || true
sudo systemctl restart ollama
for i in $(seq 1 30); do curl -sf -o /dev/null localhost:11434/api/version && break; sleep 1; done
curl -sf localhost:11434/api/version >/dev/null || { echo "Ollama did not start:"; sudo journalctl -u ollama -n 20 --no-pager; exit 1; }

echo "== 2/4 Models"
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
ok=1
if [[ " $WANT " == *" 8b "* ]]; then
  # mlabonne's abliterated 8B; Q8 is near-lossless and only ~9 GB on an 80 GB GPU
  pull_as llama3.1-8b-abliterated \
    hf.co/mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated-GGUF:Q8_0 \
    hf.co/mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated-GGUF:Q6_K \
    hf.co/bartowski/Meta-Llama-3.1-8B-Instruct-abliterated-GGUF:Q8_0 \
    mannix/llama3.1-8b-abliterated || ok=0
fi
if [[ " $WANT " == *" 70b "* ]]; then
  # The 70B is "lorablated": abliterated via a LoRA merge. Q4_K_M is ~43 GB.
  pull_as llama3.1-70b-abliterated \
    hf.co/mlabonne/Llama-3.1-70B-Instruct-lorablated-GGUF:Q4_K_M \
    hf.co/bartowski/Llama-3.1-70B-Instruct-lorablated-GGUF:Q4_K_M \
    hf.co/mlabonne/Llama-3.1-70B-Instruct-lorablated-GGUF:Q4_K_S \
    hf.co/mlabonne/Llama-3.1-70B-Instruct-lorablated-GGUF:IQ4_XS || ok=0
fi
ollama list

echo "== 3/4 Open WebUI (the chat page)"
command -v uv >/dev/null || [ -x "$HOME/.local/bin/uv" ] || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
# Open WebUI wants Python 3.11; uv fetches that version just for it
command -v open-webui >/dev/null || uv tool install --python 3.11 open-webui
mkdir -p "$HOME/open-webui-data"
cat > "$HOME/.local/bin/chat-start" <<'SH'
#!/usr/bin/env bash
tmux kill-session -t chat 2>/dev/null || true
tmux new -d -s chat "PATH=$HOME/.local/bin:\$PATH DATA_DIR=$HOME/open-webui-data WEBUI_AUTH=False \
  OLLAMA_BASE_URL=http://127.0.0.1:11434 ENABLE_OPENAI_API=False open-webui serve --host 127.0.0.1 --port 8080"
SH
chmod +x "$HOME/.local/bin/chat-start"
"$HOME/.local/bin/chat-start"
echo "   starting (the first start takes a minute or two)..."
for i in $(seq 1 180); do curl -sf -o /dev/null localhost:8080/health && break; sleep 2; done
if ! curl -sf -o /dev/null localhost:8080/health; then
  echo "Open WebUI did not start. Last log lines:"; tmux capture-pane -pt chat | tail -20; exit 1
fi

echo "== 4/4 Phone access and a test reply"
sudo tailscale serve --bg --https=8443 http://127.0.0.1:8080 >/dev/null
HOST=$(tailscale status --json 2>/dev/null | python3 -c 'import json,sys; print(json.load(sys.stdin)["Self"]["DNSName"].rstrip("."))' 2>/dev/null || true)
test_model=$(ollama list | awk 'NR>1 && /abliterated/ {print $1}' | sort | head -1)
if [ -n "$test_model" ]; then
  echo -n "CHAT TEST ($test_model): "
  curl -s localhost:11434/api/generate -d "{\"model\":\"$test_model\",\"prompt\":\"Reply with just the word ready.\",\"stream\":false,\"keep_alive\":\"1m\"}" \
    | python3 -c 'import json,sys; print(json.load(sys.stdin).get("response","").strip()[:60])'
fi
echo
echo "=============================================================="
echo " CHAT READY. On your phone (Tailscale app connected) open:"
echo "   https://${HOST:-<your-vm>.ts.net}:8443/"
echo " Pick the model at the top of the chat."
echo " After a VM restart, run: chat-start"
echo "=============================================================="
[ "$ok" = 1 ] || echo "NOTE: a model above could not be downloaded; see the COULD NOT GET line."
