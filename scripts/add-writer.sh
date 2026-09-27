#!/usr/bin/env bash
#
# A local LLM that writes documents, on the box you already have.
#
#   bash a.sh              pick a model to fit the card
#   MODEL=qwen3:32b bash a.sh    force one
#
# Two halves. Ollama runs the model and, unlike a server you leave loaded,
# it drops the weights out of VRAM after a few minutes idle - which is what
# lets an LLM share one GPU with ComfyUI instead of fighting it for memory.
# Pandoc turns what the model writes into Word, PDF and HTML, because a
# language model emits text and nothing else; every "it made me a .docx"
# is a converter standing behind it.
#
# Ollama goes into ~/.local rather than /usr: no root, and nothing to undo.

set -uo pipefail

LOG="$HOME/writer-setup.log"
OLLAMA_DIR="$HOME/.local"
export PATH="$HOME/.local/bin:$PATH"
export OLLAMA_KEEP_ALIVE="${OLLAMA_KEEP_ALIVE:-5m}"
exec > >(tee -a "$LOG") 2>&1

say() { printf '\n=== %s ===\n' "$*"; }
die() { printf '\nFAILED: %s\n(see %s)\n' "$*" "$LOG"; exit 1; }

# Some images run as root with no sudo installed at all, so "sudo apt-get"
# fails for the opposite reason to the one it looks like. Nothing here
# truly needs root anyway - it is only a faster route to pandoc.
asroot() {
  if [ "$(id -u)" = "0" ]; then "$@"
  elif command -v sudo >/dev/null 2>&1 && sudo -n true 2>/dev/null; then sudo -n "$@"
  else return 1
  fi
}

say "the card"
VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || die "no GPU"
VRAM=${VRAM:-0}

# Model names on Ollama come and go, so these are tried in order and the
# first one that actually pulls wins. Bigger is better for a long document:
# holding twenty pages together is where the small ones come apart.
if [ -n "${MODEL:-}" ]; then
  CANDIDATES="$MODEL"
elif [ "$VRAM" -ge 70000 ]; then
  CANDIDATES="llama3.3:70b qwen3:32b qwen2.5:72b mistral-small3.2:24b"
elif [ "$VRAM" -ge 40000 ]; then
  CANDIDATES="qwen3:32b gemma3:27b mistral-small3.2:24b qwen2.5:32b"
elif [ "$VRAM" -ge 20000 ]; then
  CANDIDATES="mistral-small3.2:24b qwen3:14b gemma3:12b qwen2.5:14b"
else
  CANDIDATES="qwen3:8b llama3.1:8b gemma3:4b"
fi
echo "  ${VRAM}MB of VRAM -> trying: $CANDIDATES"

say "ollama"
if command -v ollama >/dev/null; then
  echo "  already installed: $(ollama --version 2>&1 | head -1)"
else
  mkdir -p "$OLLAMA_DIR"
  echo "  downloading (about 1.5GB, it bundles the CUDA runtime)"
  curl -fL# https://ollama.com/download/ollama-linux-amd64.tgz -o /tmp/ollama.tgz \
    || die "could not download ollama"
  tar -C "$OLLAMA_DIR" -xzf /tmp/ollama.tgz || die "could not unpack ollama"
  rm -f /tmp/ollama.tgz
  command -v ollama >/dev/null || die "ollama is not on PATH after unpacking"
  echo "  -> $(command -v ollama)"
fi

say "serving"
if curl -fsS -m 3 http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
  echo "  already answering on 11434"
else
  # No systemd here - this is a container, and systemctl reports the host
  # is down. tmux is what everything else on this box uses anyway.
  tmux kill-session -t =ollama 2>/dev/null
  tmux new-session -d -s ollama \
    "OLLAMA_KEEP_ALIVE=$OLLAMA_KEEP_ALIVE $(command -v ollama) serve 2>&1 | tee -a $HOME/ollama.log"
  for i in $(seq 1 30); do
    curl -fsS -m 2 http://127.0.0.1:11434/api/tags >/dev/null 2>&1 && break
    sleep 1
  done
  curl -fsS -m 3 http://127.0.0.1:11434/api/tags >/dev/null 2>&1 \
    || { tail -20 "$HOME/ollama.log" 2>/dev/null; die "ollama did not come up"; }
  echo "  up on 127.0.0.1:11434"
fi

say "the model"
HAVE=$(curl -fsS http://127.0.0.1:11434/api/tags 2>/dev/null \
       | python3 -c 'import sys,json;print(" ".join(m["name"] for m in json.load(sys.stdin).get("models",[])))' 2>/dev/null)
echo "  already pulled: ${HAVE:-none}"
CHOSEN=""
for m in $HAVE; do
  for c in $CANDIDATES; do
    [ "$m" = "$c" ] && CHOSEN="$m" && break 2
  done
done
if [ -z "$CHOSEN" ]; then
  for c in $CANDIDATES; do
    echo "  pulling $c ..."
    if ollama pull "$c"; then CHOSEN="$c"; break; fi
    echo "  - $c is not available under that name, trying the next"
  done
fi
[ -n "$CHOSEN" ] || die "none of the candidate models could be pulled"
echo "  using: $CHOSEN"
printf '%s\n' "$CHOSEN" > "$HOME/.writer-model"

say "document tooling"
# Pandoc writes the .docx natively. PDF goes through WeasyPrint rather than
# LaTeX: a few megabytes instead of several gigabytes, and it takes CSS, so
# the PDF can be made to look like the HTML instead of like a thesis.
if command -v pandoc >/dev/null; then
  echo "  pandoc: $(pandoc --version | head -1)"
else
  asroot apt-get update -qq 2>/dev/null
  asroot apt-get install -y -qq pandoc 2>/dev/null
  if ! command -v pandoc >/dev/null; then
    # The release tarball is one static binary and needs no privileges.
    echo "  no package manager route; fetching the standalone binary"
    mkdir -p "$HOME/.local/bin"
    for v in 3.8 3.7.0.2 3.6.4 3.5; do
      url="https://github.com/jgm/pandoc/releases/download/$v/pandoc-$v-linux-amd64.tar.gz"
      if curl -fsSL "$url" -o /tmp/pandoc.tgz \
         && tar -C "$HOME/.local/bin" -xzf /tmp/pandoc.tgz \
              --strip-components=2 "pandoc-$v/bin/pandoc" 2>/dev/null; then
        rm -f /tmp/pandoc.tgz
        echo "  pandoc $v -> $HOME/.local/bin/pandoc"
        break
      fi
      rm -f /tmp/pandoc.tgz
    done
  fi
  command -v pandoc >/dev/null \
    && echo "  pandoc: $(pandoc --version | head -1)" \
    || echo "  !! no pandoc; Word and web pages will be unavailable"
fi
asroot apt-get install -y -qq libpango-1.0-0 libpangoft2-1.0-0 libharfbuzz0b \
  libfontconfig1 >/dev/null 2>&1
if python3 -c 'import weasyprint' 2>/dev/null; then
  echo "  weasyprint: present"
else
  pip install -q --break-system-packages weasyprint 2>/dev/null \
    || pip install -q --user weasyprint 2>/dev/null \
    || pip install -q weasyprint 2>/dev/null
  python3 -c 'import weasyprint; print("  weasyprint: installed")' 2>/dev/null \
    || echo "  !! weasyprint missing; PDF will be unavailable (Word still works)"
fi

say "done"
cat <<EOF

  model      $CHOSEN
  ollama     127.0.0.1:11434  (tmux session 'ollama', unloads after $OLLAMA_KEEP_ALIVE idle)
  formats    $(command -v pandoc >/dev/null && echo -n "docx html " ; python3 -c 'import weasyprint' 2>/dev/null && echo -n "pdf " ; echo "md")

Now start the writing form:

  curl -fsSL https://raw.githubusercontent.com/jsiggins40/locally-uncensored/claude/previous-chat-link-k3xjlk/scripts/start-writer-form.sh -o ~/w2.sh && bash ~/w2.sh

Full log: $LOG
EOF
