#!/usr/bin/env bash
#
# Start everything that is installed and not currently running.
#
#   bash r.sh
#   SKIP_OLLAMA=1 bash r.sh    leave the language model down
#
# Everything here lives in a tmux session, which is what lets a job
# survive a closed phone - and what does not survive the box restarting.
# After a reboot nothing is running and every page is dead, though all of
# it is still on disk. This puts it back.
#
# Safe to run at any time: anything already up is left alone.

set -uo pipefail

COMFY="${COMFY_DIR:-$HOME/ComfyUI}"
say() { printf '\n=== %s ===\n' "$*"; }

up() { tmux has-session -t "=$1" 2>/dev/null; }

say "what is already running"
tmux ls 2>/dev/null || echo "  (nothing)"

if [ -d "$COMFY" ]; then
  if up comfy; then
    echo "  comfy already up"
  else
    say "ComfyUI"
    cd "$COMFY" && tmux new-session -d -s comfy \
      "./venv/bin/python main.py --listen 127.0.0.1 --port 8288 --preview-method auto 2>&1 | tee -a $HOME/comfy-run.log"
    printf '  waiting'
    for _ in $(seq 60); do
      sleep 3
      printf '.'
      curl -fsS -m 2 http://127.0.0.1:8288/object_info >/dev/null 2>&1 && break
    done
    echo
    if curl -fsS -m 3 http://127.0.0.1:8288/object_info >/dev/null 2>&1; then
      echo "  up on 8288"
    else
      echo "  !! did not come up; see: tail -n 30 $HOME/comfy-run.log"
    fi
  fi
fi

if [ "${SKIP_OLLAMA:-}" != "1" ] && [ -x "$HOME/.local/bin/ollama" ]; then
  if up ollama; then
    echo "  ollama already up"
  else
    say "ollama"
    tmux new-session -d -s ollama \
      "OLLAMA_KEEP_ALIVE=${OLLAMA_KEEP_ALIVE:-5m} $HOME/.local/bin/ollama serve 2>&1 | tee -a $HOME/ollama.log"
    for _ in $(seq 20); do
      sleep 1
      curl -fsS -m 2 http://127.0.0.1:11434/api/tags >/dev/null 2>&1 && break
    done
    curl -fsS -m 3 http://127.0.0.1:11434/api/tags >/dev/null 2>&1 \
      && echo "  up on 11434" || echo "  !! ollama did not come up"
  fi
fi

# The forms check their own dependencies and refuse politely, so a
# missing one is reported rather than fatal.
start() {  # start <session> <script> <what>
  up "$1" && { echo "  $3 already up"; return; }
  [ -s "$2" ] || { echo "  $3: no $2 on this box"; return; }
  say "$3"
  bash "$2" 2>&1 | tail -6
}
start inbox     "$HOME/start-photo-inbox.sh" "photo inbox"
start editform  "$HOME/e.sh"                 "edit form"
start writer    "$HOME/w2.sh"                "writing form"
start trainform "$HOME/t2.sh"                "training form"

say "now running"
tmux ls 2>/dev/null || echo "  (nothing)"
cat <<EOF

  7861 photo inbox   7862 edit   7863 write   7864 train   8188 comfy

Passwords:  grep -H . ~/*-credentials.txt
EOF
