#!/usr/bin/env bash
# Short way to run this project's installers and checks on the VM, so there is no
# long line to paste (long lines break when a terminal wraps them):
#   rv6                       lists what there is
#   rv6 add-swarmui           same as: bash <(curl -fsSL .../tools/rv6-mobile/add-swarmui.sh)
#   rv6 add-models --lora "krea 2 nsfw"
#   rv6 bg add-swarmui        runs it in tmux, so a dropped connection doesn't stop it
#                             (look at it again with: tmux attach -t rv6)
# edit-restart installs it as /usr/local/bin/rv6.
REPO="https://raw.githubusercontent.com/jsiggins40/locally-uncensored"
# The newest commit by its id: GitHub caches files fetched by branch name for 5 minutes
sha=$(curl -fsSL --max-time 15 https://api.github.com/repos/jsiggins40/locally-uncensored/commits/rv6-mobile-editor 2>/dev/null \
      | python3 -c 'import json,sys; print(json.load(sys.stdin)["sha"])' 2>/dev/null)
SRC="$REPO/${sha:-rv6-mobile-editor}/tools/rv6-mobile"
if [ -z "${1:-}" ]; then
  echo "Usage: rv6 <name> [options]      or: rv6 bg <name> [options]  (keeps running if the connection drops)"
  echo "Names:"
  curl -fsSL --max-time 15 "https://api.github.com/repos/jsiggins40/locally-uncensored/contents/tools/rv6-mobile?ref=rv6-mobile-editor" 2>/dev/null \
    | python3 -c 'import json,sys; print("  " + "  ".join(sorted(f["name"][:-3] for f in json.load(sys.stdin) if f["name"].endswith(".sh") and f["name"] not in ("rv6.sh", "edit-restart.sh", "install.sh"))))' 2>/dev/null \
    || echo "  (couldn't get the list; e.g. add-swarmui, swarm-doctor, add-models, add-video, disk-report)"
  exit 0
fi
if [ "$1" = bg ]; then
  shift
  [ -n "${1:-}" ] || { echo "Usage: rv6 bg <name> [options]"; exit 1; }
  [ -z "${TMUX:-}" ] || { echo "You are already inside tmux, so it keeps running anyway: just use  rv6 $*"; exit 1; }
  tmux has-session -t rv6 2>/dev/null && { echo "A background run is still open (running, or finished and waiting). See it with: tmux attach -t rv6"; echo "Leave it running with Ctrl-b then d; when it has finished, type exit there, then try again."; exit 1; }
  exec tmux new -s rv6 "$(printf '%q ' bash "$(readlink -f "$0")" "$@"); echo; echo '== Finished. Type exit to close this window.'; bash"
fi
name="${1%.sh}"; shift
script=$(curl -fsSL "$SRC/$name.sh" 2>/dev/null) || { echo "There is no '$name' (or GitHub can't be reached). Type rv6 to see the names."; exit 1; }
exec bash <(printf '%s\n' "$script") "$@"
