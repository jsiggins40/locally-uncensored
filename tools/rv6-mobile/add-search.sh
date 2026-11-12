#!/usr/bin/env bash
# Web search for the chat page: installs SearXNG (a private metasearch engine:
# it asks Google, Bing, DuckDuckGo, Wikipedia... at once and merges what they
# find) on this VM and connects it to Open WebUI. In a chat, tap the 🌐 / +
# button and turn on "Web Search": the model searches, reads the top results
# and answers with sources.
# SearXNG listens on 127.0.0.1:8888 only (nothing outside the VM can reach it)
# and runs on the CPU, in a tmux session called "search". chat-start starts it.
# Needs the chat (add-chat.sh) first. Re-run any time to update SearXNG.
#   bash <(curl -fsSL https://raw.githubusercontent.com/jsiggins40/locally-uncensored/rv6-mobile-editor/tools/rv6-mobile/add-search.sh)
set -euo pipefail
# SearXNG runs in a tmux session called "search": if this installer itself was
# started in one with that name, rename ours first
if [ -n "${TMUX:-}" ] && [ "$(tmux display-message -p '#S' 2>/dev/null)" = search ]; then
  tmux rename-session search-install 2>/dev/null || true
fi
export PATH="$HOME/.local/bin:$PATH"
if [ ! -x "$HOME/.local/bin/chat-start" ]; then
  echo "The chat isn't installed yet: run add-chat.sh first."; exit 1
fi
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh

D="$HOME/searxng"   # src/ = SearXNG's code, venv/ = its Python
CONF="$HOME/.config/searxng/settings.yml"

echo "== SearXNG (latest version)"
mkdir -p "$D"
if [ -d "$D/src/.git" ]; then
  git -C "$D/src" fetch -q --depth 1 origin HEAD && git -C "$D/src" reset -q --hard FETCH_HEAD
else
  rm -rf "$D/src"
  git clone -q --depth 1 https://github.com/searxng/searxng "$D/src"
fi
echo "   version $(git -C "$D/src" log -1 --format='%h from %cs')"
[ -x "$D/venv/bin/python" ] || uv venv -q --python 3.12 "$D/venv"
VIRTUAL_ENV="$D/venv" uv pip install -q -r "$D/src/requirements.txt" -r "$D/src/requirements-server.txt"

echo "== Settings ($CONF)"
mkdir -p "$(dirname "$CONF")"
# Keep the secret key from an earlier run; it only signs SearXNG's own cookies
secret=$(sed -n 's/^ *secret_key: *"\([0-9a-f]*\)".*/\1/p' "$CONF" 2>/dev/null | head -1 || true)
[ -n "$secret" ] || secret=$(python3 -c 'import secrets; print(secrets.token_hex(32))')
cat > "$CONF" <<EOF
# Written by add-search.sh (re-running it rewrites this file).
# Everything not listed here is SearXNG's default.
use_default_settings: true
general:
  instance_name: "studio search"
server:
  bind_address: "127.0.0.1"   # only this VM can reach it
  port: 8888
  secret_key: "$secret"
  limiter: false              # one user: no bot protection needed
  image_proxy: false
  public_instance: false
search:
  safe_search: 0              # off (Open WebUI itself asks for "moderate")
  formats: [html, json]       # Open WebUI reads the JSON results
valkey:
  url: false                  # no cache server needed
EOF

cat > "$HOME/.local/bin/search-start" <<'SH'
#!/usr/bin/env bash
# (Re)starts SearXNG (add-search.sh) on 127.0.0.1:8888 in a tmux session called "search"
D="$HOME/searxng"
# Stop any copy still running (also one whose tmux window is gone) and wait until it has
pkill -f '^granian wsgi 127[.]0[.]0[.]1:8888 ' 2>/dev/null || true   # SearXNG's own name, not its tmux window
for _ in $(seq 20); do curl -sf -o /dev/null localhost:8888/healthz || break; sleep 0.5; done
CMD="cd $HOME && SEARXNG_SETTINGS_PATH=$HOME/.config/searxng/settings.yml PYTHONPATH=$D/src \
  $D/venv/bin/granian --interface wsgi --host 127.0.0.1 --port 8888 --workers 1 searx.webapp:app; \
  echo; echo 'SearXNG stopped'; sleep 600"
# Restart inside the existing window rather than closing it first (closing the
# last tmux session shuts tmux down while the new one is starting)
if tmux has-session -t search 2>/dev/null; then tmux respawn-pane -k -t search "$CMD"
else tmux new -d -s search "$CMD"; fi
for _ in $(seq 60); do curl -sf -o /dev/null localhost:8888/healthz && exit 0; sleep 1; done
echo "SearXNG did not start. See: tmux attach -t search"; exit 1
SH
chmod +x "$HOME/.local/bin/search-start"

echo "== Starting SearXNG"
"$HOME/.local/bin/search-start"
echo -n "SEARCH TEST: "
curl -s -m 60 "http://127.0.0.1:8888/search?q=weather&format=json" | python3 -c '
import json, sys
try:
    d = json.load(sys.stdin)
    r = d.get("results", [])
    engines = {x.get("engine") for x in r}
    print(len(r), "results from", len(engines), "search engines")
    bad = [e[0] for e in d.get("unresponsive_engines", [])]
    if bad: print("   (not answering right now: " + ", ".join(bad) + "; the others cover for them)")
except Exception as e:
    print("no results:", e)'

echo "== Connecting it to the chat page"
# chat-start passes these to Open WebUI (its settings come from chat-start, not the page)
python3 - "$HOME/.local/bin/chat-start" <<'PY'
import sys
p = sys.argv[1]
s = open(p).read()
if "search-start" in s:
    print("   chat-start already starts it")
    raise SystemExit
anchor = "tmux kill-session -t chat 2>/dev/null || true"
if anchor not in s:
    raise SystemExit("chat-start doesn't look as expected: re-run add-chat.sh, then this again")
block = '''# Web search (add-search.sh): SearXNG on this VM for the chat's "Web Search"
if [ -x "$HOME/.local/bin/search-start" ]; then
  "$HOME/.local/bin/search-start" || true
  env+=(ENABLE_WEB_SEARCH=True WEB_SEARCH_ENGINE=searxng SEARXNG_QUERY_URL=http://127.0.0.1:8888/search WEB_SEARCH_RESULT_COUNT=5)
fi
'''
open(p, "w").write(s.replace(anchor, block + anchor, 1))
print("   chat-start now starts SearXNG and turns on web search")
PY

echo "== Restarting the chat (the model reloads: about a minute)"
"$HOME/.local/bin/chat-start"
for _ in $(seq 1 90); do curl -sf -o /dev/null localhost:8080/health && break; sleep 2; done
if ! curl -sf -o /dev/null localhost:8080/health; then
  echo "The chat page did not come back. Last log lines:"; tmux capture-pane -pt chat | tail -20; exit 1
fi

HOST=$(tailscale status --json 2>/dev/null | python3 -c 'import json,sys; print(json.load(sys.stdin)["Self"]["DNSName"].rstrip("."))' 2>/dev/null || true)
echo
echo "=============================================================="
echo " WEB SEARCH READY. Open the chat: https://${HOST:-<your-vm>.ts.net}:8443/"
echo " In a chat, tap + (or 🌐) and switch on Web Search, then ask."
echo " After a VM restart, chat-start starts search too."
echo "=============================================================="
