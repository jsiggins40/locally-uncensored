#!/usr/bin/env bash
# Checks the chat (add-chat.sh) from end to end, fixes what it safely can, and says
# in plain words what is wrong. Prints no chat text: only status, addresses and codes.
#   bash <(curl -fsSL https://raw.githubusercontent.com/jsiggins40/locally-uncensored/rv6-mobile-editor/tools/rv6-mobile/chat-doctor.sh)
export PATH="$HOME/.local/bin:$PATH"
LOG="$HOME/open-webui.log"
ok()  { echo "  OK    $*"; }
bad() { echo "  PROBLEM  $*"; }
say() { echo; echo "== $*"; }
model_state() { curl -s -m 5 localhost:8081/health 2>/dev/null | grep -q '"ok"' && echo ok || { curl -s -m 5 localhost:8081/health 2>/dev/null | grep -q Loading && echo loading || echo down; }; }
page_up() { curl -sf -m 5 -o /dev/null localhost:8080/health; }

say "VM"
echo "  up $(uptime -p | sed 's/^up //'); GPU memory used: $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader 2>/dev/null)"

say "Chat page and model"
if ! page_up || ! tmux has-session -t qwenchat 2>/dev/null || [ "$(model_state)" = down ]; then
  bad "something is not running: starting the chat (chat-start)"
  chat-start >/dev/null 2>&1
fi
echo -n "  waiting for the model (up to 4 minutes)"
for _ in $(seq 48); do [ "$(model_state)" = ok ] && page_up && break; echo -n .; sleep 5; done; echo
page_up && ok "chat page answers" || bad "chat page does not answer (log: ~/open-webui.log)"
case "$(model_state)" in
  ok) ok "model loaded" ;;
  loading) bad "model still loading after 4 minutes" ;;
  *) bad "model server is down; last error lines:"; tmux capture-pane -pt qwenchat 2>/dev/null | grep -iE "error|cuda|fail|abort|assert|stopped" | tail -5 | sed 's/^/        /' ;;
esac
r=$(curl -s -m 180 localhost:8081/v1/chat/completions -H 'Content-Type: application/json' \
    -d '{"messages":[{"role":"user","content":"Reply with just the word ready."}],"max_tokens":200,"chat_template_kwargs":{"enable_thinking":false}}' \
    | python3 -c 'import json,sys
try:
    d=json.load(sys.stdin); print("ANSWERED" if d["choices"][0]["message"].get("content") or d["choices"][0]["message"].get("reasoning_content") else "EMPTY")
except Exception as e: print("FAILED", type(e).__name__)' 2>/dev/null)
[ "$r" = ANSWERED ] && ok "model answers a test message" || bad "model test message: ${r:-no reply}"
[ -f "$LOG" ] && echo "  chat page restarts since its last start: $(grep -c 'Open WebUI stopped' "$LOG")"

say "Phone link (Tailscale)"
if ! sudo tailscale serve status 2>/dev/null | grep -q ':8443'; then
  bad "chat link :8443 was not published: publishing it now"
  sudo tailscale serve --bg --https=8443 http://127.0.0.1:8080 >/dev/null 2>&1
fi
HOST=$(tailscale status --json 2>/dev/null | python3 -c 'import json,sys; print(json.load(sys.stdin)["Self"]["DNSName"].rstrip("."))' 2>/dev/null)
code=$(curl -s -m 20 -o /dev/null -w '%{http_code}' "https://$HOST:8443/health")
[ "$code" = 200 ] && ok "https://$HOST:8443/ works inside your Tailscale network" || bad "https://$HOST:8443/ gives $code"
tailscale status --json 2>/dev/null | python3 -c '
import json, sys
d = json.load(sys.stdin)
phones = [p for p in d.get("Peer", {}).values() if p.get("OS", "").lower() in ("ios", "android")]
if not phones: print("  PROBLEM  no phone in your Tailscale network")
for p in phones:
    how = "direct" if p.get("CurAddr") else ("relay " + p.get("Relay", "?") if p.get("Relay") else "not connected right now")
    print("  phone %-20s %-7s ip %s, link: %s" % (p.get("HostName", "?"), "online" if p.get("Online") else "OFFLINE",
          (p.get("TailscaleIPs") or ["?"])[0], how))
    open("/tmp/rv6m-phone-ips", "a").write((p.get("TailscaleIPs") or [""])[0] + "\n")
'

say "What the phone asked the chat page for (last 6; addresses and codes only)"
if [ -f "$LOG" ] && [ -s /tmp/rv6m-phone-ips ]; then
  grep -F -f /tmp/rv6m-phone-ips "$LOG" | grep -oE '"(GET|POST|PUT|DELETE) [^ ?"]*[^"]*" [0-9]{3}' | sed -E 's/\?[^ "]*//' | tail -6 | sed 's/^/  /'
  grep -qF -f /tmp/rv6m-phone-ips "$LOG" || echo "  nothing yet from the phone since the chat page started"
fi
rm -f /tmp/rv6m-phone-ips
echo
echo "Now open the chat on the phone in a NEW tab and send a message; then run this again"
echo "and send the whole output if it still fails."
