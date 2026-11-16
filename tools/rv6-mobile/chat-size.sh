#!/usr/bin/env bash
# Shows what fills the chat model's memory (its 64K-token context): for the most
# recently used chats, how long each message is and how much text is attached to it
# (web search results, files). Prints sizes only, never any chat text or titles.
#   bash <(curl -fsSL https://raw.githubusercontent.com/jsiggins40/locally-uncensored/rv6-mobile-editor/tools/rv6-mobile/chat-size.sh)
echo "== Settings in use right now"
echo "  model server: $(pgrep -a llama-server | grep -oE -- '-c [0-9]+|--cache-ram 0|--no-reasoning-preserve' | tr '\n' ' ')"
pid=$(pgrep -f "open-webui serve" | head -1)
echo "  chat page: page cap $([ -n "$pid" ] && tr '\0' '\n' < "/proc/$pid/environ" 2>/dev/null | grep -oE 'WEB_FETCH_MAX_CONTENT_LENGTH=[0-9]+' || echo 'NOT SET'), version $(open-webui --version 2>/dev/null | tail -1 || true)"
python3 - "${DATA_DIR:-$HOME/open-webui-data}/webui.db" <<'PY'
import json, sqlite3, sys, time
try:
    con = sqlite3.connect(f"file:{sys.argv[1]}?mode=ro", uri=True)
    rows = con.execute("select chat, updated_at from chat order by updated_at desc limit 3").fetchall()
except Exception as e:
    sys.exit(f"Can't read the chats ({type(e).__name__}: {e})")
size = lambda v: len(v) if isinstance(v, str) else len(json.dumps(v, ensure_ascii=False)) if v not in (None, [], {}) else 0
for n, (raw, upd) in enumerate(rows, 1):
    d = json.loads(raw) if isinstance(raw, (str, bytes)) else (raw or {})
    msgs = list(((d.get("history") or {}).get("messages") or {}).values()) or d.get("messages") or []
    msgs.sort(key=lambda m: m.get("timestamp") or 0)
    upd = upd / 1e9 if upd and upd > 1e12 else upd or 0
    print(f"\n== Chat {n}: last used {max(0, (time.time() - upd) / 60):.0f} min ago, {len(msgs)} messages")
    total = 0
    extras = {}
    for i, m in enumerate(msgs, 1):
        text = size(m.get("content"))
        other = {k: size(v) for k, v in m.items() if k not in ("content", "id", "parentId", "childrenIds", "role", "timestamp", "model", "modelName", "models", "done", "usage", "info")}
        other = {k: v for k, v in other.items() if v > 500}
        for k, v in other.items(): extras[k] = extras.get(k, 0) + v
        total += text + sum(other.values())
        if text > 4000 or other or len(msgs) <= 12:
            print(f"  #{i:<3} {m.get('role', '?'):<9} text {text:>8,} chars" + "".join(f" | {k} {v:,}" for k, v in other.items()))
    print(f"  total about {total:,} chars = roughly {total // 4:,} tokens if all of it were sent (limit: 65,536)")
    if extras: print("  attached to messages: " + ", ".join(f"{k} {v:,} chars" for k, v in sorted(extras.items(), key=lambda x: -x[1])))
PY
