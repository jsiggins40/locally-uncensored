#!/usr/bin/env bash
# Checks SwarmUI (add-swarmui.sh) from end to end and says in plain words what is
# wrong and which command fixes it. Changes nothing. Prints no prompts and no
# pictures: only status, file names and error messages. Ends with a small test
# picture (an apple) through SwarmUI, to show the exact error if generating fails.
#   bash <(curl -fsSL https://raw.githubusercontent.com/jsiggins40/locally-uncensored/rv6-mobile-editor/tools/rv6-mobile/swarm-doctor.sh)
#   ... swarm-doctor.sh) minimax        test with the model whose name contains "minimax"
#   ... swarm-doctor.sh) notest         skip the test picture
export PATH="$HOME/.local/bin:$PATH"
SW="$HOME/SwarmUI"; C="$HOME/ComfyUI"; WANT="${1:-}"
SRC="https://raw.githubusercontent.com/jsiggins40/locally-uncensored/rv6-mobile-editor/tools/rv6-mobile"
ok()  { echo "  OK    $*"; }
bad() { echo "  PROBLEM  $*"; PROBLEMS=$((PROBLEMS + 1)); }
say() { echo; echo "== $*"; }
PROBLEMS=0
code() { curl -s -o /dev/null -m "${2:-5}" -w '%{http_code}' "$1" 2>/dev/null || true; }

say "VM"
echo "  up $(uptime -p | sed 's/^up //'); GPU memory used: $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader 2>/dev/null || echo '?')"
echo "  free disk: $(df --output=avail -BG "$HOME" | tail -1 | tr -dc 0-9) GB"

say "Installed?"
if [ -d "$SW/.git" ]; then ok "SwarmUI is in ~/SwarmUI ($(git -C "$SW" log -1 --format='%h from %cs' 2>/dev/null))"
else bad "SwarmUI is not installed. Run: edit-restart   and then: rv6 bg add-swarmui"; echo; echo "Nothing more to check."; exit 0; fi
[ -x "$HOME/.local/bin/swarm-start" ] && ok "swarm-start exists" || bad "swarm-start is missing: add-swarmui.sh did not finish. Run it again."
sdks=$("$HOME/.dotnet/dotnet" --list-sdks 2>/dev/null | awk '{print $1}' | tr '\n' ' ')
[ -n "$sdks" ] && ok ".NET: $sdks" || bad ".NET is missing (SwarmUI is built with it). See: cd ~/SwarmUI && ./launchtools/linux-dotnet-install.sh ~/.dotnet"
built=$(find "$SW/src/bin" -name SwarmUI.dll 2>/dev/null | head -1)
[ -n "$built" ] && ok "built ($(date -r "$built" +%F))" || bad "not built yet: the first swarm-start builds it (about 5 minutes)"

say "Running?"
tmux has-session -t swarm 2>/dev/null && ok "tmux session 'swarm' exists" || bad "no tmux session 'swarm': SwarmUI was not started. Run: swarm-start"
# its front address answers with a redirect to /Text2Image, so the page itself is asked for
c=$(code 127.0.0.1:7801/Text2Image 10)
case "$c" in 2??|3??) c=200 ;; esac     # any answer that is not an error means it is up
if [ "$c" = 200 ]; then ok "SwarmUI answers on this VM (port 7801)"
else
  bad "SwarmUI does not answer on port 7801 (got: ${c:-nothing}). Last lines of its window:"
  tmux capture-pane -pt swarm -S -200 2>/dev/null | grep -v '^\s*$' | tail -25 | cut -c1-220 | sed 's/^/        /'
fi
last=$(ls -t "$SW"/Logs/*.log 2>/dev/null | head -1)
if [ -n "$last" ]; then
  n=$(grep -c '\[Error\]' "$last" 2>/dev/null || true)
  echo "  newest log: $(basename "$last"), ${n:-0} error lines; the last ones:"
  grep -E '\[(Error|Warning)\]' "$last" | grep -vi 'prompt' | tail -8 | cut -c1-300 | sed 's/^/        /'
fi

say "ComfyUI (SwarmUI draws through it)"
[ "$(code localhost:8188)" = 200 ] && ok "ComfyUI answers (port 8188)" || bad "ComfyUI does not answer. Run: edit-restart"
if curl -s -m 10 localhost:8188/api/object_info/SwarmKSampler 2>/dev/null | grep -q SwarmKSampler; then ok "ComfyUI has SwarmUI's nodes"
else
  bad "ComfyUI has not loaded SwarmUI's nodes, so SwarmUI cannot draw."
  [ -e "$C/custom_nodes/SwarmComfyCommon/__init__.py" ] && echo "        the node folder is linked; ComfyUI needs a restart (edit-restart), or it failed to load them:" \
    || echo "        the node folder is not linked (custom_nodes/SwarmComfyCommon). Run add-swarmui.sh again."
  tmux capture-pane -pt comfy -S -400 2>/dev/null | grep -iE "swarm.*(error|fail|cannot|traceback)|IMPORT FAILED.*Swarm|No module named" | tail -5 | cut -c1-220 | sed 's/^/        /'
fi

say "Published on Tailscale"
if sudo -n tailscale serve status 2>/dev/null | grep -q ':8444'; then ok "https://$(tailscale status --json 2>/dev/null | python3 -c 'import json,sys; print(json.load(sys.stdin)["Self"]["DNSName"].rstrip("."))' 2>/dev/null):8444/"
else bad "port 8444 is not published. Run: sudo tailscale serve --bg --https=8444 http://127.0.0.1:7801"; fi

say "The edit page's 🐝 Swarm mode"
st=$(curl -s -m 60 localhost:8188/rv6m/swarm/status 2>/dev/null)
python3 - "$st" <<'PY' || true
import json, sys
try: j = json.loads(sys.argv[1])
except Exception: print("  PROBLEM  the edit page's bridge gave no answer: the page code on this VM is old or ComfyUI is down. Run: edit-restart"); raise SystemExit
if not j.get("installed"): print("  PROBLEM  the bridge cannot reach SwarmUI:", str(j.get("why"))[:200])
else:
    ms = j.get("models") or []
    print(f"  OK    the bridge sees {len(ms)} models, {len(j.get('loras') or [])} LoRAs" + (", uncensored MiniMax encoder" if j.get("mmEnc") else ""))
    for m in ms[:40]:
        print("        ", str(m.get("name"))[:60], "|", m.get("cls") or m.get("arch") or "type not recognised", "|", m.get("preset") or "no Auto preset")
PY

[ "$c" = 200 ] || { echo; echo "SwarmUI is not running, so no test picture. $PROBLEMS problem(s) above."; exit 0; }
say "SwarmUI's own view, and a test picture"
python3 - "$WANT" <<'PY' || true
import json, sys, time, urllib.request
B, want = "http://127.0.0.1:7801/API/", sys.argv[1].lower()
def api(name, body, timeout=120):
    req = urllib.request.Request(B + name, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))
try: sid = api("GetNewSession", {})["session_id"]
except Exception as e: print("  PROBLEM  SwarmUI's API does not answer:", type(e).__name__, str(e)[:200]); raise SystemExit
try:
    for k, v in api("ListBackends", {"session_id": sid}).items():
        if isinstance(v, dict):
            s = v.get("status")
            print(("  OK    " if s in ("running", "idle") else "  PROBLEM  ") + f"backend '{v.get('title') or v.get('type')}': {s}"
                  + ("" if s in ("running", "idle") else " (it must be 'running': is ComfyUI up, with SwarmUI's nodes?)"))
except Exception as e: print("  (couldn't read the backends:", type(e).__name__, str(e)[:150] + ")")
try: files = api("ListModels", {"session_id": sid, "path": "", "depth": 8, "subtype": "Stable-Diffusion", "allowRemote": False}).get("files") or []
except Exception as e: files = []; print("  PROBLEM  couldn't list the models:", type(e).__name__, str(e)[:150])
print(f"  SwarmUI lists {len(files)} models")
if want == "notest" or not files: raise SystemExit
try: presets = {p.get("title") for p in api("GetMyUserData", {"session_id": sid}).get("presets") or []}
except Exception: presets = set()
try: links = (api("GetMyUserData", {"session_id": sid}).get("model_preset_links") or {}).get("Stable-Diffusion") or {}
except Exception: links = {}
clean = lambda n: n[:-len(".safetensors")] if n.endswith(".safetensors") else n
names = [f.get("name") or "" for f in files]
# Picture models first: ones with an Auto preset, then anything that isn't a video model by its name
import re
rank = lambda n: (0 if links.get(clean(n)) else 1, 1 if re.search(r"2va|[ti]2v|video|wan2", n, re.I) else 0)
names.sort(key=rank)
pick = next((n for n in names if want and want in n.lower()), None) \
    or next((n for n in names if "krea" in n.lower() and "turbo" in n.lower()), None) or names[0]
if rank(pick)[1]: print("  NOTE  this is a video model: SwarmUI will try to make a video with it, which is slow and is not what the edit page's Swarm mode is for")
auto = [t for t in links.get(clean(pick)) or [] if t in presets]
print(f"  test model: {pick} | preset: {auto[0] if auto else 'none (SwarmUI defaults)'}")
print("  making a 1024 px test picture (loading a model can take a few minutes)...")
t = time.time()
try:
    r = api("GenerateText2Image", {"session_id": sid, "images": 1, "model": clean(pick), "width": 1024, "height": 1024, "seed": 1,
                                   "prompt": "a red apple on a wooden table, photo", "donotsave": True, **({"presets": auto[:1]} if auto else {})}, 1800)
except Exception as e:
    body = ""
    try: body = e.read().decode()[:400]
    except Exception: pass
    print(f"  PROBLEM  the request failed after {time.time() - t:.0f}s: {type(e).__name__}: {str(e)[:200]} {body}"); raise SystemExit
if r.get("images"): print(f"  OK    SwarmUI made a picture in {time.time() - t:.0f}s. Generating works with this model.")
else: print(f"  PROBLEM  no picture after {time.time() - t:.0f}s. SwarmUI said: {str(r.get('error') or r)[:600]}")
PY
echo
echo "Done. $PROBLEMS problem(s) found in the checks above (the test picture's result is the last lines). Send this whole output."
