#!/usr/bin/env bash
# Checks SwarmUI (add-swarmui.sh) from end to end and says in plain words what is
# wrong and which command fixes it. Prints no prompts and no pictures: only status,
# file names and error messages. Ends with a small test picture (an apple) through
# SwarmUI, to show the exact error if generating fails; it says every 15 seconds
# what the GPU is doing, and gives up after 15 minutes. Changes nothing, except
# with "clear", which cancels everything that is running or waiting.
#   bash <(curl -fsSL https://raw.githubusercontent.com/jsiggins40/locally-uncensored/rv6-mobile-editor/tools/rv6-mobile/swarm-doctor.sh)
#   ... swarm-doctor.sh) minimax        test with the model whose name contains "minimax"
#   ... swarm-doctor.sh) notest         skip the test picture
#   ... swarm-doctor.sh) clear          cancel all running and waiting jobs, free the GPU memory
export PATH="$HOME/.local/bin:$PATH"
SW="$HOME/SwarmUI"; C="$HOME/ComfyUI"; WANT="${1:-}"
SRC="https://raw.githubusercontent.com/jsiggins40/locally-uncensored/rv6-mobile-editor/tools/rv6-mobile"
ok()  { echo "  OK    $*"; }
bad() { echo "  PROBLEM  $*"; PROBLEMS=$((PROBLEMS + 1)); }
say() { echo; echo "== $*"; }
PROBLEMS=0
code() { curl -s -o /dev/null -m "${2:-5}" -w '%{http_code}' "$1" 2>/dev/null || true; }
# How many jobs ComfyUI is running and has waiting (numbers only)
queue() { curl -s -m 10 127.0.0.1:8188/api/queue 2>/dev/null | python3 -c 'import json,sys; j=json.load(sys.stdin); print(len(j.get("queue_running") or []), len(j.get("queue_pending") or []))' 2>/dev/null || echo "? ?"; }

if [ "$WANT" = clear ]; then
  echo "Jobs before: $(queue | awk '{print $1" running, "$2" waiting"}')"
  python3 - <<'PY' || true
import json, urllib.request
def post(url, body):
    try: urllib.request.urlopen(urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=60).read(); return True
    except Exception as e: print("   (", url.split("/")[-1], "didn't answer:", type(e).__name__, ")"); return False
try:
    sid = json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:7801/API/GetNewSession", b"{}", {"Content-Type": "application/json"}), timeout=30))["session_id"]
    post("http://127.0.0.1:7801/API/InterruptAll", {"session_id": sid, "other_sessions": True}) and print("  SwarmUI: told to stop everything")
except Exception as e: print("  (SwarmUI didn't answer:", type(e).__name__ + ")")
post("http://127.0.0.1:8188/api/queue", {"clear": True}) and print("  ComfyUI: waiting jobs removed")
post("http://127.0.0.1:8188/api/interrupt", {}) and print("  ComfyUI: running job stopped")
post("http://127.0.0.1:8188/api/free", {"unload_models": True, "free_memory": True}) and print("  ComfyUI: models unloaded from the GPU")
PY
  sleep 5
  echo "Jobs now: $(queue | awk '{print $1" running, "$2" waiting"}'); GPU memory used: $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader 2>/dev/null || echo '?')"
  exit 0
fi

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

say "Memory, and whether something was killed for using too much"
free -g | awk 'NR==2{print "  RAM: " $2 " GB in the VM, " $7 " GB free right now"}'
echo "  GPU memory by program:"
nvidia-smi --query-compute-apps=process_name,used_memory --format=csv,noheader 2>/dev/null | sed 's|.*/||; s/^/        /' | head -6
oom=$(sudo -n dmesg -T 2>/dev/null | grep -iE "out of memory|oom-kill|killed process" | tail -3)
if [ -n "$oom" ]; then echo "  PROBLEM  Linux killed a program because the VM ran out of RAM (the newest lines):"; echo "$oom" | cut -c1-200 | sed 's/^/        /'
else echo "  OK    nothing was killed for lack of RAM since the VM started"; fi
tmux has-session -t comfy 2>/dev/null || echo "  PROBLEM  ComfyUI's window is gone: it crashed or was stopped. Run: edit-restart"
for f in "$HOME/comfyui.log" "$HOME/comfyui.log.1"; do
  [ -s "$f" ] || continue
  e=$(grep -n -iE "out of memory|OutOfMemory|CUDA error|Killed|Segmentation|core dumped|Fatal Python error|Traceback|Error:|Exception" "$f" | grep -vi "prompt" | tail -6)
  [ -n "$e" ] && { echo "  error lines in $(basename "$f") (the newest last):"; echo "$e" | cut -c1-230 | sed 's/^/        /'; }
  end=$(grep "== ComfyUI stopped with exit code" "$f" | tail -1)
  if [ -n "$end" ]; then
    c=$(echo "$end" | sed 's/.*exit code \([0-9]*\).*/\1/')
    case "$c" in
      0|143|130|129) why="it was stopped on purpose (a restart or an installer)" ;;
      137) why="it was KILLED, usually for using too much RAM" ;;
      139|134|132|136) why="it CRASHED inside a library (not a Python error): the lines before it say where" ;;
      *) why="it ended with an error" ;;
    esac
    echo "  $(basename "$f"): ${end#== }: $why"
    case "$c" in 0|143|130|129) ;; *) echo "  its last lines:"; grep -v "^\s*$" "$f" | grep -vi "prompt" | tail -14 | cut -c1-230 | sed 's/^/        /' ;; esac
  fi
done

say "ComfyUI (SwarmUI draws through it)"
[ "$(code localhost:8188)" = 200 ] && ok "ComfyUI answers (port 8188)" || bad "ComfyUI does not answer. Run: edit-restart"
read -r qr qp <<<"$(queue)"
if [ "$qr" = 0 ] && [ "$qp" = 0 ]; then ok "ComfyUI is idle"
else echo "  NOTE  ComfyUI is busy: $qr job(s) running, $qp waiting (a picture you started, or an earlier test that was cut off)."
     echo "        The test picture below waits behind them. To cancel them all: rv6 swarm-doctor clear"; fi
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
# The bridge asks SwarmUI four things in a row, so it gets time; how long it took is printed
tmpf=$(mktemp)
read -r bcode btime <<<"$(curl -s -m 200 -o "$tmpf" -w '%{http_code} %{time_total}' 127.0.0.1:8188/rv6m/swarm/status 2>/dev/null)"
st=$(head -c 400000 "$tmpf"); rm -f "$tmpf"
echo "  the bridge answered with HTTP ${bcode:-nothing} after ${btime:-?} s"
if [ "$bcode" = 500 ]; then
  echo "  its error in ComfyUI's window:"
  tmux capture-pane -pt comfy -S -300 2>/dev/null | grep -n -E "Traceback|Error|rv6_mobile|File \"" | tail -12 | cut -c1-220 | sed 's/^/        /'
fi
python3 - "$st" "$bcode" <<'PY' || true
import json, sys
try: j = json.loads(sys.argv[1])
except Exception:
    why = {"000": "no answer in 200 seconds: SwarmUI is not answering the bridge's questions (timings below)", "404": "this VM's page code is old. Run: edit-restart",
           "500": "the bridge's code failed (the lines above say where)"}.get(sys.argv[2], "an answer that is not what the page expects")
    print("  PROBLEM  the edit page's bridge:", why); raise SystemExit
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
# How long SwarmUI takes for each thing the edit page's bridge asks it (a slow one is what makes the page's Swarm mode hang)
for label, name, body in (("models", "ListModels", {"subtype": "Stable-Diffusion"}), ("LoRAs", "ListModels", {"subtype": "LoRA"}),
                          ("text encoders", "ListModels", {"subtype": "Clip"}), ("presets", "GetMyUserData", {})):
    t0 = time.time()
    try:
        r = api(name, {"session_id": sid, **({"path": "", "depth": 8, "sortBy": "Name", "allowRemote": False} if name == "ListModels" else {}), **body}, 90)
        n = len(r.get("files") or r.get("presets") or [])
        print(f"    {label}: {n} in {time.time() - t0:.1f}s" + ("" if not r.get("error") else f" | SwarmUI said: {str(r.get('error'))[:200]}"), flush=True)
    except Exception as e:
        print(f"  PROBLEM  {label}: no answer after {time.time() - t0:.0f}s ({type(e).__name__})", flush=True)
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
print("  making a 1024 px test picture (loading a model can take a few minutes)...", flush=True)
import subprocess, threading
LIMIT = 900
res = {}
def work():
    try:
        res["r"] = api("GenerateText2Image", {"session_id": sid, "images": 1, "model": clean(pick), "width": 1024, "height": 1024, "seed": 1,
                                              "prompt": "a red apple on a wooden table, photo", "donotsave": True, **({"presets": auto[:1]} if auto else {})}, LIMIT + 60)
    except Exception as e:
        body = ""
        try: body = e.read().decode()[:400]
        except Exception: pass
        res["e"] = f"{type(e).__name__}: {str(e)[:200]} {body}"
def state():
    try: gpu = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader"], capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception: gpu = "?"
    try:
        q = json.load(urllib.request.urlopen("http://127.0.0.1:8188/api/queue", timeout=10))
        jobs = f"{len(q.get('queue_running') or [])} running, {len(q.get('queue_pending') or [])} waiting"
    except Exception: jobs = "ComfyUI didn't answer"
    return f"GPU {gpu} | ComfyUI jobs: {jobs}"
def stop():
    try: api("InterruptAll", {"session_id": sid, "other_sessions": False}, 30)
    except Exception: pass
th = threading.Thread(target=work, daemon=True); t = time.time(); th.start()
try:
    while th.is_alive() and time.time() - t < LIMIT:
        th.join(15)
        if th.is_alive(): print(f"    {time.time() - t:>4.0f}s  still working… {state()}", flush=True)
except KeyboardInterrupt:
    stop(); print("\n  Stopped by you; the test picture was cancelled."); raise SystemExit
took = time.time() - t
if th.is_alive():
    stop()
    print(f"  PROBLEM  no picture after {LIMIT // 60} minutes; the test was cancelled. Last state: {state()}")
    print("           If the GPU showed 0 % the whole time, the job never started: look at the lines below, and try: rv6 swarm-doctor clear")
    try:
        pane = subprocess.run(["tmux", "capture-pane", "-pt", "comfy", "-S", "-200"], capture_output=True, text=True, timeout=10).stdout
        lines = [l for l in pane.splitlines() if re.search(r"error|exception|traceback|out of memory|cuda|killed|failed", l, re.I)]
        for l in lines[-8:]: print("        comfy:", l[:220])
    except Exception: pass
elif "e" in res: print(f"  PROBLEM  the request failed after {took:.0f}s: {res['e']}")
elif res["r"].get("images"): print(f"  OK    SwarmUI made a picture in {took:.0f}s. Generating works with this model.")
else: print(f"  PROBLEM  no picture after {took:.0f}s. SwarmUI said: {str(res['r'].get('error') or res['r'])[:600]}")
PY
echo
echo "Done. Every line above that starts with PROBLEM is something to fix; if there is none, all is well. Send this whole output."
