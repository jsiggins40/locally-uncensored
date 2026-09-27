#!/usr/bin/env bash
#
# A plain form that trains a face LoRA.
#
#   bash t2.sh            start on port 7864
#
# Upload twenty-odd photos of one person, give them a trigger word, press
# go. An hour or two later the LoRA is in ComfyUI's folder and the edit
# form offers it like any other.
#
# No JavaScript, same as the rest. Progress arrives by meta refresh, and
# the sample images the trainer makes along the way are the honest signal:
# if step 750 does not look like the person, no number of further steps
# will fix it - the dataset will.

set -uo pipefail

PORT="${PORT:-7864}"
TRAINER="${TRAINER_DIR:-$HOME/ai-toolkit}"
ROOT="${TRAIN_ROOT:-$HOME/training}"
COMFY_LORAS="${COMFY_LORAS:-$HOME/ComfyUI/models/loras}"
CREDS="$HOME/train-credentials.txt"
SCRIPT="$HOME/train-form.py"
LOG="$HOME/train-form.log"

say() { printf '\n=== %s ===\n' "$*"; }
die() { printf '\nFAILED: %s\n' "$*"; exit 1; }

[ -x "$TRAINER/venv/bin/python" ] || die "no trainer at $TRAINER (run add-trainer.sh first)"
[ -s "$HOME/.trainer-profile" ] || die "no trainer profile (run add-trainer.sh first)"
BASE=$(sed -n 1p "$HOME/.trainer-profile")
PROFILE=$(sed -n 2p "$HOME/.trainer-profile")
mkdir -p "$ROOT/datasets" "$ROOT/output" "$COMFY_LORAS"

# iPhones hand over HEIC, which nothing in this chain reads.
"$TRAINER/venv/bin/python" -c 'import pillow_heif' 2>/dev/null \
  || "$TRAINER/venv/bin/pip" install -q pillow-heif 2>&1 | tail -1

say "writing the server"
cat > "$SCRIPT" <<'PYEOF'
#!/usr/bin/env python3
"""A JavaScript-free form that trains one face into a LoRA."""

import argparse, base64, hmac, html, json, os, re, shutil, subprocess, sys
import threading, time, urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MAX_BYTES = 512 * 1024 * 1024        # thirty photos off a phone
PIC = (".png", ".jpg", ".jpeg", ".webp", ".heic", ".heif")


def parse_multipart(body, boundary):
    """Yield (field, filename, bytes). A list, because one field carries
    every photo - keying by name would leave you training on one image."""
    for chunk in body.split(b"--" + boundary):
        head, sep, data = chunk.partition(b"\r\n\r\n")
        if not sep:
            continue
        disp = next((l for l in head.split(b"\r\n")
                     if l.lower().startswith(b"content-disposition:")), b"")
        if not disp:
            continue
        name = re.search(rb'name="([^"]*)"', disp)
        fn = re.search(rb'filename="([^"]*)"', disp)
        yield (name.group(1).decode() if name else "",
               fn.group(1).decode() if fn else None,
               data[:-2] if data.endswith(b"\r\n") else data)


def safe(n):
    return re.sub(r"[^A-Za-z0-9._-]", "_", os.path.basename(n or "x"))[:80] or "x"


def slug(n):
    return re.sub(r"[^a-z0-9_]", "_", (n or "").lower()).strip("_")[:40]


def to_png(blob, path):
    """Whatever came off the phone, written as something trainable."""
    from PIL import Image
    import io as _io
    try:
        import pillow_heif
        pillow_heif.register_heif_opener()
    except Exception:
        pass
    im = Image.open(_io.BytesIO(blob))
    if im.mode not in ("RGB", "L"):
        im = im.convert("RGB")
    im.save(path, "PNG")
    return im.size


class Run:
    def __init__(self):
        self.lock = threading.Lock()
        self.reset()

    def reset(self):
        self.active = False
        self.name = ""
        self.trigger = ""
        self.step = 0
        self.total = 0
        self.loss = ""
        self.started = 0.0
        self.error = ""
        self.done_file = ""
        self.images = 0
        self.tail = ""

    def snapshot(self):
        with self.lock:
            return dict(active=self.active, name=self.name, step=self.step,
                        total=self.total, loss=self.loss, error=self.error,
                        started=self.started, done_file=self.done_file,
                        trigger=self.trigger, images=self.images,
                        tail=self.tail)


RUN = Run()

CONFIG = """---
job: extension
config:
  name: "{name}"
  process:
    - type: 'diffusion_trainer'
      training_folder: "{outdir}"
      device: cuda:0
      network:
        type: "lora"
        linear: {rank}
        linear_alpha: {rank}
      save:
        dtype: float16
        save_every: {save_every}
        max_step_saves_to_keep: 2
      datasets:
        - folder_path: "{dataset}"
          caption_ext: "txt"
          caption_dropout_rate: 0.05
          resolution: [ 512, 768, 1024 ]
          trigger_word: "{trigger}"
      train:
        batch_size: 1
        cache_text_embeddings: true
        steps: {steps}
        gradient_accumulation: 1
        timestep_type: "weighted"
        train_unet: true
        train_text_encoder: false
        gradient_checkpointing: true
        noise_scheduler: "flowmatch"
        optimizer: "adamw8bit"
        lr: 1e-4
        dtype: bf16
      model:
        name_or_path: "{base}"
        arch: "qwen_image_edit_plus"
        quantize: true
        qtype: "{qtype}"
        quantize_te: true
        qtype_te: "qfloat8"
        low_vram: true
      sample:
{sample}
meta:
  name: "[name]"
  version: '1.0'
"""

SAMPLE_ON = """        sampler: "flowmatch"
        sample_every: {every}
        sample_start_step: 0
        width: 1024
        height: 1024
        samples:
          - prompt: "a photo of {trigger}, looking at the camera"
            ctrl_img_1: "{ctrl}"
          - prompt: "{trigger} standing outdoors in daylight"
            ctrl_img_1: "{ctrl}"
        neg: ""
        seed: 42
        walk_seed: true
        guidance_scale: 3
        sample_steps: 25"""

SAMPLE_OFF = """        sampler: "flowmatch"
        disable_sampling: true"""

# uint3 with an accuracy-recovery adapter is what makes this fit on 32GB.
# It costs quality, so it is not used where there is room for 8-bit.
QTYPE = {
    "8bit": "qfloat8",
    "3bit": "uint3|ostris/accuracy_recovery_adapters/"
            "qwen_image_edit_2509_torchao_uint3.safetensors",
}
PYEOF
echo "  (part 1)"

cat >> "$SCRIPT" <<'PYEOF'


STEP_RE = re.compile(r"(\d+)\s*/\s*(\d+)")
LOSS_RE = re.compile(r"loss:\s*([0-9.]+e?[-+]?\d*)")


def watch(proc, logpath):
    """Follow the trainer's output and keep the numbers current.

    It writes a tqdm bar, so progress arrives on carriage returns rather
    than newlines - reading by line would sit silent for an hour and then
    print everything at once.
    """
    buf = ""
    with open(logpath, "ab", buffering=0) as raw:
        while True:
            ch = proc.stdout.read(1)
            if not ch:
                break
            raw.write(ch)
            c = ch.decode("utf-8", "replace")
            if c in "\r\n":
                line, buf = buf, ""
                if not line.strip():
                    continue
                m = STEP_RE.search(line)
                l = LOSS_RE.search(line)
                with RUN.lock:
                    if m and int(m.group(2)) > 1:
                        RUN.step, RUN.total = int(m.group(1)), int(m.group(2))
                    if l:
                        RUN.loss = l.group(1)
                    if len(line) > 4:
                        RUN.tail = line.strip()[-300:]
            else:
                buf += c
                if len(buf) > 4000:
                    buf = buf[-2000:]


def trainer(cfg):
    """Write the dataset out, write a config, run it, put the result away."""
    try:
        name, root = cfg["name"], cfg["root"]
        dataset = os.path.join(root, "datasets", name)
        outdir = os.path.join(root, "output")
        os.makedirs(dataset, exist_ok=True)
        os.makedirs(outdir, exist_ok=True)

        kept = 0
        for i, (fn, blob) in enumerate(cfg["photos"]):
            dst = os.path.join(dataset, "%03d.png" % i)
            try:
                to_png(blob, dst)
            except Exception as e:
                print("skipping %s: %s" % (fn, e), file=sys.stderr)
                continue
            with open(dst[:-4] + ".txt", "w") as fh:
                fh.write(cfg["caption"])
            kept += 1
        with RUN.lock:
            RUN.images = kept
        if kept < 5:
            raise RuntimeError("only %d usable photos - train on 15 to 30" % kept)

        first = os.path.join(dataset, "000.png")
        sample = (SAMPLE_ON.format(every=cfg["save_every"],
                                   trigger=cfg["trigger"], ctrl=first)
                  if cfg["samples"] else SAMPLE_OFF)
        text = CONFIG.format(
            name=name, outdir=outdir, dataset=dataset, base=cfg["base"],
            steps=cfg["steps"], save_every=cfg["save_every"],
            rank=cfg["rank"], trigger=cfg["trigger"],
            qtype=QTYPE[cfg["profile"]], sample=sample)
        cpath = os.path.join(root, name + ".yaml")
        with open(cpath, "w") as fh:
            fh.write(text)

        logpath = os.path.join(root, name + ".log")
        open(logpath, "wb").close()
        env = dict(os.environ, PYTHONUNBUFFERED="1", HF_HUB_ENABLE_HF_TRANSFER="0")
        proc = subprocess.Popen(
            [os.path.join(cfg["trainer"], "venv/bin/python"), "run.py", cpath],
            cwd=cfg["trainer"], stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, env=env)
        with RUN.lock:
            RUN.total = cfg["steps"]
        watch(proc, logpath)
        rc = proc.wait()
        if rc != 0:
            with open(logpath, "rb") as fh:
                fh.seek(max(0, os.path.getsize(logpath) - 1200))
                tail = fh.read().decode("utf-8", "replace")
            raise RuntimeError("the trainer stopped (exit %d):\n%s" % (rc, tail))

        # Newest weights win: the last periodic save is what finished.
        made = []
        for base, _d, files in os.walk(os.path.join(outdir, name)):
            for f in files:
                if f.endswith(".safetensors"):
                    p = os.path.join(base, f)
                    made.append((os.path.getmtime(p), p))
        if not made:
            raise RuntimeError("training finished but produced no .safetensors")
        made.sort()
        src = made[-1][1]
        dst = os.path.join(cfg["loras"], name + ".safetensors")
        shutil.copy2(src, dst)
        with RUN.lock:
            RUN.done_file = os.path.basename(dst)
            RUN.active = False
    except Exception as e:
        with RUN.lock:
            RUN.error = str(e)[:2000]
            RUN.active = False


PAGE = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
{refresh}<title>Train</title><style>
:root{{color-scheme:dark light}}
body{{font:17px/1.5 -apple-system,system-ui,sans-serif;margin:0;padding:20px 16px;max-width:640px}}
h1{{font-size:22px;margin:0 0 2px}} p.sub{{color:#888;margin:0 0 18px;font-size:15px}}
form{{border:1px solid #8884;border-radius:12px;padding:18px}}
label{{display:block;font-size:14px;color:#888;margin:14px 0 4px}}
input[type=text],input[type=number],input[type=file],select{{width:100%;
  box-sizing:border-box;font-size:17px;padding:10px;border-radius:8px;
  border:1px solid #8886;background:#8881;color:inherit}}
button{{font-size:17px;padding:13px;width:100%;margin-top:18px;border:0;
  border-radius:10px;background:#6a4fb5;color:#fff}}
.live{{border:1px solid #8884;border-radius:12px;padding:14px 16px;margin-bottom:20px}}
.bar{{height:10px;border-radius:5px;background:#8883;overflow:hidden;margin:8px 0 6px}}
.bar>i{{display:block;height:100%;background:#6a4fb5}}
.grid{{display:grid;gap:10px;grid-template-columns:repeat(auto-fill,minmax(150px,1fr))}}
.grid img{{width:100%;border-radius:8px;display:block;background:#8882}}
pre{{white-space:pre-wrap;background:#8881;border-radius:9px;padding:10px;
  font-size:13px;color:#888;max-height:8em;overflow:hidden}}
.note{{color:#888;font-size:14px}} .err{{color:#c55}} .ok{{color:#4a4}}
.fmt label{{display:inline-block;color:inherit;font-size:16px}}
.fmt input{{width:auto;margin-right:6px}}
</style></head><body>
<h1>Train a face</h1>
<p class="sub">{status}</p>
{msg}
{live}
<form method="post" enctype="multipart/form-data" action="/">
  <label>Photos of one person &mdash; 15 to 30, varied</label>
  <input type="file" name="photos" accept="image/*" multiple>

  <label>Name for the LoRA</label>
  <input type="text" name="name" value="{name}" placeholder="james">

  <label>Trigger word &mdash; a word the model does not already know</label>
  <input type="text" name="trigger" value="{trigger}" placeholder="jsggns">

  <label>What the photos are of</label>
  <input type="text" name="caption" value="{caption}" placeholder="a photo of a man">

  <div class="fmt"><label><input type="checkbox" name="samples" checked>
    make preview images while it trains</label></div>

  <label>Steps</label>
  <input type="number" name="steps" value="{steps}" min="250" max="6000" step="250">
  <label>Rank &mdash; 16 is plenty for one face</label>
  <input type="number" name="rank" value="{rank}" min="4" max="64" step="4">

  <p class="note">Varied is what matters: different angles, lighting,
  distances, expressions; no sunglasses, no heavy filters, nobody else in
  frame. Twenty good photos beat fifty similar ones. Crop to the head and
  shoulders where you can.<br><br>
  About 1500 steps for a face, an hour or two. It keeps going if you close
  the page.</p>
  <button type="submit">Start training</button>
</form>
</body></html>"""


class H(BaseHTTPRequestHandler):
    root = ""
    trainer = ""
    loras = ""
    base = ""
    profile = "3bit"
    credential = ""
    last = {"name": "", "trigger": "", "caption": "a photo of a man",
            "steps": "1500", "rank": "16"}

    def log_message(self, f, *a):
        sys.stderr.write("%s %s\n" % (self.address_string(), f % a))

    def authed(self):
        if not self.credential:
            return True
        want = "Basic " + base64.b64encode(self.credential.encode()).decode()
        if hmac.compare_digest(self.headers.get("Authorization", ""), want):
            return True
        self.send_response(401)
        self.send_header("WWW-Authenticate", 'Basic realm="train"')
        self.send_header("Content-Length", "0")
        self.end_headers()
        return False

    def samples_of(self, name, limit=6):
        d = os.path.join(self.root, "output", name, "samples")
        if not os.path.isdir(d):
            return []
        out = []
        for f in os.listdir(d):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                p = os.path.join(d, f)
                try:
                    out.append((os.path.getmtime(p), f))
                except OSError:
                    pass
        out.sort(reverse=True)
        return [f for _m, f in out[:limit]]

    def live_block(self):
        s = RUN.snapshot()
        if not s["name"]:
            return "", ""
        el = int(time.time() - s["started"]) if s["started"] else 0
        elapsed = "%d:%02d:%02d" % (el // 3600, el // 60 % 60, el % 60)
        out = ['<div class="live">']
        refresh = ""
        if s["active"]:
            refresh = '<meta http-equiv="refresh" content="10">'
            pct = int(100.0 * s["step"] / s["total"]) if s["total"] else 0
            out.append('<p><b>{}</b> &middot; {} photos</p>'.format(
                html.escape(s["name"]), s["images"]))
            out.append('<div class="bar"><i style="width:{}%"></i></div>'.format(pct))
            if s["step"]:
                left = int(el / s["step"] * (s["total"] - s["step"])) if s["step"] else 0
                out.append('<p class="note">step {} of {} &middot; loss {} '
                           '&middot; {} elapsed &middot; about {}h{:02d}m left</p>'
                           .format(s["step"], s["total"], s["loss"] or "?",
                                   elapsed, left // 3600, left // 60 % 60))
            else:
                out.append('<p class="note">{} elapsed. The base model is '
                           'about 60GB and downloads once, on the first run - '
                           'that is what the long silence at the start is.</p>'
                           .format(elapsed))
            pics = self.samples_of(s["name"])
            if pics:
                out.append('<p class="note">previews, newest first &mdash; if '
                           'these do not look like the person by a third of '
                           'the way in, stop and fix the photos</p>')
                out.append('<div class="grid">' + "".join(
                    '<img src="/sample?j={}&n={}" alt="">'.format(
                        urllib.parse.quote(s["name"]), urllib.parse.quote(f))
                    for f in pics) + '</div>')
            if s["tail"]:
                out.append("<pre>" + html.escape(s["tail"]) + "</pre>")
        elif s["error"]:
            out.append('<p class="err">{}</p>'.format(
                html.escape(s["error"]).replace("\n", "<br>")))
            out.append('<p class="note">If it died while making a preview, '
                       'untick the preview box and start it again.</p>')
        elif s["done_file"]:
            out.append('<p class="ok">done in {} &mdash; {}</p>'.format(
                elapsed, html.escape(s["done_file"])))
            out.append('<p class="note">It is in ComfyUI\'s loras folder. '
                       'Restart the edit form and it will be in the list. '
                       'Use it by putting <b>{}</b> in the prompt.</p>'
                       .format(html.escape(s["trigger"])))
            pics = self.samples_of(s["name"])
            if pics:
                out.append('<div class="grid">' + "".join(
                    '<img src="/sample?j={}&n={}" alt="">'.format(
                        urllib.parse.quote(s["name"]), urllib.parse.quote(f))
                    for f in pics) + '</div>')
        out.append('</div>')
        return refresh, "".join(out)

    def render(self, msg=""):
        refresh, live = self.live_block()
        body = PAGE.format(
            refresh=refresh, live=live, msg=msg,
            status=html.escape("%s &middot; %s" % (self.base, self.profile))
            .replace("&amp;middot;", "&middot;"),
            name=html.escape(self.last["name"]),
            trigger=html.escape(self.last["trigger"]),
            caption=html.escape(self.last["caption"]),
            steps=html.escape(self.last["steps"]),
            rank=html.escape(self.last["rank"])).encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if not self.authed():
            return
        u = urllib.parse.urlsplit(self.path)
        if u.path == "/sample":
            q = urllib.parse.parse_qs(u.query)
            d = os.path.join(self.root, "output", q.get("j", [""])[0], "samples")
            p = os.path.abspath(os.path.join(d, q.get("n", [""])[0]))
            if not p.startswith(os.path.abspath(d) + os.sep) or not os.path.isfile(p):
                self.send_response(404); self.send_header("Content-Length", "0")
                self.end_headers(); return
            blob = open(p, "rb").read()
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg"
                             if p.lower().endswith((".jpg", ".jpeg")) else "image/png")
            self.send_header("Content-Length", str(len(blob)))
            self.end_headers(); self.wfile.write(blob); return
        self.render()

    def do_POST(self):
        if not self.authed():
            return
        ctype = self.headers.get("Content-Type", "")
        m = re.search(r'boundary=(?:"([^"]+)"|([^;]+))', ctype)
        if not m:
            return self.render('<p class="err">not a form post</p>')
        n = int(self.headers.get("Content-Length") or 0)
        if n <= 0 or n > MAX_BYTES:
            return self.render('<p class="err">too much at once - '
                               'try fewer photos</p>')
        body = self.rfile.read(n)

        fields, photos = {}, []
        for fname, fn, data in parse_multipart(
                body, (m.group(1) or m.group(2)).strip().encode()):
            if fn and data and os.path.splitext(fn)[1].lower() in PIC:
                photos.append((safe(fn), data))
            elif fname and not fn:
                fields[fname] = data.decode("utf-8", "replace").strip()

        for k in ("name", "trigger", "caption", "steps", "rank"):
            if k in fields:
                H.last[k] = fields[k]

        # Read the flag, then let go: render() reads RUN itself, and a
        # plain Lock is not reentrant - holding it across the call hangs
        # the request rather than answering it.
        with RUN.lock:
            busy = RUN.active
        if busy:
            return self.render('<p class="err">something is already '
                               'training. One at a time - they each want '
                               'the whole card.</p>')
        name = slug(fields.get("name", ""))
        trigger = fields.get("trigger", "").strip()
        if not name:
            return self.render('<p class="err">give it a name</p>')
        if not trigger or " " in trigger:
            return self.render('<p class="err">the trigger must be one word, '
                               'and an odd one - a real word already means '
                               'something to the model</p>')
        if len(photos) < 5:
            return self.render('<p class="err">only {} photos came through. '
                               'Pick 15 to 30.</p>'.format(len(photos)))
        try:
            steps = max(250, min(6000, int(fields.get("steps") or 1500)))
            rank = max(4, min(64, int(fields.get("rank") or 16)))
        except ValueError:
            return self.render('<p class="err">steps and rank are numbers</p>')

        cfg = dict(name=name, trigger=trigger,
                   caption=(fields.get("caption") or "a photo of a person"),
                   steps=steps, rank=rank, save_every=max(100, steps // 6),
                   samples=bool(fields.get("samples")),
                   photos=photos, root=self.root, trainer=self.trainer,
                   loras=self.loras, base=self.base, profile=self.profile)
        with RUN.lock:
            RUN.reset()
            RUN.active = True
            RUN.name = name
            RUN.trigger = trigger
            RUN.total = steps
            RUN.started = time.time()
        threading.Thread(target=trainer, args=(cfg,), daemon=True).start()
        self.send_response(303)
        self.send_header("Location", "/")
        self.send_header("Content-Length", "0")
        self.end_headers()


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--port", type=int, default=7864)
    a.add_argument("--root", required=True)
    a.add_argument("--trainer", required=True)
    a.add_argument("--loras", required=True)
    a.add_argument("--base", required=True)
    a.add_argument("--profile", default="3bit")
    a.add_argument("--user", default="")
    a.add_argument("--password", default="")
    o = a.parse_args()
    H.root, H.trainer = os.path.abspath(o.root), os.path.abspath(o.trainer)
    H.loras, H.base, H.profile = os.path.abspath(o.loras), o.base, o.profile
    if not (o.user and o.password):
        print("refusing to start without a password", file=sys.stderr)
        return 2
    H.credential = "{}:{}".format(o.user, o.password)
    print("base %s, %s, output in %s" % (H.base, H.profile, H.root))
    print("serving on 0.0.0.0:%d" % o.port)
    ThreadingHTTPServer(("0.0.0.0", o.port), H).serve_forever()


if __name__ == "__main__":
    sys.exit(main() or 0)
PYEOF

python3 -c "compile(open('$SCRIPT').read(),'$SCRIPT','exec')" || die "embedded server is broken"

if [ ! -s "$CREDS" ]; then
  printf 'user: train\npass: %s\n' "$(openssl rand -base64 15 | tr -d '/+=')" > "$CREDS"
  chmod 600 "$CREDS"
fi
U=$(awk '/^user:/{print $2}' "$CREDS")
P=$(awk '/^pass:/{print $2}' "$CREDS")

say "starting"
tmux kill-session -t =trainform 2>/dev/null
tmux new-session -d -s trainform \
  "'$TRAINER/venv/bin/python' -u '$SCRIPT' --port $PORT --root '$ROOT' --trainer '$TRAINER' --loras '$COMFY_LORAS' --base '$BASE' --profile '$PROFILE' --user '$U' --password '$P' 2>&1 | tee -a '$LOG'"
sleep 2
tmux has-session -t =trainform 2>/dev/null || { tail -20 "$LOG"; die "did not start"; }
CODE=$(curl -sS -o /dev/null -w '%{http_code}' -u "$U:$P" "http://127.0.0.1:$PORT/" 2>/dev/null)

cat <<EOF

=== done ===

  port      $PORT   (http $CODE)
  base      $BASE
  profile   $PROFILE
  output    $ROOT/output
  log       tail -n 40 $LOG
  stop      tmux kill-session -t =trainform

$(cat "$CREDS")

Forward port $PORT and open it. The first run downloads the base model,
about 60GB, before step 1 - so nothing appears to happen for a while.
EOF
