#!/usr/bin/env bash
#
# A plain form in front of ComfyUI, for editing photos from a phone.
#
#   bash e.sh            start on port 7862
#   PORT=7863 bash e.sh
#
# ComfyUI's canvas is a poor fit for a phone even at full speed, and under
# Safari's Lockdown Mode - which disables JIT - it crawls. This serves one
# form: pick a photo, type an instruction, submit. No JavaScript, so
# Lockdown has nothing to block, and no node graph to drag around.
#
# It builds the workflow itself and posts it to ComfyUI's API. Rather than
# hardcoding node names, it reads /object_info at startup and adapts - node
# signatures move between ComfyUI releases, and a graph that silently wires
# the wrong input is worse than one that refuses to start.
#
# /status shows the image forming. ComfyUI streams step counts and preview
# frames over a websocket that a browser would normally hold - this server
# holds it instead and keeps the last frame in memory, so the page itself
# is static HTML on a meta refresh and still works with JavaScript off.
# ComfyUI must be started with --preview-method auto for the frames.

set -uo pipefail

PORT="${PORT:-7862}"
COMFY_URL="${COMFY_URL:-http://127.0.0.1:8288}"
COMFY_DIR="${COMFY_DIR:-$HOME/ComfyUI}"
# The form's input and output folders are passed explicitly, derived from
# COMFY_DIR, because its own defaults expand ~ at runtime - and the form
# and ComfyUI need not agree on what ~ is. tmux hands a session the
# environment its *server* started with, so a form launched through tmux
# can hold a different HOME than the ComfyUI it talks to. That put uploads
# in /home/ComfyUI/input while ComfyUI read /home/work/ComfyUI/input, and
# every edit failed with "Invalid image file" for a file that was fine.
CREDS="$HOME/edit-form-credentials.txt"
SCRIPT="$HOME/edit-form.py"
LOG="$HOME/edit-form.log"
# One session name per port, so a second copy on another port does not
# kill the first. The default port keeps the plain name, which is what
# the bootstrap and every earlier instruction refer to.
SESSION="editform"
[ "$PORT" = "7862" ] || { SESSION="editform$PORT"; LOG="$HOME/edit-form-$PORT.log"; }
exec > >(tee -a "$LOG") 2>&1

say() { printf '\n=== %s ===\n' "$*"; }
die() { printf '\nFAILED: %s\n' "$*"; exit 1; }

[ -d "$COMFY_DIR" ] || die "no ComfyUI at $COMFY_DIR"
curl -sS -o /dev/null -m 5 "$COMFY_URL/object_info" \
  || die "ComfyUI is not answering at $COMFY_URL (tmux attach -t comfy)"

say "writing the server"
cat > "$SCRIPT" <<'PYEOF'
#!/usr/bin/env python3
"""A JavaScript-free form that drives ComfyUI's API.

Node signatures change between ComfyUI releases - the encoder for Qwen edit
has been TextEncodeQwenImageEdit and TextEncodeQwenImageEditPlus, taking
`image` in one version and `image1` in another. So the graph is assembled
from what /object_info actually reports rather than from a fixed template,
and if a required node is absent the page says which one instead of posting
a graph that fails somewhere inside ComfyUI.
"""

import argparse, base64, hmac, html, json, os, re, shutil, socket, struct, zlib
import subprocess, sys, threading, time, urllib.parse, urllib.request, uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MAX_BYTES = 64 * 1024 * 1024
# A LoRA is a few hundred megabytes, and some of them will not come down
# over Civitai's API at all - only a signed-in browser gets them. So the
# form takes one as an upload.
MAX_LORA = 800 * 1024 * 1024
IMG_EXT = (".png", ".jpg", ".jpeg", ".webp", ".webm", ".mp4")
VID_EXT = (".webm", ".mp4")
MIME = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".webp": "image/webp", ".gif": "image/gif",
        ".webm": "video/webm", ".mp4": "video/mp4"}


MASK_COLS = 24          # across; rows follow the photo's own proportions
MASK_W = 320            # how wide the photo is drawn, in CSS pixels.
# It has to be a number the server knows, not 100%: an <input type=image>
# reports where it was tapped in the element's own pixels, and that is the
# only way a page with no JavaScript can find out where someone pointed.
MASK_PREFIX = "_mask_"  # masks live beside the photos, hidden from the picker

BRUSH_JS = """
<script>
(function(){
 var c=document.getElementById('pc'); if(!c||!c.getContext) return;
 var f=document.getElementById('f'), pd=document.getElementById('pd');
 if(!f||!pd) return;
 document.documentElement.className+=' js';
 var x=c.getContext('2d'), size=Math.round(c.width/8), erase=false, down=false;
 function wipe(){x.fillStyle='#000';x.fillRect(0,0,c.width,c.height);}
 wipe();
 function at(e){var r=c.getBoundingClientRect(),
   t=(e.touches&&e.touches[0])||e;
   return [(t.clientX-r.left)*c.width/r.width,
           (t.clientY-r.top)*c.height/r.height];}
 function dab(p){x.fillStyle=erase?'#000':'#fff';
   x.beginPath();x.arc(p[0],p[1],size/2,0,6.2832);x.fill();}
 function line(a,b){x.strokeStyle=erase?'#000':'#fff';x.lineWidth=size;
   x.lineCap='round';x.lineJoin='round';x.beginPath();
   x.moveTo(a[0],a[1]);x.lineTo(b[0],b[1]);x.stroke();}
 var last=null;
 function start(e){down=true;last=at(e);dab(last);e.preventDefault();}
 function move(e){if(!down)return;var p=at(e);line(last,p);last=p;
   e.preventDefault();}
 function end(){down=false;last=null;}
 c.addEventListener('touchstart',start,{passive:false});
 c.addEventListener('touchmove',move,{passive:false});
 c.addEventListener('touchend',end);
 c.addEventListener('touchcancel',end);
 c.addEventListener('mousedown',start);
 c.addEventListener('mousemove',move);
 window.addEventListener('mouseup',end);
 function on(id,fn){var b=document.getElementById(id); if(b)b.onclick=fn;}
 on('bsm',function(){size=Math.max(4,Math.round(size/1.5));});
 on('bbg',function(){size=Math.min(c.width,Math.round(size*1.5));});
 on('ber',function(){erase=!erase;this.textContent=erase?'Erasing':'Erase';});
 on('bcl',function(){wipe();});
 f.addEventListener('submit',function(){
   var d=x.getImageData(0,0,c.width,c.height).data, any=false;
   for(var i=0;i<d.length;i+=4){if(d[i]>127){any=true;break;}}
   pd.value=any?c.toDataURL('image/png'):'';
 });
})();
</script>
"""

MASK = """
  <label>Mask &mdash; paint what may change</label>
  <input type="hidden" name="mcells" value="{cells}">
  <input type="hidden" name="paintdata" id="pd" value="">
  <div class="maskwrap" style="width:{w}px">
    <input type="image" name="tap" src="/src?n={src}" width="{w}" alt="tap the photo">
    <div class="cells" style="grid-template-columns:repeat({cols},1fr)">{tiles}</div>
    <canvas id="pc" width="{cw}" height="{ch}"></canvas>
  </div>
  <div class="brush">
    <div class="row ops">
      <button type="button" id="bsm" class="second">Smaller</button>
      <button type="button" id="bbg" class="second">Bigger</button>
      <button type="button" id="ber" class="second">Erase</button>
      <button type="button" id="bcl" class="second">Clear</button>
    </div>
    <p class="note">Drag your finger over what should change. <b>Erase</b>
    toggles the brush to rub out. Everything outside the paint is left exactly
    as photographed.</p>
  </div>
  <div class="tapping">
    <p class="note">{state}</p>
    <div class="row ops">
      <button type="submit" formaction="/mask?op=grow" class="second">Grow</button>
      <button type="submit" formaction="/mask?op=invert" class="second">Invert</button>
      <button type="submit" formaction="/mask?op=clear" class="second">Clear</button>
    </div>
    <p class="note">Tap two opposite corners of the area that may change and
    everything between them fills in.</p>
  </div>
  <p class="note">Nothing outside the marked area is touched at all, so Denoise
  applies only inside &mdash; which is what lets you put it to 1.0 on a shirt and
  keep the face exactly as photographed. Leave it empty to edit the whole
  picture as before.</p>
"""


def image_size(path):
    """(width, height) of a PNG or JPEG, from the header alone.

    Needed so a mask can be written at the photo's own proportions - a
    square mask over a portrait would stretch across the resize and the
    marked region would land somewhere else. Reading two headers by hand
    is cheaper than a dependency; this whole form is stdlib on purpose.
    """
    try:
        with open(path, "rb") as fh:
            head = fh.read(32)
            if head[:8] == b"\x89PNG\r\n\x1a\n":
                w, h = struct.unpack(">II", head[16:24])
                return int(w), int(h)
            if head[:2] != b"\xff\xd8":
                return 0, 0
            fh.seek(2)
            while True:
                b = fh.read(1)
                if not b:
                    return 0, 0
                if b != b"\xff":
                    continue
                marker = fh.read(1)
                while marker == b"\xff":        # fill bytes are legal
                    marker = fh.read(1)
                if not marker:
                    return 0, 0
                m = marker[0]
                if m in (0xD8, 0xD9) or 0xD0 <= m <= 0xD7:
                    continue                     # no length field on these
                size = struct.unpack(">H", fh.read(2))[0]
                # Every frame-start marker but the two that are not frames.
                if 0xC0 <= m <= 0xCF and m not in (0xC4, 0xC8, 0xCC):
                    h, w = struct.unpack(">HH", fh.read(5)[1:5])
                    return int(w), int(h)
                fh.seek(size - 2, 1)
    except Exception:
        return 0, 0


PAINT_MASK = """
import sys
from PIL import Image, ImageChops, ImageFilter
orig, painted, out = sys.argv[1], sys.argv[2], sys.argv[3]
a = Image.open(orig).convert("RGB")
b = Image.open(painted).convert("RGB")
if b.size != a.size:                      # a screenshot instead of a copy
    b = b.resize(a.size, Image.LANCZOS)
d = ImageChops.difference(a, b).convert("L")
# Markup strokes are opaque and nothing else in the picture moved, so the
# difference is close to binary already. The threshold only has to clear
# jpeg noise.
m = d.point(lambda v: 255 if v > 28 else 0)
m = m.filter(ImageFilter.MaxFilter(9))    # close the gaps in a loose scribble
m = m.filter(ImageFilter.MinFilter(5))    # and take back the spread
w, h = m.size
if w > 1024 or h > 1024:
    k = 1024.0 / max(w, h)
    m = m.resize((max(1, int(w * k)), max(1, int(h * k))), Image.LANCZOS)
    m = m.point(lambda v: 255 if v > 96 else 0)
white = sum(m.point(lambda v: 1 if v else 0).getdata())
print("%d %d %d" % (m.size[0], m.size[1], white))
m.save(out)
"""


def mask_from_paint(orig, painted, out, comfy_dir, timeout=120):
    """Where the photo was painted over, as a mask.

    The form itself has no image library on purpose - it is stdlib all
    the way down - so this borrows ComfyUI's, the same way joining video
    clips borrows its PyAV. Pillow is certain to be there: ComfyUI cannot
    run without it.
    """
    py = os.path.join(comfy_dir, "venv", "bin", "python")
    if not os.path.exists(py):
        raise RuntimeError("no ComfyUI venv to borrow Pillow from")
    r = subprocess.run([py, "-c", PAINT_MASK, orig, painted, out],
                       capture_output=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.decode("utf-8", "replace").strip()[-300:]
                           or "could not read the painted copy")
    try:
        w, h, white = (int(x) for x in r.stdout.decode().split())
    except Exception:
        raise RuntimeError("the mask came back unreadable")
    if not white:
        raise RuntimeError("nothing looks painted on")
    return w, h, white


def paint_png(data_url, path):
    """Save a canvas the brush painted, which is already the mask.

    The canvas is black where nothing was painted and white where it was,
    which is exactly what SetLatentNoiseMask wants - so unlike everything
    else here it needs no image library at all, only base64.
    """
    head, _, b64 = (data_url or "").partition(",")
    if not b64 or "image/png" not in head or "base64" not in head:
        return False
    try:
        raw = base64.b64decode(b64, validate=True)
    except Exception:
        return False
    if raw[:8] != b"\x89PNG\r\n\x1a\n" or len(raw) > 8 * 1024 * 1024:
        return False
    with open(path, "wb") as fh:
        fh.write(raw)
    return True


def tap_cell(sx, sy, disp_w, w, h, cols, rows):
    """Which cell a tap landed in, or None if it was not a tap.

    The browser reports the hit in the element's own pixels, so the width
    it is drawn at has to be a number we chose rather than a percentage -
    otherwise there is nothing to divide by.
    """
    if not (sx and sy and str(sx).isdigit() and str(sy).isdigit()):
        return None
    disp_h = max(1, int(round(disp_w * (h or 1) / float(w or 1))))
    c = min(cols - 1, max(0, int(sx) * cols // max(1, disp_w)))
    r = min(rows - 1, max(0, int(sy) * rows // max(1, disp_h)))
    return r, c


def mask_op(op, marks, cols, rows):
    """Tapping 220 cells one at a time is nobody's idea of a good evening.

    Every one of these is worked out on the server from the ticks that
    came with the form, which is the only place a page with no JavaScript
    can do anything at all.
    """
    cur = set()
    for v in marks or ():
        r, _, c = str(v).partition("-")
        if r.isdigit() and c.isdigit():
            cur.add((int(r), int(c)))
    if op == "clear":
        cur = set()
    elif op == "invert":
        cur = {(r, c) for r in range(rows) for c in range(cols)} - cur
    elif op == "box" and cur:
        rs = [r for r, _ in cur]
        cs = [c for _, c in cur]
        cur = {(r, c)
               for r in range(min(rs), max(rs) + 1)
               for c in range(min(cs), max(cs) + 1)}
    elif op == "grow" and cur:
        cur = {(r + dr, c + dc) for r, c in cur
               for dr in (-1, 0, 1) for dc in (-1, 0, 1)
               if 0 <= r + dr < rows and 0 <= c + dc < cols}
    return sorted("%d-%d" % rc for rc in cur)


def write_mask_png(path, cols, rows, cells, w=0, h=0):
    """A greyscale PNG: white where the sampler may work, black elsewhere.

    Written by hand because the form has no image library, and a mask is
    the one picture simple enough to emit that way - one byte a pixel,
    one filter byte a row, deflate, three chunks.
    """
    # Shrink to something cheap to write, but keep the photo's proportions:
    # ComfyUI stretches the mask onto the latent, so a square mask over a
    # portrait would slide the marked region somewhere it was never put.
    cap = 1024
    if w and h:
        if max(w, h) > cap:
            if w >= h:
                w, h = cap, max(1, int(round(h * cap / float(w))))
            else:
                w, h = max(1, int(round(w * cap / float(h)))), cap
    else:
        w, h = cols * 64, rows * 64
    w, h = max(cols, w), max(rows, h)
    on = set(cells or ())
    raw = bytearray()
    for y in range(h):
        r = min(rows - 1, y * rows // h)
        row = bytes(255 if (r, min(cols - 1, x * cols // w)) in on else 0
                    for x in range(w))
        raw += b"\x00" + row                     # filter 0: none
    def chunk(tag, data):
        c = tag + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c))
    png = (b"\x89PNG\r\n\x1a\n"
           + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(bytes(raw), 6))
           + chunk(b"IEND", b""))
    with open(path, "wb") as fh:
        fh.write(png)
    return w, h


def png_text(path):
    """The text chunks PNG carries alongside the pixels.

    ComfyUI writes the whole graph into every image it saves, so the
    prompt that made a picture is inside the picture. Nothing has to have
    been recorded when it ran - this works on files that are already
    there.
    """
    out = {}
    try:
        with open(path, "rb") as fh:
            if fh.read(8) != b"\x89PNG\r\n\x1a\n":
                return out
            while True:
                head = fh.read(8)
                if len(head) < 8:
                    break
                n = struct.unpack(">I", head[:4])[0]
                kind = head[4:8]
                if kind in (b"IDAT", b"IEND"):
                    break          # text comes before the pixels; stop early
                if n > 8 << 20:
                    break
                data = fh.read(n)
                fh.read(4)         # crc
                if kind == b"tEXt":
                    k, _, v = data.partition(b"\x00")
                    out[k.decode("latin1")] = v.decode("utf-8", "replace")
                elif kind == b"iTXt":
                    k, _, rest = data.partition(b"\x00")
                    if len(rest) < 2:
                        continue
                    compressed, rest = rest[0], rest[2:]
                    rest = rest.split(b"\x00", 2)          # lang, translated
                    if len(rest) < 3:
                        continue
                    v = rest[2]
                    if compressed:
                        import zlib
                        try:
                            v = zlib.decompress(v)
                        except Exception:
                            continue
                    out[k.decode("latin1")] = v.decode("utf-8", "replace")
    except Exception:
        pass
    return out


def prompt_of(path, _cache={}):
    """The positive prompt that produced this image, or ''."""
    try:
        key = (path, os.path.getmtime(path))
    except OSError:
        return ""
    if key in _cache:
        return _cache[key]
    text = ""
    raw = png_text(path).get("prompt", "")
    if raw:
        try:
            g = json.loads(raw)
            # Follow the sampler's positive input rather than guessing:
            # the negative is a text node too, and on a video graph it is
            # not even the shorter of the two.
            enc = None
            for node in g.values():
                if not isinstance(node, dict):
                    continue
                if "KSampler" in str(node.get("class_type", "")):
                    link = (node.get("inputs") or {}).get("positive")
                    if isinstance(link, list) and link:
                        enc = str(link[0])
                        break
            src = g.get(enc) if enc else None
            if isinstance(src, dict):
                ins = src.get("inputs") or {}
                text = ins.get("prompt") or ins.get("text") or ""
                # Conditioning can be wrapped; fall through if it is.
                if not isinstance(text, str):
                    text = ""
            if not text:
                cands = []
                for node in g.values():
                    if isinstance(node, dict) and \
                            "TextEncode" in str(node.get("class_type", "")):
                        t = (node.get("inputs") or {}).get("prompt") \
                            or (node.get("inputs") or {}).get("text") or ""
                        if isinstance(t, str) and t.strip():
                            cands.append(t)
                text = max(cands, key=len) if cands else ""
        except Exception:
            text = ""
    if len(_cache) > 400:
        _cache.clear()
    _cache[key] = text.strip()
    return _cache[key]


def cell(rel, prompt="", converting=False):
    """One tile in a gallery: a still, or a player for a clip.

    A clip in an <img> renders as a broken image, which is what a video
    mode does by default if nothing here separates the two. <video
    controls> needs no JavaScript, so it survives Lockdown Mode.
    """
    q = urllib.parse.quote(rel)
    if rel.lower().endswith(VID_EXT):
        note = ('<p class="said">converting to mp4 so it plays on a '
                'phone&hellip;</p>' if converting else "")
        return ('<div class="card"><video controls playsinline preload="metadata"'
                ' src="/out?n={q}"></video>'
                '<a class="dl" href="/out?n={q}&amp;dl=1">save it</a>{n}</div>'
                .format(q=q, n=note))
    extra = ""
    if prompt:
        # Shown in full and selectable, because copying by hand is the only
        # way to get at the clipboard without JavaScript - and the link
        # beside it is the thing you actually wanted the clipboard for.
        extra = ('<p class="said">{p}</p>'
                 '<a class="dl" href="/useprompt?n={q}">use this prompt '
                 '&rarr;</a>'.format(p=html.escape(prompt), q=q))
    # Tapping goes to a page, not to the raw file. The file itself is a
    # dead end on a phone - a bare image with nothing to tap to get back -
    # but a picture you cannot tap at all is its own kind of useless.
    return ('<div class="card">'
            '<a href="/view?n={q}"><img loading="lazy" src="/out?n={q}" '
            'alt=""></a>'
            '<a class="dl" href="/reuse?n={q}">edit this one &rarr;</a>'
            '<a class="dl" href="/out?n={q}&amp;dl=1">save it</a>{e}'
            '</div>'.format(q=q, e=extra))


def api(url, path, payload=None, timeout=30):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        url + path, data=data,
        headers={"Content-Type": "application/json"} if data else {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


# --------------------------------------------------------------- progress
#
# ComfyUI reports what it is doing over a websocket: step counts as JSON,
# and a decoded preview of the latent as a binary frame. Normally the
# browser holds that socket, but this page runs no JavaScript - that is the
# whole point of it, because Lockdown Mode makes JavaScript-heavy pages
# unusable. So this server holds the socket instead and keeps the latest
# message and frame in memory. The status page is plain HTML with a meta
# refresh, and each reload reads whatever is here.

CLIENT_ID = uuid.uuid4().hex


class Live:
    def __init__(self):
        self.lock = threading.Lock()
        self.connected = False
        self.queue = 0
        self.seq = 0
        self.preview = b""
        self.ptype = "image/jpeg"
        self._clear("", "")

    def _clear(self, prompt_id, label):
        self.prompt_id = prompt_id
        self.label = label
        self.clip = 0
        self.clips = 0
        self.unit = "clip"
        self.reel = ""
        self.node = ""
        self.value = 0
        self.max = 0
        self.done = not prompt_id
        self.error = ""
        self.cancelled = False
        self.outputs = []
        self.started = time.time()

    def request_cancel(self):
        """Stop was pressed. The workers read this between jobs."""
        with self.lock:
            self.cancelled = True

    def stopped(self):
        """A stop finished. Not an error - nothing went wrong, it ended.

        `cancelled` deliberately stays set: a worker may still be between
        jobs and has to see it. The next run clears it, in _clear().
        """
        with self.lock:
            self.done = True
            self.label = "stopped"
            self.clip = self.clips = 0
            self.value = self.max = 0

    def forget_error(self):
        """Drop the last failure, because a new attempt is being made.

        Only queued() and execution_start cleared it, and both happen after
        ComfyUI has accepted the job. Everything that fails earlier - a
        rejected prompt, a model that has since been moved away, a graph
        that will not build - returned without clearing, so the page went
        on showing the previous failure. Read three times over, one OOM
        looks like three OOMs, and you go hunting a fault that is no
        longer there.
        """
        with self.lock:
            self.error = ""
            self.cancelled = False

    def queued(self, prompt_id, label, clip=0, clips=0, unit="clip"):
        """Called when this form posts a job, before ComfyUI says anything."""
        with self.lock:
            self._clear(prompt_id, label)
            self.clip, self.clips, self.unit = clip, clips, unit
            self.preview = b""
            self.seq += 1

    def snapshot(self):
        with self.lock:
            return dict(connected=self.connected, prompt_id=self.prompt_id,
                        label=self.label, node=self.node, value=self.value,
                        max=self.max, queue=self.queue, done=self.done,
                        error=self.error, outputs=list(self.outputs),
                        cancelled=self.cancelled,
                        started=self.started, seq=self.seq,
                        clip=self.clip, clips=self.clips, reel=self.reel,
                        unit=self.unit,
                        preview=bool(self.preview))

    def frame(self):
        with self.lock:
            return self.preview, self.ptype

    # ------------------------------------------------- incoming messages
    def on_text(self, raw):
        try:
            m = json.loads(raw)
        except Exception:
            return
        t = m.get("type")
        d = m.get("data") or {}
        with self.lock:
            if t == "status":
                try:
                    self.queue = d["status"]["exec_info"]["queue_remaining"]
                except Exception:
                    pass
            elif t == "execution_start":
                if d.get("prompt_id"):
                    self.prompt_id = d["prompt_id"]
                self.done = False
                self.error = ""
                self.started = time.time()
            elif t == "executing":
                if d.get("node") is None:
                    self.done = True
                    self.preview = b""
                else:
                    self.node = str(d.get("node"))
                    self.done = False
            elif t == "progress":
                self.value = int(d.get("value") or 0)
                self.max = int(d.get("max") or 0)
                self.done = False
            elif t == "progress_state":
                # Newer releases send one entry per node instead; take the
                # one that is actually running.
                run = [n for n in (d.get("nodes") or {}).values()
                       if n.get("state") in (None, "running")]
                if run:
                    self.value = int(run[0].get("value") or 0)
                    self.max = int(run[0].get("max") or 0)
                    self.done = False
            elif t == "executed":
                for key in ("images", "gifs", "videos"):
                    for it in ((d.get("output") or {}).get(key) or []):
                        if not isinstance(it, dict) or it.get("type") != "output":
                            continue
                        rel = it.get("filename") or ""
                        if it.get("subfolder"):
                            rel = it["subfolder"] + "/" + rel
                        if rel and rel not in self.outputs:
                            self.outputs.append(rel)
            elif t == "execution_interrupted":
                self.done = True
                self.label = "stopped"
                self.value = self.max = 0
                self.preview = b""
            elif t == "execution_error":
                self.error = str(d.get("exception_message")
                                 or d.get("exception_type") or "failed")
                self.done = True
                self.preview = b""
            elif t == "execution_success":
                self.done = True
                self.preview = b""

    def on_binary(self, blob):
        # The binary layout has changed across releases - a 4-byte event
        # type, then either a 4-byte image type or a JSON header. Looking
        # for the image signature instead works for all of them.
        for magic, mime in ((b"\xff\xd8\xff", "image/jpeg"),
                            (b"\x89PNG\r\n\x1a\n", "image/png"),
                            (b"RIFF", "image/webp")):
            i = blob.find(magic, 0, 96)
            if i >= 0:
                with self.lock:
                    self.preview = blob[i:]
                    self.ptype = mime
                    self.seq += 1
                return


LIVE = Live()


def ws_frames(sock, buf):
    """Yield (opcode, payload) from a server-to-client websocket stream."""
    def need(n):
        nonlocal buf
        while len(buf) < n:
            chunk = sock.recv(1 << 16)
            if not chunk:
                raise ConnectionError("closed")
            buf += chunk
        out, buf = buf[:n], buf[n:]
        return out

    op, acc = 0, b""
    while True:
        h = need(2)
        fin, code, ln = h[0] & 0x80, h[0] & 0x0F, h[1] & 0x7F
        if h[1] & 0x80:
            raise ConnectionError("server masked a frame")
        if ln == 126:
            ln = struct.unpack(">H", need(2))[0]
        elif ln == 127:
            ln = struct.unpack(">Q", need(8))[0]
        data = need(ln) if ln else b""
        if code == 0x8:
            raise ConnectionError("server closed")
        if code == 0x9:                     # ping; a client must mask its pong
            k = os.urandom(4)
            sock.sendall(b"\x8a" + bytes([0x80 | len(data)]) + k
                         + bytes(b ^ k[i % 4] for i, b in enumerate(data)))
            continue
        if code == 0xA:
            continue
        if code in (0x1, 0x2):
            op, acc = code, data
        elif code == 0x0:
            acc += data
        if fin and op:
            yield op, acc
            op, acc = 0, b""


def ws_listen(base, live):
    """Hold ComfyUI's websocket open forever, reconnecting when it drops."""
    u = urllib.parse.urlsplit(base)
    host = u.hostname or "127.0.0.1"
    port = u.port or 80
    delay = 1
    while True:
        sock = None
        try:
            sock = socket.create_connection((host, port), timeout=10)
            key = base64.b64encode(os.urandom(16)).decode()
            sock.sendall((
                "GET /ws?clientId=%s HTTP/1.1\r\nHost: %s:%d\r\n"
                "Upgrade: websocket\r\nConnection: Upgrade\r\n"
                "Sec-WebSocket-Key: %s\r\nSec-WebSocket-Version: 13\r\n\r\n"
                % (CLIENT_ID, host, port, key)).encode())
            head = b""
            while b"\r\n\r\n" not in head:
                c = sock.recv(4096)
                if not c:
                    raise ConnectionError("no handshake")
                head += c
            head, _, rest = head.partition(b"\r\n\r\n")
            if b" 101 " not in head.split(b"\r\n")[0] + b" ":
                raise ConnectionError(head.split(b"\r\n")[0].decode("latin1"))
            sock.settimeout(None)
            with live.lock:
                live.connected = True
            print("websocket: attached as", CLIENT_ID, file=sys.stderr)
            delay = 1
            for op, payload in ws_frames(sock, rest):
                if op == 0x1:
                    live.on_text(payload.decode("utf-8", "replace"))
                else:
                    live.on_binary(payload)
        except Exception as e:
            print("websocket: %s (retry in %ds)" % (e, delay), file=sys.stderr)
        finally:
            with live.lock:
                live.connected = False
            if sock is not None:
                try:
                    sock.close()
                except Exception:
                    pass
        time.sleep(delay)
        delay = min(delay * 2, 20)


class Cancelled(Exception):
    """Someone pressed Stop. Not a failure, and not reported as one."""


def wait_done(prompt_id, timeout=7200):
    """Block until ComfyUI finishes that job, and hand back its outputs."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        s = LIVE.snapshot()
        if s["cancelled"]:
            raise Cancelled()
        if s["error"]:
            raise RuntimeError(s["error"])
        if s["prompt_id"] == prompt_id and s["done"]:
            return s["outputs"]
        time.sleep(1.5)
    raise RuntimeError("gave up waiting for ComfyUI after %d minutes"
                       % (timeout // 60))


PYAV_CONCAT = """
import av, sys
out, ins = sys.argv[1], sys.argv[2:]
o = av.open(out, "w")
ostream, offset = None, 0
for path in ins:
    c = av.open(path)
    vs = c.streams.video[0]
    if ostream is None:
        # This call was renamed between PyAV versions, and ComfyUI's venv
        # is whatever it happens to be.
        ostream = (o.add_stream_from_template(vs)
                   if hasattr(o, "add_stream_from_template")
                   else o.add_stream(template=vs))
    end = 0
    for pkt in c.demux(vs):
        if pkt.dts is None:
            continue
        pkt.stream = ostream
        pkt.pts += offset
        pkt.dts += offset
        end = max(end, pkt.pts + (pkt.duration or 0))
        o.mux(pkt)
    offset = end
    c.close()
o.close()
"""


PYAV_TO_MP4 = """
import av, sys
src, dst = sys.argv[1], sys.argv[2]
i = av.open(src)
vs = i.streams.video[0]
o = av.open(dst, "w", format="mp4")
st = None
for name in ("libx264", "h264"):
    try:
        st = o.add_stream(name, rate=vs.average_rate or 16)
        break
    except Exception:
        continue
if st is None:
    raise SystemExit("no h264 encoder in this pyav")
st.width = vs.codec_context.width
st.height = vs.codec_context.height
st.pix_fmt = "yuv420p"
st.options = {"crf": "20"}
for frame in i.decode(vs):
    frame.pts = None
    for pkt in st.encode(frame):
        o.mux(pkt)
for pkt in st.encode():
    o.mux(pkt)
o.close()
i.close()
"""

MP4_JOBS = set()
MP4_LOCK = threading.Lock()


def to_mp4(src, dst, comfy_dir):
    """Re-encode a clip to H.264 in MP4.

    ComfyUI writes vp9 in webm, which an iPhone will not play - it
    downloads instead, and then sits in Files doing nothing. H.264 is what
    every Apple device decodes in hardware. ffmpeg if the box has it,
    otherwise ComfyUI's own PyAV, which must be there because that is
    what wrote the webm.
    """
    tmp = dst + ".part"
    ff = shutil.which("ffmpeg")
    if ff:
        r = subprocess.run([ff, "-y", "-i", src, "-c:v", "libx264",
                            "-pix_fmt", "yuv420p", "-crf", "20",
                            "-movflags", "+faststart", "-f", "mp4", tmp],
                           capture_output=True, timeout=3600)
        if r.returncode:
            raise RuntimeError(r.stderr.decode("utf-8", "replace")[-300:])
    else:
        py = os.path.join(comfy_dir, "venv", "bin", "python")
        if not os.path.exists(py):
            raise RuntimeError("no ffmpeg and no ComfyUI venv to borrow PyAV from")
        r = subprocess.run([py, "-c", PYAV_TO_MP4, src, tmp],
                           capture_output=True, timeout=3600)
        if r.returncode:
            raise RuntimeError(r.stderr.decode("utf-8", "replace")[-300:])
    os.replace(tmp, dst)           # only ever appears finished
    return dst


def stitch(paths, out, comfy_dir):
    """Join the clips end to end without re-encoding them.

    ffmpeg if the box has it. Otherwise ComfyUI's own environment, which
    must have PyAV because that is what ComfyUI writes webm with - so
    there is no new dependency either way.
    """
    ff = shutil.which("ffmpeg")
    if ff:
        listing = out + ".txt"
        with open(listing, "w") as fh:
            for p in paths:
                fh.write("file '%s'\n" % p.replace("'", "'\\''"))
        try:
            subprocess.run([ff, "-y", "-f", "concat", "-safe", "0",
                            "-i", listing, "-c", "copy", out],
                           check=True, capture_output=True, timeout=900)
            return out
        finally:
            try:
                os.remove(listing)
            except OSError:
                pass
    py = os.path.join(comfy_dir, "venv", "bin", "python")
    if os.path.exists(py):
        r = subprocess.run([py, "-c", PYAV_CONCAT, out] + list(paths),
                           capture_output=True, timeout=900)
        if r.returncode != 0:
            # Without this the failure arrives as a bare exit code and the
            # reason stays inside the subprocess.
            raise RuntimeError(r.stderr.decode("utf-8", "replace").strip()[-400:]
                               or "pyav exited %d" % r.returncode)
        return out
    raise RuntimeError("no ffmpeg and no ComfyUI venv to borrow PyAV from")


def seed_for(base, i=0):
    """The seed for run i: the one that was typed, or the clock.

    A fixed seed with the same prompt six times over would give six
    identical pictures, so the offset stays either way - what a typed seed
    buys is that the same number gives the same six again tomorrow.
    """
    root = int(time.time() * 1000) if base is None else int(base)
    return (root + i * 7919) % 2**31


def batch_worker(cfg):
    """Several stills from one submission, queued one at a time.

    Either a storyboard - one line of prompt per frame - or the same
    instruction several times over, which with a different seed each time
    is how you get variations to choose between rather than one roll of
    the dice.
    """
    made = []
    prompts = cfg["prompts"]
    n = len(prompts)
    try:
        for i, prompt in enumerate(prompts):
            if LIVE.snapshot()["cancelled"]:
                raise Cancelled()
            g = cfg["build"](prompt, seed_for(cfg.get("seed"), i))
            r = api(cfg["comfy"], "/prompt",
                    {"prompt": g, "client_id": CLIENT_ID})
            pid = str(r.get("prompt_id", ""))
            LIVE.queued(pid, "image %d of %d - %s" % (i + 1, n, prompt[:60]),
                        i + 1, n, "image")
            outs = wait_done(pid, cfg["timeout"])
            made += [o for o in outs
                     if o.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))]
        with LIVE.lock:
            LIVE.outputs = made
            LIVE.label = "%d images" % n
            LIVE.done = True
            LIVE.clip = n
    except Cancelled:
        with LIVE.lock:
            LIVE.outputs = made          # whatever finished before the stop
        LIVE.stopped()
    except Exception as e:
        with LIVE.lock:
            LIVE.outputs = made          # keep whatever did come out
            LIVE.error = "%s (after %d of %d)" % (str(e)[:600], len(made), n)
            LIVE.done = True


def sequence_worker(cfg):
    """One clip after another, each starting on the last frame of the one
    before, then joined into a single file.

    Wan was trained on five seconds. Asking it for thirty in one go does
    not give you thirty good seconds, it gives you five good ones and
    twenty-five of drift. Chaining keeps every clip inside what the model
    knows."""
    outdir, indir = cfg["outdir"], cfg["indir"]
    prompts = cfg["prompts"]
    n = len(prompts)
    clips = []
    try:
        current = cfg["image"]
        for i, prompt in enumerate(prompts):
            g = cfg["graph"].build_video(
                high=cfg["high"], low=cfg["low"], clip=cfg["clip"],
                vae=cfg["vae"], image=current, prompt=prompt,
                negative=cfg.get("negative") or "static, still, blurry",
                steps=cfg["steps"], cfg=cfg["cfg"],
                seed=seed_for(cfg.get("seed"), i),
                width=cfg["width"], height=cfg["height"],
                length=cfg["length"], loras=cfg["loras"],
                end_image=None, save_last=(i < n - 1))
            r = api(cfg["comfy"], "/prompt",
                    {"prompt": g, "client_id": CLIENT_ID})
            pid = str(r.get("prompt_id", ""))
            LIVE.queued(pid, "clip %d of %d - %s" % (i + 1, n, prompt[:60]),
                        i + 1, n, "clip")
            outs = wait_done(pid, cfg["timeout"])
            vids = [o for o in outs if o.lower().endswith((".webm", ".mp4"))]
            stills = [o for o in outs
                      if os.path.basename(o).startswith("lastframe")]
            if not vids:
                raise RuntimeError("clip %d produced no video file" % (i + 1))
            clips.append(os.path.join(outdir, vids[0]))
            if i < n - 1:
                if not stills:
                    raise RuntimeError(
                        "clip %d gave no last frame, so the next one has "
                        "nothing to continue from. This ComfyUI has no "
                        "ImageFromBatch node." % (i + 1))
                nm = time.strftime("%H%M%S_") + "chain%02d.png" % (i + 1)
                shutil.copy2(os.path.join(outdir, stills[0]),
                             os.path.join(indir, nm))
                current = nm

        with LIVE.lock:
            LIVE.label = "joining %d clips" % len(clips)
        reel = os.path.join(outdir, time.strftime("reel_%H%M%S") + ".webm")
        try:
            stitch(clips, reel, cfg["comfy_dir"])
            with LIVE.lock:
                LIVE.reel = os.path.basename(reel)
                LIVE.label = "%d clips, about %d seconds" % (
                    n, n * cfg["length"] // 16)
        except Exception as e:
            with LIVE.lock:
                LIVE.reel = ""
                LIVE.label = ("the clips are all there, but joining them "
                              "failed: %s" % str(e)[:200])
        with LIVE.lock:
            LIVE.done = True
            LIVE.clip = n
    except Cancelled:
        LIVE.stopped()
    except Exception as e:
        with LIVE.lock:
            LIVE.error = "%s (after %d of %d clips)" % (str(e)[:600],
                                                        len(clips), n)
            LIVE.done = True


class Graph:
    """Builds the Qwen-edit graph against whatever this ComfyUI provides."""

    def __init__(self, info):
        self.info = info
        self.notes = []
        self.encoder = self.pick(
            ["TextEncodeQwenImageEditPlus", "TextEncodeQwenImageEdit"])
        # Video is optional; its absence is reported on the page rather than
        # treated as a broken install.
        self.i2v = next((n for n in ("WanImageToVideo",) if n in info), None)
        # Two ways to pin the last frame, depending on the release: a
        # dedicated node, or an optional end_image on the ordinary one.
        self.flf = next((n for n in ("WanFirstLastFrameToVideo",) if n in info), None)
        self.i2v_end = bool(self.i2v and "end_image" in self.inputs_of(self.i2v))
        self.video_saver = next(
            (n for n in ("SaveWEBM", "SaveAnimatedWEBP", "SaveVideo") if n in info),
            None)
        for n in ("UNETLoader", "CLIPLoader", "VAELoader", "VAEEncode",
                  "VAEDecode", "KSampler", "LoadImage", "SaveImage",
                  "CheckpointLoaderSimple", "LoraLoader"):
            if n not in info:
                self.notes.append(f"missing node: {n}")

    def pick(self, names):
        for n in names:
            if n in self.info:
                return n
        self.notes.append("no Qwen edit encoder found (tried " +
                          ", ".join(names) + ")")
        return None

    def fill_required(self, node, given):
        """Add any required input this ComfyUI wants that we did not set.

        Node signatures drift between releases - ImageScaleToTotalPixels
        grew a resolution_steps field - and one missing required input has
        the whole graph rejected. The defaults are sitting in
        /object_info, so take them from there rather than guess, which is
        the same reason the rest of this class reads the node list instead
        of hardcoding it.
        """
        spec = (self.info.get(node) or {}).get("input", {}).get("required", {})
        out = dict(given)
        for name, meta in spec.items():
            if name in out:
                continue
            kind = meta[0] if isinstance(meta, (list, tuple)) and meta else None
            opts = (meta[1] if isinstance(meta, (list, tuple)) and len(meta) > 1
                    and isinstance(meta[1], dict) else {})
            if "default" in opts:
                out[name] = opts["default"]
            elif isinstance(kind, list) and kind:
                out[name] = kind[0]          # an enum: take its first value
            elif kind == "INT":
                out[name] = 0
            elif kind == "FLOAT":
                out[name] = 0.0
            elif kind == "STRING":
                out[name] = ""
            # Anything else wants a link from another node, and an invented
            # one would fail further in, where it is harder to read.
        return out

    def inputs_of(self, node):
        spec = self.info.get(node, {}).get("input", {})
        return list(spec.get("required", {})) + list(spec.get("optional", {}))

    # The full /object_info is read once at startup, which was fine until
    # a model got installed while the form was already running: the
    # dropdown kept the list it had and the new file was invisible, with
    # nothing on the page to say why. ComfyUI will describe one node at a
    # time, which is small enough to re-read on a render, so the loaders
    # get refreshed and everything else stays as it was.
    LOADERS = ("UNETLoader", "CLIPLoader", "VAELoader", "LoraLoader",
               "CheckpointLoaderSimple")

    def refresh_loaders(self, comfy, max_age=20):
        now = time.time()
        if now - getattr(self, "_refreshed", 0) < max_age:
            return
        self._refreshed = now
        for node in self.LOADERS:
            try:
                got = api(comfy, "/object_info/" + node, timeout=10)
            except Exception:
                continue          # keep what we had; a stale list beats none
            spec = got.get(node)
            if isinstance(spec, dict) and spec.get("input"):
                self.info[node] = spec

    def enum_for(self, node, field):
        spec = self.info.get(node, {}).get("input", {})
        for grp in ("required", "optional"):
            if field in spec.get(grp, {}):
                v = spec[grp][field][0]
                return v if isinstance(v, list) else []
        return []

    def image_field(self):
        """Whether the encoder wants `image` or `image1`."""
        return (self.image_fields() or [None])[0]

    def image_fields(self):
        """Every reference image slot the encoder has, in order.

        The Plus encoder takes image1, image2, image3 - that is the whole
        point of it, and it is what lets one person from one photo and
        another from a second end up in the same picture. The older
        encoder has a single `image`, so the count is read off the node
        rather than assumed.
        """
        fields = self.inputs_of(self.encoder) if self.encoder else []
        numbered = sorted(f for f in fields
                          if re.fullmatch(r"image[1-9]", f or ""))
        if numbered:
            return numbered
        return ["image"] if "image" in fields else []

    def build(self, *, mode, unet, clip, vae, ckpt, images, prompt,
              steps, cfg, seed, sampler, scheduler, loras=(), denoise=1.0,
              width=0, height=0, mask=None):
        if not self.encoder:
            raise RuntimeError("no Qwen edit encoder node available")
        g = {}
        if mode == "checkpoint":
            # A merged AIO carries model, clip and vae in one file.
            g["ck"] = {"class_type": "CheckpointLoaderSimple",
                       "inputs": {"ckpt_name": ckpt}}
            M, C, V = ["ck", 0], ["ck", 1], ["ck", 2]
        else:
            types = self.enum_for("CLIPLoader", "type")
            qwen = next((t for t in types if "qwen" in t.lower()), "qwen_image")
            g["un"] = {"class_type": "UNETLoader",
                       "inputs": {"unet_name": unet, "weight_dtype": "default"}}
            g["cl"] = {"class_type": "CLIPLoader",
                       "inputs": {"clip_name": clip, "type": qwen}}
            g["va"] = {"class_type": "VAELoader", "inputs": {"vae_name": vae}}
            M, C, V = ["un", 0], ["cl", 0], ["va", 0]

        # LoRAs sit between the loaders and everything downstream: the
        # sampler takes the model, the encoder the clip. Each one is fed
        # the output of the one before it, so a speed LoRA and a content
        # LoRA stack rather than replacing each other.
        for i, (name, strength) in enumerate(loras or ()):
            if not name:
                continue
            k = "lo%d" % i
            g[k] = {"class_type": "LoraLoader", "inputs": {
                "model": M, "clip": C, "lora_name": name,
                "strength_model": strength, "strength_clip": strength}}
            M, C = [k, 0], [k, 1]

        images = [i for i in (images or []) if i]
        if not images:
            raise RuntimeError("no input image")
        slots = self.image_fields()
        for n, name in enumerate(images[:len(slots)] or images[:1]):
            g["im%d" % n] = {"class_type": "LoadImage",
                             "inputs": {"image": name}}

        def enc(text):
            ins = {"clip": C, "prompt": text}
            if "vae" in self.inputs_of(self.encoder):
                ins["vae"] = V
            for n, field in enumerate(slots):
                if n < len(images):
                    ins[field] = ["im%d" % n, 0]
            return {"class_type": self.encoder, "inputs": ins}

        g["po"] = enc(prompt)
        g["ne"] = enc("")
        # The first image is what the latent is encoded from, so it is
        # also what sets the size of the output - which means an untouched
        # phone photo has been quietly asking for twelve megapixels, and
        # getting them: minutes of sampling and a twenty-megabyte png
        # nobody wanted. The Size field is what was actually chosen, so
        # scale to it. By total pixels rather than to exact dimensions,
        # because squashing someone's aspect ratio is its own bug.
        src = ["im0", 0]
        mp = (width * height) / 1e6 if width and height else 0
        if mp and "ImageScaleToTotalPixels" in self.info:
            g["sc"] = {"class_type": "ImageScaleToTotalPixels",
                       "inputs": self.fill_required(
                           "ImageScaleToTotalPixels",
                           {"image": src, "upscale_method": "lanczos",
                            "megapixels": round(mp, 2)})}
            src = ["sc", 0]
        elif width and height and "ImageScale" in self.info:
            g["sc"] = {"class_type": "ImageScale",
                       "inputs": self.fill_required(
                           "ImageScale",
                           {"image": src, "upscale_method": "lanczos",
                            "width": width, "height": height,
                            "crop": "disabled"})}
            src = ["sc", 0]
        # Note the encoder still gets the untouched images: the reference
        # conditioning is where facial detail comes from, and there is no
        # reason to hand it less than it was given.
        g["la"] = {"class_type": "VAEEncode",
                   "inputs": {"pixels": src, "vae": V}}
        # Denoise is what decides whether the photo is a starting point or
        # merely a suggestion. At 1.0 every pixel is regenerated and the
        # face is redrawn from the encoder's idea of it; below that the
        # sampler starts from the actual image and the face survives as
        # pixels. Too low and the edit simply does not happen.
        # A mask makes denoise local. Without one it is a single dial over
        # the whole picture, which is why changing a shirt has always cost
        # you the face: the only setting strong enough to redraw clothing
        # redraws everything else with it. Inside the marked cells the
        # sampler works at full strength; outside, the original latent is
        # kept and nothing moves at all.
        latent = ["la", 0]
        if mask and "SetLatentNoiseMask" in self.info \
           and "ImageToMask" in self.info:
            g["mi"] = {"class_type": "LoadImage", "inputs": {"image": mask}}
            g["mk"] = {"class_type": "ImageToMask",
                       "inputs": self.fill_required(
                           "ImageToMask", {"image": ["mi", 0], "channel": "red"})}
            g["nm"] = {"class_type": "SetLatentNoiseMask",
                       "inputs": {"samples": ["la", 0], "mask": ["mk", 0]}}
            latent = ["nm", 0]

        g["ks"] = {"class_type": "KSampler", "inputs": {
            "model": M, "positive": ["po", 0], "negative": ["ne", 0],
            "latent_image": latent, "seed": seed, "steps": steps,
            "cfg": cfg, "sampler_name": sampler, "scheduler": scheduler,
            "denoise": denoise}}
        g["de"] = {"class_type": "VAEDecode",
                   "inputs": {"samples": ["ks", 0], "vae": V}}
        g["sv"] = {"class_type": "SaveImage",
                   "inputs": {"images": ["de", 0], "filename_prefix": "edit"}}
        return g


    def masks_ok(self):
        """Whether this ComfyUI has the two nodes a mask needs."""
        return ("SetLatentNoiseMask" in self.info
                and "ImageToMask" in self.info)

    def can_end_frame(self):
        return bool(self.flf or self.i2v_end)

    def build_video(self, *, high, low, clip, vae, image, prompt, negative,
                    steps, cfg, seed, width, height, length, loras=(),
                    end_image=None, save_last=False):
        """Wan 2.2 I2V: two experts over one latent, high noise then low."""
        if not self.i2v:
            raise RuntimeError("WanImageToVideo node is not available")
        if not (high and low and clip and vae):
            raise RuntimeError("need both Wan experts, the umt5 encoder and a vae")

        types = self.enum_for("CLIPLoader", "type")
        wan = next((t for t in types if "wan" in t.lower()), "wan")
        g = {
            "uh": {"class_type": "UNETLoader",
                   "inputs": {"unet_name": high, "weight_dtype": "default"}},
            "ul": {"class_type": "UNETLoader",
                   "inputs": {"unet_name": low, "weight_dtype": "default"}},
            "cl": {"class_type": "CLIPLoader",
                   "inputs": {"clip_name": clip, "type": wan}},
            "va": {"class_type": "VAELoader", "inputs": {"vae_name": vae}},
            "im": {"class_type": "LoadImage", "inputs": {"image": image}},
        }
        # Two parallel chains, one per expert, built in step. Both experts
        # have to carry every LoRA or the halves disagree halfway through
        # the same clip. Where a LoRA ships as a high/low pair each expert
        # gets its own file; otherwise both get the same one. The text
        # conditioning is shared, so the clip is taken from the high chain
        # and the low chain's clip output goes unused.
        MH, ML, C = ["uh", 0], ["ul", 0], ["cl", 0]
        for i, (hi, lo, strength) in enumerate(loras or ()):
            if not hi:
                continue
            a, b = "lh%d" % i, "ll%d" % i
            g[a] = {"class_type": "LoraLoader", "inputs": {
                "model": MH, "clip": C, "lora_name": hi,
                "strength_model": strength, "strength_clip": strength}}
            g[b] = {"class_type": "LoraLoader", "inputs": {
                "model": ML, "clip": C, "lora_name": lo or hi,
                "strength_model": strength, "strength_clip": strength}}
            MH, ML, C = [a, 0], [b, 0], [a, 1]

        g["po"] = {"class_type": "CLIPTextEncode",
                   "inputs": {"clip": C, "text": prompt}}
        g["ne"] = {"class_type": "CLIPTextEncode",
                   "inputs": {"clip": C, "text": negative}}
        ins = {"positive": ["po", 0], "negative": ["ne", 0], "vae": ["va", 0],
               "width": width, "height": height, "length": length,
               "batch_size": 1, "start_image": ["im", 0]}
        node = self.i2v
        if end_image:
            if not self.can_end_frame():
                raise RuntimeError(
                    "this ComfyUI has no way to pin a last frame - neither "
                    "WanFirstLastFrameToVideo nor an end_image input")
            g["im2"] = {"class_type": "LoadImage", "inputs": {"image": end_image}}
            ins["end_image"] = ["im2", 0]
            # The dedicated node is the better path where it exists; the
            # optional input on the plain node is the fallback.
            node = self.flf or self.i2v
        g["wv"] = {"class_type": node, "inputs": ins}

        half = max(1, steps // 2)
        g["k1"] = {"class_type": "KSamplerAdvanced", "inputs": {
            "model": MH, "add_noise": "enable", "noise_seed": seed,
            "steps": steps, "cfg": cfg, "sampler_name": "euler",
            "scheduler": "simple", "positive": ["wv", 0], "negative": ["wv", 1],
            "latent_image": ["wv", 2], "start_at_step": 0,
            "end_at_step": half, "return_with_leftover_noise": "enable"}}
        g["k2"] = {"class_type": "KSamplerAdvanced", "inputs": {
            "model": ML, "add_noise": "disable", "noise_seed": seed,
            "steps": steps, "cfg": cfg, "sampler_name": "euler",
            "scheduler": "simple", "positive": ["wv", 0], "negative": ["wv", 1],
            "latent_image": ["k1", 0], "start_at_step": half,
            "end_at_step": 10000, "return_with_leftover_noise": "disable"}}
        g["de"] = {"class_type": "VAEDecode",
                   "inputs": {"samples": ["k2", 0], "vae": ["va", 0]}}

        # The last frame, saved as a still. A clip that starts where the
        # previous one ended is how you get past five seconds, and pulling
        # that frame out of a finished video file afterwards means decoding
        # vp9 somewhere - here it costs one node.
        if save_last and "ImageFromBatch" in self.info:
            g["lf"] = {"class_type": "ImageFromBatch",
                       "inputs": self.fill_required(
                           "ImageFromBatch",
                           {"image": ["de", 0], "batch_index": length - 1,
                            "length": 1})}
            g["ls"] = {"class_type": "SaveImage", "inputs": {
                "images": ["lf", 0], "filename_prefix": "lastframe"}}

        # Which video writer exists moves between releases; frames are the
        # fallback, and beat failing outright.
        if self.video_saver == "SaveWEBM":
            g["sv"] = {"class_type": "SaveWEBM", "inputs": {
                "images": ["de", 0], "filename_prefix": "video",
                "codec": "vp9", "fps": 16.0, "crf": 32.0}}
        elif self.video_saver == "SaveAnimatedWEBP":
            g["sv"] = {"class_type": "SaveAnimatedWEBP", "inputs": {
                "images": ["de", 0], "filename_prefix": "video",
                "fps": 16.0, "lossless": False, "quality": 85, "method": "default"}}
        else:
            g["sv"] = {"class_type": "SaveImage", "inputs": {
                "images": ["de", 0], "filename_prefix": "video_frame"}}
        return g


VIEW = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{name}</title><style>
:root{{color-scheme:dark light}}
body{{font:17px/1.5 -apple-system,system-ui,sans-serif;margin:0;padding:12px;
  max-width:900px}}
img,video{{width:100%;border-radius:10px;display:block;background:#8882}}
a.btn{{display:block;text-align:center;font-size:17px;padding:13px;margin-top:10px;
  border-radius:10px;background:#d2691e;color:#fff;text-decoration:none}}
a.btn.plain{{background:#8883;color:inherit}}
p.said{{color:#888;font-size:14px;margin:12px 2px;-webkit-user-select:text;
  user-select:text}}
</style></head><body>
{media}
{said}
<a class="btn" href="/">back to the form</a>
{reuse}
<a class="btn plain" href="/out?n={q}&amp;dl=1">save it</a>
</body></html>"""

STATUS = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
{refresh}<title>{head}</title><style>
:root{{color-scheme:dark light}}
body{{font:17px/1.5 -apple-system,system-ui,sans-serif;margin:0;padding:20px 16px;max-width:640px}}
h1{{font-size:22px;margin:0 0 2px}} p.sub{{color:#888;margin:0 0 16px;font-size:15px}}
.bar{{height:10px;border-radius:5px;background:#8883;overflow:hidden;margin:16px 0 6px}}
.bar>i{{display:block;height:100%;background:#d2691e}}
.shot{{width:100%;border-radius:10px;display:block;background:#8882}}
a.btn{{display:block;text-align:center;font-size:17px;padding:13px;margin-top:18px;
  border-radius:10px;background:#d2691e;color:#fff;text-decoration:none}}
a.btn.plain{{background:#8883;color:inherit}}
.grid{{display:grid;gap:10px;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));margin-top:14px}}
.card img,.card video{{width:100%;border-radius:8px;display:block;background:#8882}}
.card a.dl{{font-size:13px;color:#888;display:block;padding:4px 2px}}
.card p.said{{font-size:12px;color:#888;margin:4px 2px 0;line-height:1.35;
  -webkit-user-select:text;user-select:text}}
.live p.said{{font-size:15px;color:inherit;margin:0 0 8px}}
.note{{color:#888;font-size:14px}} .err{{color:#c55}} .ok{{color:#4a4}}
</style></head><body>
<h1>{head}</h1>
<p class="sub">{sub}</p>
{body}
<a class="btn" href="/status">refresh now</a>
<a class="btn plain" href="/">back to the form</a>
</body></html>"""

PAGE = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
{refresh}<title>Edit</title><style>
:root{{color-scheme:dark light}}
body{{font:17px/1.5 -apple-system,system-ui,sans-serif;margin:0;padding:20px 16px;max-width:640px}}
h1{{font-size:22px;margin:0 0 2px}} p.sub{{color:#888;margin:0 0 20px;font-size:15px}}
form{{border:1px solid #8884;border-radius:12px;padding:18px;margin-bottom:24px}}
label{{display:block;font-size:14px;color:#888;margin:14px 0 4px}}
select,input[type=text],input[type=number],textarea,input[type=file]{{
  width:100%;box-sizing:border-box;font-size:17px;padding:10px;
  border-radius:8px;border:1px solid #8886;background:#8881;color:inherit}}
textarea{{min-height:76px}}
button{{font-size:17px;padding:13px;width:100%;margin-top:18px;border:0;
  border-radius:10px;background:#d2691e;color:#fff}}
button.second{{margin-top:10px;background:#8883;color:inherit}}
form.stop{{margin:10px 0 0}}
form.stop button{{margin:0;background:#8883;color:inherit}}
.maskwrap{{position:relative;margin-top:8px;border-radius:10px;overflow:hidden;
  max-width:100%}}
/* grid-auto-rows is not optional: the rows are implicit, and without a
   size they fall back to auto - which for a label holding nothing but an
   empty span is no height at all. The overlay then exists but cannot be
   seen or tapped. */
/* The tint must not swallow the taps - they belong to the photo under it. */
.cells{{position:absolute;top:0;left:0;right:0;bottom:0;display:grid;
  grid-auto-rows:1fr;pointer-events:none}}
.cells i{{display:block}}
.cells i.on{{background:#d2691e99}}
.cells i.c{{background:#fff;outline:2px solid #d2691e}}
.maskwrap>input[type=image]{{display:block;width:100%;height:auto;
  touch-action:manipulation}}
/* Without JavaScript the canvas never appears and the tapping stays; with
   it, the brush replaces both. Only the browser knows which it got, so it
   is CSS that decides. */
#pc{{display:none}} .brush{{display:none}}
html.js #pc{{display:block;position:absolute;top:0;left:0;width:100%;
  height:100%;opacity:.45;touch-action:none}}
html.js .brush{{display:block}}
html.js .tapping{{display:none}}
html.js .cells{{display:none}}
.row{{display:flex;gap:10px}} .row>div{{flex:1}}
.row.ops{{margin-top:8px}} .row.ops>button{{flex:1;margin-top:0;font-size:15px;padding:10px}}
.grid{{display:grid;gap:10px;grid-template-columns:repeat(auto-fill,minmax(150px,1fr))}}
.card img,.card video{{width:100%;border-radius:8px;display:block;background:#8882}}
.card a.dl{{font-size:13px;color:#888;display:block;padding:4px 2px}}
.card p.said{{font-size:12px;color:#888;margin:4px 2px 0;line-height:1.35;
  -webkit-user-select:text;user-select:text}}
.live p.said{{font-size:15px;color:inherit;margin:0 0 8px}}
.note{{color:#888;font-size:14px}} .err{{color:#c55}} .ok{{color:#4a4}}
.live{{border:1px solid #8884;border-radius:12px;padding:14px 16px;margin-bottom:20px}}
.bar{{height:10px;border-radius:5px;background:#8883;overflow:hidden;margin:4px 0 6px}}
.bar>i{{display:block;height:100%;background:#d2691e}}
.shot{{width:100%;border-radius:10px;display:block;background:#8882}}
</style></head><body>
<h1>Edit a photo</h1>
<p class="sub">{status}</p>
{msg}
{live}
<form id="f" method="post" enctype="multipart/form-data" action="/">
  <label>Photo — upload one</label>
  <input type="file" name="photo" accept="image/*">
  <label>…or pick one already on the box</label>
  <select name="existing">{choices}</select>

  <label>{extra}</label>
  <input type="file" name="photo2" accept="image/*">
  <select name="existing2">{choices2}</select>

  <label>{third}</label>
  <input type="file" name="photo3" accept="image/*">
  <select name="existing3">{choices3}</select>

  {maskgrid}

  <label>Instruction</label>
  <textarea name="prompt" placeholder="change the shirt to green">{last}</textarea>

  <label>Negative &mdash; what to keep out (video)</label>
  <textarea name="negative" style="min-height:52px">{negative}</textarea>

  <label>What to do</label>
  <select name="mode">{modes}</select>

  <label>Model (image editing only)</label>
  <select name="model">{models}</select>

  <label>Add a LoRA by link &mdash; paste the download URL</label>
  <input type="text" name="lora_url" placeholder="https://...">

  <label>&hellip;or from your phone (small ones only)</label>
  <input type="file" name="lora_file">

  <label>LoRA (optional)</label>
  <select name="lora">{loras}</select>
  <label>strength</label>
  <input type="text" name="lora_strength" value="{lstr}">

  <label>Second LoRA &mdash; stacks on the first</label>
  <select name="lora2">{loras2}</select>
  <label>strength</label>
  <input type="text" name="lora_strength2" value="{lstr2}">

  <div class="row">
    <div><label>Steps</label><input type="number" name="steps" value="{steps}" min="1" max="60"></div>
    <div><label>CFG</label><input type="text" name="cfg" value="{cfg}"></div>
  </div>
  <p class="note">Some LoRAs only come to a signed-in browser, not to an
  API. Open the download in Safari, long-press and copy the link, and
  paste it above &mdash; the box fetches it itself, which avoids sending a
  few hundred megabytes up from your phone through a proxy that will
  refuse it. The file picker below is for small ones only.</p>
  <p class="note">Six images from one press: put six lines in the
  instruction box and you get six frames of a scene, or leave one line and
  you get six goes at it with different seeds, to pick from. Each keeps
  its own prompt underneath it in Results.</p>
  <p class="note">Size applies to editing as well as video: the photo is
  scaled to that many pixels before it is worked on. A phone photo left
  alone is twelve megapixels, which is twelve times the work for a picture
  you cannot tell apart on a phone screen.</p>
  <div class="row">
    <div><label>Denoise (editing) &mdash; lower keeps more of the photo</label>
    <input type="text" name="denoise" value="{denoise}"></div>
    <div><label>Seed &mdash; blank for a new one each time</label>
    <input type="text" name="seed" value="{seed}"></div>
  </div>
  <p class="note">Put a number in Seed while you are tuning Denoise or LoRA
  strength. With it blank every run also draws fresh noise, so a picture
  that came out better may have come out better by luck, and you cannot
  tell that from the dial you just moved. Any number will do; clear it
  again when you go back to wanting variety.</p>
  <div class="row">
    <div><label>Frames (video)</label><input type="number" name="length" value="{length}" min="9" max="161"></div>
    <div><label>Size</label><input type="text" name="size" value="{size}"></div>
  </div>
  <label>Images to make (editing) &mdash; one prompt line each, or one line
  repeated for variations</label>
  <input type="number" name="count" value="{count}" min="1" max="6">

  <label>Clips to chain (video) &mdash; each carries on from the last frame of the one before</label>
  <input type="number" name="clips" value="{clips}" min="1" max="8">
  <p class="note">If an edit changes a face you wanted kept, lower Denoise
  before anything else. At 1.0 the whole picture is regenerated and the
  face is redrawn from scratch; at 0.85 the sampler starts from your actual
  photo and the face survives as pixels. Below about 0.6 the edit stops
  happening at all &mdash; work down from 0.9.<br><br>
  Faces hold at 1280x720 and drift at 832x480 &mdash; at
  480p there are barely eighty pixels of face for the model to keep, so it
  invents the rest. Crop your start image close for the same reason, and
  describe only the motion: every word about how someone looks is an
  invitation to redraw them.<br><br>
  For a long video, raise Clips. At 81 frames each that is
  5 seconds a clip, so 4 gives 20 seconds and 6 gives 30. Write one line
  of instruction per clip in the box above and each gets its own; one line
  is reused for all of them. They are joined into a single file at the
  end.<br><br>
  To put two people in one picture, pick one in each
  image slot and say which is which in the instruction &mdash; &ldquo;the
  man from image 1 and the woman from image 2, sitting at a table&rdquo;.
  The first image sets the size of the output.<br><br>
  Editing: the merged AIO wants 4 steps and CFG 1; the
  separate bf16 wants 20 and CFG 3. Video: 20 steps, CFG 3.5, 81 frames
  (5s, the length Wan was trained on). Identity holds over short clips
  and drifts over long ones.<br><br>
  Two slots because a speed LoRA and a content LoRA do different jobs:
  put a Lightning one in either slot and drop to 4 steps at CFG 1, and
  keep the content LoRA in the other. Picking both halves of the same
  pair does nothing extra &mdash; the second is ignored.</p>
  <button type="submit">Run</button>
  <button type="submit" formaction="/mask" class="second">Mask</button>
</form>

<h2 style="font-size:18px">Results</h2>
<p class="note">Newest first. Tap one, then press and hold to save it.</p>
<div class="grid">{outs}</div>
</body></html>"""


def parse_multipart(body, boundary):
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
    return re.sub(r"[^A-Za-z0-9._-]", "_", os.path.basename(n or "x"))[:100] or "x"


def squash(s):
    """Lowercase, and drop everything that is not a letter or a digit."""
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def lora_pair(name, available):
    """Wan 2.2 A14B LoRAs often ship as two files, one per expert.

    The two are named alike but for 'high'/'low', so the partner can be
    found from the filename - except that the halves of one pair are
    rarely punctuated the same way. A real pair on Civitai is uploaded as
    'highnoise' and 'low noise', which no amount of substring replacement
    turns into each other. So both names are squashed to letters and
    digits before they are compared.

    Matching a substring would otherwise be reckless, since
    'highheels.safetensors' contains 'high'. What makes it safe is that
    the partner has to actually be on disk: nothing squashes to
    'lowheels', so that falls through to one file on both experts, which
    is what a single-file LoRA wants anyway.

    Returns (for the high-noise expert, for the low-noise one).
    """
    if not name:
        return None, None
    me = squash(name)
    for src, dst, mine_is_high in (("high", "low", True), ("low", "high", False)):
        if src not in me:
            continue
        want = me.replace(src, dst)
        for other in (available or ()):
            if other != name and squash(other) == want:
                return (name, other) if mine_is_high else (other, name)
    return name, name


# 'high' and 'low' as words of their own, or in front of 'noise'. Bare
# substrings would call 'highheels' and 'slowmo' halves of a pair.
HALF = re.compile(r"(?:^|[^a-z])(?:high|low)(?:noise|[^a-z]|$)", re.I)


def lora_orphan(name, available):
    """True when this looks like half of a pair whose other half is absent.

    Worth saying out loud, because it is not an error ComfyUI reports: it
    loads the file onto both experts and quietly wastes half the adapter.
    """
    if not name or not HALF.search(name):
        return False
    hi, lo = lora_pair(name, available)
    return hi == lo


class H(BaseHTTPRequestHandler):
    comfy = "http://127.0.0.1:8288"
    indir = ""
    outdir = ""
    credential = ""
    graph = None
    # Every setting, kept between runs. Editing is iterative - the second
    # attempt is the first one with one thing changed - and a form that
    # forgets makes you retype the nine things you did not want to change.
    last = {"prompt": "", "model": "", "mode": "image", "clips": "1",
            "existing": "", "existing2": "", "existing3": "", "mask": [],
            "corner": "",
            "negative": "static, still, blurry, distorted, deformed face, "
                        "changing face, extra limbs, watermark"}

    # ...but the sampler settings are per mode, and sharing them was a
    # quiet way to ruin a video. Four steps at CFG 1 is right for the
    # merged image model and is two steps per expert on Wan, which comes
    # out as mush. Switching mode now brings that mode's numbers with it.
    modes = {
        "image": {"steps": "4", "cfg": "1.0", "size": "1024x1024",
                  "length": "81", "lora": "", "lstr": "1.0",
                  "lora2": "", "lstr2": "1.0", "denoise": "1.0",
                  "count": "1", "seed": ""},
        "video": {"steps": "20", "cfg": "3.5", "size": "832x480",
                  "length": "81", "lora": "", "lstr": "1.0",
                  "lora2": "", "lstr2": "1.0", "denoise": "1.0",
                  "count": "1", "seed": ""},
    }

    def log_message(self, f, *a):
        sys.stderr.write("%s %s\n" % (self.address_string(), f % a))

    def authed(self):
        if not self.credential:
            return True
        want = "Basic " + base64.b64encode(self.credential.encode()).decode()
        if hmac.compare_digest(self.headers.get("Authorization", ""), want):
            return True
        self.send_response(401)
        self.send_header("WWW-Authenticate", 'Basic realm="edit"')
        self.send_header("Content-Length", "0")
        self.end_headers()
        return False

    # ------------------------------------------------------------ helpers
    def listing(self, root, limit=40):
        out = []
        for base, _d, files in os.walk(root):
            for f in files:
                if os.path.splitext(f)[1].lower() in IMG_EXT:
                    p = os.path.join(base, f)
                    try:
                        out.append((os.path.getmtime(p), os.path.relpath(p, root)))
                    except OSError:
                        pass
        out.sort(reverse=True)
        names = {r for _m, r in out}
        # A clip and its mp4 are the same clip; show only the one that
        # plays.
        return [r for _m, r in out
                if not (r.lower().endswith(".webm")
                        and r[:-5] + ".mp4" in names)][:limit]

    def models(self):
        self.graph.refresh_loaders(self.comfy)
        def enum(node, field):
            return self.graph.enum_for(node, field)
        unets = enum("UNETLoader", "unet_name")
        return {"wan_high": [u for u in unets if "wan" in u.lower() and "high" in u.lower()],
                "wan_low": [u for u in unets if "wan" in u.lower() and "low" in u.lower()],
                "wan_clip": [c for c in enum("CLIPLoader", "clip_name") if "umt5" in c.lower()],
                "wan_vae": [v for v in enum("VAELoader", "vae_name") if "wan" in v.lower()],
                "lora": enum("LoraLoader", "lora_name"),
                "ckpt": enum("CheckpointLoaderSimple", "ckpt_name"),
                "unet": enum("UNETLoader", "unet_name"),
                "clip": enum("CLIPLoader", "clip_name"),
                "vae": enum("VAELoader", "vae_name")}

    def video_ready(self, m):
        return bool(self.graph.i2v and m["wan_high"] and m["wan_low"]
                    and m["wan_clip"] and m["wan_vae"])

    def render(self, msg=""):
        L = self.last
        L.setdefault("clips", "1")
        mode = L["mode"]
        M = self.modes[mode if mode in self.modes else "image"]
        steps, cfg = M["steps"], M["cfg"]
        lora, lstr = M["lora"], M["lstr"]
        lora2, lstr2 = M["lora2"], M["lstr2"]
        length, size = M["length"], M["size"]
        m = self.models()
        opts = []
        def sel(v):
            return " selected" if v == L["model"] else ""
        for c in m["ckpt"]:
            opts.append('<option value="ckpt:{0}"{2}>{1} (merged, 4 steps)</option>'
                        .format(html.escape(c), html.escape(c), sel("ckpt:" + c)))
        # diffusion_models holds whatever else is installed - Klein, on this
        # box. Wrapping a Qwen graph around a FLUX model fails in a way that
        # looks like a broken pipeline, so only offer what belongs here.
        for u in [x for x in m["unet"] if "qwen" in x.lower()]:
            opts.append('<option value="unet:{0}"{2}>{1} (separate, 20 steps)</option>'
                        .format(html.escape(u), html.escape(u), sel("unet:" + u)))
        def lora_label(l):
            if lora_orphan(l, m["lora"]):
                return l + "  [half a pair - the other file is missing]"
            hi, lo = lora_pair(l, m["lora"])
            return l + ("  [paired: high + low]" if hi != lo else "")
        def lora_menu(chosen):
            return '<option value="">(none)</option>' + "".join(
                '<option value="{0}"{2}>{1}</option>'.format(
                    html.escape(l), html.escape(lora_label(l)),
                    " selected" if l == chosen else "")
                for l in m["lora"])
        loras, loras2 = lora_menu(lora), lora_menu(lora2)
        vsel = " selected" if mode == "video" else ""
        if self.video_ready(m):
            modes = ('<option value="image"{0}>edit an image</option>'
                     '<option value="video"{1}>animate an image (Wan 2.2)</option>'
                     .format("" if vsel else " selected", vsel))
        else:
            modes = '<option value="image" selected>edit an image</option>'
            if self.graph.i2v:
                modes += '<option value="" disabled>video: models not installed</option>'
            else:
                modes += '<option value="" disabled>video: WanImageToVideo node missing</option>'
        avail = [f for f in self.listing(self.indir)
                 if not os.path.basename(f).startswith(MASK_PREFIX)]
        choices = "".join(
            '<option value="{0}"{2}>{1}</option>'.format(
                html.escape(f), html.escape(f),
                " selected" if f == L["existing"] else "")
            for f in avail) or "<option value=''>(none)</option>"
        def picker(chosen):
            return "<option value=''>(none)</option>" + "".join(
                '<option value="{0}"{2}>{1}</option>'.format(
                    html.escape(f), html.escape(f),
                    " selected" if f == chosen else "")
                for f in avail)
        choices2, choices3 = picker(L["existing2"]), picker(L["existing3"])
        slots = len(self.graph.image_fields())
        if slots >= 2:
            extra = ("Second image &mdash; another person or object to bring "
                     "in" + (", or the end frame for video"
                             if self.graph.can_end_frame() else ""))
            third = ("Third image" if slots >= 3
                     else "Third image &mdash; this encoder only takes two")
        else:
            extra = ("Second image &mdash; end frame for video only; this "
                     "encoder takes one image for editing")
            third = "Third image &mdash; not supported by this encoder"
        outs = "".join(self.cells(self.listing(self.outdir))) \
            or '<p class="note">nothing yet</p>'
        status = "ComfyUI at {} · encoder {}".format(
            self.comfy, self.graph.encoder or "NONE")
        if self.graph.notes:
            status += " · " + "; ".join(self.graph.notes)
        refresh, live = self.live_block()
        body = PAGE.format(refresh=refresh, live=live,
                           status=html.escape(status), msg=msg, choices=choices,
                           models="".join(opts) or "<option value=''>(no Qwen edit model installed)</option>",
                           loras=loras, lstr=html.escape(str(lstr)),
                           loras2=loras2, lstr2=html.escape(str(lstr2)),
                           modes=modes, length=length, size=html.escape(size),
                           clips=html.escape(str(L["clips"])),
                           count=html.escape(str(M.get("count", "1"))),
                           choices2=choices2, choices3=choices3,
                           extra=extra, third=third,
                           outs=outs, last=html.escape(L["prompt"]),
                           negative=html.escape(L.get("negative", "")),
                           maskgrid=self.mask_grid(L.get("existing", "")),
                           denoise=html.escape(str(M.get("denoise", "1.0"))),
                           seed=html.escape(str(M.get("seed", ""))),
                           steps=steps, cfg=html.escape(str(cfg))).encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def ensure_mp4(self, rel):
        """The playable mp4 for a clip, or None while one is being made."""
        if not rel.lower().endswith(".webm"):
            return rel
        mp4 = rel[:-5] + ".mp4"
        if os.path.isfile(os.path.join(self.outdir, mp4)):
            return mp4
        src = os.path.join(self.outdir, rel)
        with MP4_LOCK:
            if rel in MP4_JOBS or not os.path.isfile(src):
                return None
            MP4_JOBS.add(rel)
        comfy_dir = os.path.dirname(self.outdir)
        dst = os.path.join(self.outdir, mp4)

        def run():
            try:
                to_mp4(src, dst, comfy_dir)
            except Exception as e:
                print("mp4 of %s: %s" % (rel, e), file=sys.stderr)
            finally:
                with MP4_LOCK:
                    MP4_JOBS.discard(rel)

        threading.Thread(target=run, daemon=True).start()
        return None

    def cells(self, rels):
        for r in rels:
            better = self.ensure_mp4(r)
            yield cell(better or r,
                       prompt_of(os.path.join(self.outdir, better or r)),
                       converting=(better is None))

    def sweep_masks(self, keep_minutes=180):
        """Bin masks from runs that are long finished."""
        cut = time.time() - keep_minutes * 60
        try:
            for f in os.listdir(self.indir):
                if not f.startswith(MASK_PREFIX):
                    continue
                p = os.path.join(self.indir, f)
                if os.path.isfile(p) and os.path.getmtime(p) < cut:
                    os.remove(p)
        except Exception:
            pass          # tidying is never worth failing a run over

    def mask_rows(self, w, h):
        """How many rows make cells roughly square on this photo."""
        return max(8, min(64, int(round(MASK_COLS * (h or 1) / float(w or 1)))))

    def mask_grid(self, name):
        """The photo, tappable, with what is marked tinted over it.

        The tint is a grid of plain divs and takes no taps of its own -
        the taps go to the photo underneath, which is an <input
        type="image"> and reports where it was hit. Two of those make a
        rectangle. It is the one pointing device HTML has without
        JavaScript, and it beats hunting for cells by a mile.
        """
        if not name or not self.graph.masks_ok():
            return ""
        p = os.path.join(self.indir, os.path.basename(name))
        if not os.path.isfile(p):
            return ""
        w, h = image_size(p)
        rows = self.mask_rows(w, h)
        on = set(self.last.get("mask") or ())
        corner = self.last.get("corner") or ""
        tiles = "".join(
            '<i class="{k}"></i>'.format(
                k=("c" if "%d-%d" % (r, c) == corner
                   else "on" if "%d-%d" % (r, c) in on else ""))
            for r in range(rows) for c in range(MASK_COLS))
        if corner:
            state = "one corner set &mdash; tap the opposite one."
        elif on:
            state = ("%d of %d cells marked. Tap two corners again to add "
                     "another area." % (len(on), rows * MASK_COLS))
        else:
            state = "nothing marked yet &mdash; the whole picture is editable."
        cw = 512
        ch = max(1, int(round(cw * (h or 1) / float(w or 1))))
        if ch > 1024:                       # a very tall photo
            ch, cw = 1024, max(1, int(round(1024 * (w or 1) / float(h or 1))))
        # BRUSH_JS is appended, never formatted into: it is mostly braces,
        # and str.format would read every one of them as a field.
        return MASK.format(src=urllib.parse.quote(os.path.basename(name)),
                           cols=MASK_COLS, tiles=tiles, state=state,
                           w=MASK_W, cw=cw, ch=ch,
                           cells=",".join(sorted(on))) + BRUSH_JS

    def live_block(self):
        """What is happening right now, as a fragment above the form.

        This used to be its own page, which meant every run ended on a
        dead end: read the result, press back, retype. Putting it here
        costs a meta refresh while something is running and gives back a
        form that is still filled in when it finishes.
        """
        s = LIVE.snapshot()
        # Between the images of a batch the current job is finished while
        # the batch is not, and without this the page flashes "done" with
        # one result in it before carrying on.
        mid_batch = bool(s["clips"]) and s["clip"] < s["clips"]
        running = (bool(s["prompt_id"]) or mid_batch) and \
                  (not s["done"] or mid_batch)
        el = int(time.time() - s["started"])
        elapsed = "%d:%02d" % (el // 60, el % 60)
        out = []
        if running:
            # With previews off there is otherwise nothing on screen but a
            # bar, and no way to tell which of two similar attempts is the
            # one currently running.
            if s["clips"] > 1:
                out.append('<p class="note">{} {} of {}</p>'.format(
                    s.get("unit", "clip"), s["clip"], s["clips"]))
            if s["label"]:
                out.append('<p class="said">{}</p>'.format(html.escape(s["label"])))
            if s["max"]:
                pct = min(100, int(100.0 * s["value"] / s["max"]))
                # Elapsed alone answers the wrong question. Once a few
                # steps have gone by their rate is a decent guide to the
                # rest - and on a chained video, to the clips after this
                # one as well.
                eta = ""
                if s["value"] >= 2 and el > 0:
                    per = el / float(s["value"])
                    rest = per * (s["max"] - s["value"])
                    if s["clips"] > 1:
                        rest += per * s["max"] * (s["clips"] - s["clip"])
                    m, sec = int(rest) // 60, int(rest) % 60
                    eta = (" &middot; about %dm left" % m if m
                           else " &middot; under a minute left")
                    if m > 90:
                        eta = " &middot; about %dh%02dm left" % (m // 60, m % 60)
                out.append('<div class="bar"><i style="width:{}%"></i></div>'
                           '<p class="note">step {} of {} &middot; {} elapsed{}'
                           '</p>'.format(pct, s["value"], s["max"], elapsed, eta))
            else:
                out.append('<p class="note">starting up &middot; {} elapsed. '
                           'Loading a model off disk takes a while the first '
                           'time.</p>'.format(elapsed))
            if s["preview"]:
                out.append('<img class="shot" src="/preview?s={}" alt="">'
                           .format(s["seq"]))
                out.append('<p class="note">A rough decode of the latent, not '
                           'the final image - it sharpens as it goes.</p>')
            elif s["max"]:
                out.append('<p class="note">No preview frames are arriving. '
                           'ComfyUI only sends them when started with '
                           '<code>--preview-method auto</code>.</p>')
            out.append('<form class="stop" method="post" action="/stop">'
                       '<button type="submit">Stop</button></form>')
            refresh = '<meta http-equiv="refresh" content="3">'
        elif s["error"]:
            out.append('<p class="err">{}</p>'.format(html.escape(s["error"][:600])))
            refresh = ""
        elif s["prompt_id"]:
            outs = [o for o in s["outputs"]
                    if os.path.isfile(os.path.join(self.outdir, o))] \
                or self.listing(self.outdir, 1)
            if s["reel"]:
                out.append('<p class="ok">{} &mdash; the whole thing, joined'
                           '</p>'.format(html.escape(s["label"])))
                out.append('<div class="grid">' + cell(s["reel"]) + '</div>')
            elif s["clips"] > 1:
                out.append('<p class="ok">{}</p>'.format(html.escape(s["label"])))
            out.append('<p class="ok">done in {}</p>'.format(elapsed))
            out.append('<div class="grid">'
                       + "".join(self.cells(outs)) + '</div>')
            refresh = ""
        else:
            refresh = ""
        if out:
            out.insert(0, '<div class="live">')
            out.append('</div>')
        return refresh, "".join(out)

    def redirect(self, to):
        self.send_response(303)
        self.send_header("Location", to)
        self.send_header("Content-Length", "0")
        self.end_headers()

    # --------------------------------------------------------------- http
    def do_GET(self):
        if not self.authed():
            return
        u = urllib.parse.urlsplit(self.path)
        if u.path == "/status":
            return self.redirect("/")     # it all lives on one page now
        if u.path == "/view":
            rel = urllib.parse.parse_qs(u.query).get("n", [""])[0]
            p = os.path.abspath(os.path.join(self.outdir, rel))
            if not p.startswith(os.path.abspath(self.outdir) + os.sep) \
               or not os.path.isfile(p):
                self.send_response(404); self.send_header("Content-Length", "0")
                self.end_headers(); return
            q = urllib.parse.quote(rel)
            is_video = rel.lower().endswith(VID_EXT)
            media = ('<video controls playsinline preload="metadata" '
                     'src="/out?n={q}"></video>' if is_video
                     else '<img src="/out?n={q}" alt="">').format(q=q)
            said = prompt_of(p)
            body = VIEW.format(
                name=html.escape(os.path.basename(rel)), q=q, media=media,
                said='<p class="said">%s</p>' % html.escape(said) if said else "",
                reuse=("" if is_video else
                       '<a class="btn plain" href="/reuse?n=%s">edit this one'
                       '</a>' % q)
                      + ('<a class="btn plain" href="/useprompt?n=%s">use this '
                         'prompt</a>' % q if said else "")).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers(); self.wfile.write(body); return
        if u.path == "/useprompt":
            rel = urllib.parse.parse_qs(u.query).get("n", [""])[0]
            p = os.path.abspath(os.path.join(self.outdir, rel))
            if p.startswith(os.path.abspath(self.outdir) + os.sep) \
               and os.path.isfile(p):
                said = prompt_of(p)
                if said:
                    H.last["prompt"] = said
            return self.redirect("/")
        if u.path == "/reuse":
            # Editing an edit is the normal second step, and it should not
            # mean downloading the result and uploading it again.
            rel = urllib.parse.parse_qs(u.query).get("n", [""])[0]
            src = os.path.abspath(os.path.join(self.outdir, rel))
            if src.startswith(os.path.abspath(self.outdir) + os.sep) \
               and os.path.isfile(src) and not src.lower().endswith(VID_EXT):
                nm = time.strftime("%H%M%S_") + safe(os.path.basename(src))
                os.makedirs(self.indir, exist_ok=True)
                with open(src, "rb") as a:
                    with open(os.path.join(self.indir, nm), "wb") as b:
                        b.write(a.read())
                H.last["existing"] = nm
            return self.redirect("/")
        if u.path == "/preview":
            blob, mime = LIVE.frame()
            if not blob:
                self.send_response(404); self.send_header("Content-Length", "0")
                self.end_headers(); return
            self.send_response(200)
            self.send_header("Content-Type", mime)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(blob)))
            self.end_headers(); self.wfile.write(blob); return
        if u.path == "/src":
            # The mask grid needs the photo itself behind it. Outputs are
            # served from /out; this is the same idea for the input side.
            rel = os.path.basename(
                urllib.parse.parse_qs(u.query).get("n", [""])[0])
            p = os.path.abspath(os.path.join(self.indir, rel))
            if not p.startswith(os.path.abspath(self.indir) + os.sep) \
               or not os.path.isfile(p):
                self.send_response(404); self.send_header("Content-Length", "0")
                self.end_headers(); return
            with open(p, "rb") as fh:
                data = fh.read()
            self.send_response(200)
            self.send_header("Content-Type", MIME.get(
                os.path.splitext(p)[1].lower(), "application/octet-stream"))
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "max-age=300")
            self.end_headers(); self.wfile.write(data); return
        if u.path == "/out":
            rel = urllib.parse.parse_qs(u.query).get("n", [""])[0]
            p = os.path.abspath(os.path.join(self.outdir, rel))
            if not p.startswith(os.path.abspath(self.outdir) + os.sep) \
               or not os.path.isfile(p):
                self.send_response(404); self.send_header("Content-Length", "0")
                self.end_headers(); return
            blob = open(p, "rb").read()
            mime = MIME.get(os.path.splitext(p)[1].lower(),
                            "application/octet-stream")
            # Safari asks for a byte range before it will play anything,
            # and treats a plain 200 as a file it cannot stream. Serving
            # 206 is what makes a clip playable on the phone rather than
            # only downloadable.
            start, end = 0, len(blob) - 1
            rng = re.match(r"bytes=(\d*)-(\d*)\s*$",
                           self.headers.get("Range", "") or "")
            partial = False
            if rng and blob:
                if rng.group(1):
                    start = int(rng.group(1))
                    if rng.group(2):
                        end = min(int(rng.group(2)), end)
                else:                       # bytes=-N means the last N bytes
                    start = max(0, len(blob) - int(rng.group(2) or 0))
                if start > end or start >= len(blob):
                    self.send_response(416)
                    self.send_header("Content-Range", "bytes */%d" % len(blob))
                    self.send_header("Content-Length", "0")
                    self.end_headers(); return
                partial = True
            body = blob[start:end + 1]
            self.send_response(206 if partial else 200)
            self.send_header("Content-Type", mime)
            self.send_header("Accept-Ranges", "bytes")
            if partial:
                self.send_header("Content-Range", "bytes %d-%d/%d"
                                 % (start, end, len(blob)))
            if urllib.parse.parse_qs(u.query).get("dl"):
                self.send_header("Content-Disposition",
                                 'attachment; filename="%s"' % os.path.basename(p))
            self.send_header("Content-Length", str(len(body)))
            self.end_headers(); self.wfile.write(body); return
        self.render()

    def stop_run(self):
        """Interrupt what is running now, and drop whatever is queued behind.

        Both halves are needed: /interrupt ends only the job in flight, so
        a batch of six would carry straight on to the seventh. The flag is
        what stops our own workers, which sit between jobs where ComfyUI
        cannot reach them.
        """
        LIVE.request_cancel()
        for path, payload in (("/interrupt", {}), ("/queue", {"clear": True})):
            try:
                api(self.comfy, path, payload, timeout=10)
            except Exception:
                pass     # both answer with an empty body, which api() cannot read
        LIVE.stopped()
        return self.redirect("/")

    def do_POST(self):
        if not self.authed():
            return
        if urllib.parse.urlsplit(self.path).path == "/stop":
            return self.stop_run()
        ctype = self.headers.get("Content-Type", "")
        m = re.search(r'boundary=(?:"([^"]+)"|([^;]+))', ctype)
        if not m:
            return self.render('<p class="err">not a form post</p>')
        n = int(self.headers.get("Content-Length") or 0)
        if n <= 0 or n > MAX_LORA:
            return self.render('<p class="err">too large</p>')
        body = self.rfile.read(n)

        fields, uploads = {}, {}
        for name, fn, data in parse_multipart(body, (m.group(1) or m.group(2)).strip().encode()):
            if fn and data:
                uploads[name] = (safe(fn), data)   # keyed: two pickers now
            elif name:
                fields[name] = data.decode("utf-8", "replace").strip()

        def stash(field, existing):
            up = uploads.get(field)
            if up:
                os.makedirs(self.indir, exist_ok=True)
                nm = time.strftime("%H%M%S_") + up[0]
                with open(os.path.join(self.indir, nm), "wb") as fh:
                    fh.write(up[1])
                return nm
            return fields.get(existing) or ""

        # The grid can only be drawn over a photo the server knows about,
        # and picking one in a dropdown tells it nothing without
        # JavaScript. So the mask has its own button: same form, same
        # fields, formaction="/mask" - it stores the choice, uploads the
        # photo if that is where it came from, draws the grid, and runs
        # nothing.
        # The mask travels in one hidden field now, not a checkbox per
        # cell: taps land on the photo itself, and the tint over it is
        # only a picture of what is already marked.
        def cells_now():
            return [v for v in (fields.get("mcells") or "").split(",") if v]

        if urllib.parse.urlsplit(self.path).path == "/mask":
            picked = stash("photo", "existing")
            H.last["existing"] = picked or ""
            H.last["mask"] = sorted(set(cells_now()))
            if not picked:
                H.last["corner"] = ""
                return self.render('<p class="err">pick or upload a photo '
                                   'first, then press Mask.</p>')
            if not self.graph.masks_ok():
                return self.render('<p class="err">this ComfyUI has no '
                                   'SetLatentNoiseMask node, so it cannot '
                                   'mask.</p>')
            w, h = image_size(os.path.join(self.indir, picked))
            rows = self.mask_rows(w, h)
            op = urllib.parse.parse_qs(
                urllib.parse.urlsplit(self.path).query).get("op", [""])[0]
            if op:
                H.last["mask"] = mask_op(op, H.last["mask"], MASK_COLS, rows)
                H.last["corner"] = ""
                return self.render()

            # Where the photo was tapped, in the element's own pixels.
            # The browser sends these for an <input type="image"> and for
            # nothing else, which is the whole reason the photo is one.
            hit = tap_cell(fields.get("tap.x"), fields.get("tap.y"),
                           MASK_W, w, h, MASK_COLS, rows)
            if hit is None:
                H.last["corner"] = ""
                return self.render('<p class="ok">tap two opposite corners of '
                                   'the area that may change.</p>')
            first = H.last.get("corner") or ""
            if not first:
                H.last["corner"] = "%d-%d" % hit
                return self.render()
            fr, _, fc = first.partition("-")
            box = mask_op("box", ["%s-%s" % (fr, fc), "%d-%d" % hit],
                          MASK_COLS, rows)
            H.last["mask"] = sorted(set(H.last["mask"]) | set(box))
            H.last["corner"] = ""
            return self.render()

        # Remember the lot before anything can go wrong, so an error comes
        # back to a filled-in form rather than an empty one.
        for k in ("prompt", "model", "clips", "negative"):
            if k in fields:
                H.last[k] = fields[k]
        was = H.last["mode"]
        now = "video" if fields.get("mode") == "video" else "image"
        H.last["mode"] = now
        switched = (now != was)
        M = H.modes[now]
        if not switched:
            # Only trust the numbers on the form when the form was drawn
            # for this mode. On a switch they are the other mode's, which
            # the person never chose and cannot see they are sending.
            for key, field in (("lora", "lora"), ("lora2", "lora2"),
                               ("lstr", "lora_strength"),
                               ("lstr2", "lora_strength2"),
                               ("steps", "steps"), ("cfg", "cfg"),
                               ("denoise", "denoise"), ("count", "count"),
                               ("length", "length"), ("size", "size")):
                if field in fields:
                    M[key] = fields[field]

        # Adding a LoRA is its own errand: no photo, no prompt, nothing
        # generated. Handled before any of that is required.
        #
        # By link rather than by upload wherever possible: a forwarding
        # proxy sits between the phone and this box and refuses a body of
        # a few hundred megabytes, which is what a LoRA is. Fetching it
        # here has no such limit and runs at the box's speed rather than
        # the phone's.
        link = (fields.get("lora_url") or "").strip()
        if link:
            if not re.match(r"https?://", link):
                return self.render('<p class="err">that is not an http '
                                   'link</p>')
            d = os.path.join(os.path.dirname(self.outdir), "models", "loras")
            os.makedirs(d, exist_ok=True)
            tmp = os.path.join(d, "incoming.part")
            try:
                req = urllib.request.Request(
                    link, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=600) as r:
                    # The server's own name for it beats the one in the
                    # path, which for a signed CDN link is usually junk.
                    disp = r.headers.get("Content-Disposition", "")
                    m = re.search(r'filename\*?=(?:UTF-8'')?"?([^";]+)',
                                  disp)
                    name = (m.group(1) if m else
                            os.path.basename(urllib.parse.urlsplit(link).path))
                    got = 0
                    with open(tmp, "wb") as fh:
                        while True:
                            chunk = r.read(1 << 20)
                            if not chunk:
                                break
                            got += len(chunk)
                            if got > MAX_LORA:
                                raise RuntimeError("bigger than %dMB"
                                                   % (MAX_LORA // 2**20))
                            fh.write(chunk)
            except Exception as e:
                try:
                    os.remove(tmp)
                except OSError:
                    pass
                return self.render('<p class="err">could not fetch it: {}</p>'
                                   .format(html.escape(str(e)[:300])))
            with open(tmp, "rb") as fh:
                head = fh.read(16)
            if not (len(head) == 16 and b'{"' in head[8:16]):
                os.remove(tmp)
                return self.render('<p class="err">what came back is not a '
                                   'safetensors file. That usually means the '
                                   'link needed a login and you were handed a '
                                   'web page instead.</p>')
            name = safe(name) or "lora.safetensors"
            if not name.lower().endswith(".safetensors"):
                name = os.path.splitext(name)[0] + ".safetensors"
            os.replace(tmp, os.path.join(d, name))
            return self.render('<p class="ok">fetched {} ({:.0f}MB). It is in '
                               'the list below.</p>'.format(
                                   html.escape(name), got / 1e6))

        up = uploads.get("lora_file")
        if up:
            name, blob = up
            if not name.lower().endswith(".safetensors"):
                name += ".safetensors"
            if not (len(blob) > 16 and b'{"' in blob[8:16]):
                return self.render('<p class="err">that is not a safetensors '
                                   'file &mdash; the header does not look '
                                   'right. Nothing was saved.</p>')
            d = os.path.join(os.path.dirname(self.outdir), "models", "loras")
            os.makedirs(d, exist_ok=True)
            tmp = os.path.join(d, name + ".part")
            with open(tmp, "wb") as fh:
                fh.write(blob)
            os.replace(tmp, os.path.join(d, name))
            return self.render('<p class="ok">saved {} ({:.0f}MB). It is in '
                               'the list below.</p>'.format(
                                   html.escape(name), len(blob) / 1e6))

        image = stash("photo", "existing")
        image2 = stash("photo2", "existing2")
        image3 = stash("photo3", "existing3")
        end_image = image2          # in video mode the second one is the last frame
        H.last["existing"] = image or ""
        H.last["existing2"] = image2 or ""
        H.last["existing3"] = image3 or ""
        if not image:
            return self.render('<p class="err">pick or upload a photo</p>')

        # Turn the ticked cells into a greyscale png beside the photo.
        # A fresh name per run, because ComfyUI opens the file when the
        # node runs rather than when the job is queued - with one reused
        # name, the second image of a batch would sample against the mask
        # the third one had already written.
        marked = cells_now()
        H.last["mask"] = sorted(set(marked))
        mask_name = ""
        painted = fields.get("paintdata") or ""
        if painted and self.graph.masks_ok():
            nm = MASK_PREFIX + uuid.uuid4().hex[:8] + ".png"
            if paint_png(painted, os.path.join(self.indir, nm)):
                mask_name = nm
                self.sweep_masks()
        if not mask_name and marked and self.graph.masks_ok():
            cells = set()
            for v in marked:
                r, _, c = v.partition("-")
                if r.isdigit() and c.isdigit():
                    cells.add((int(r), int(c)))
            if cells:
                w, h = image_size(os.path.join(self.indir, image))
                mask_name = MASK_PREFIX + uuid.uuid4().hex[:8] + ".png"
                write_mask_png(os.path.join(self.indir, mask_name),
                               MASK_COLS, self.mask_rows(w, h), cells, w, h)
                self.sweep_masks()

        prompt = fields.get("prompt", "")
        if not prompt:
            return self.render('<p class="err">type an instruction</p>')
        if switched:
            return self.render(
                '<p class="ok">switched to {}, so the settings below are the '
                'ones that mode wants ({} steps, CFG {}). Check them and '
                'press Run.</p>'.format(now, M["steps"], M["cfg"]))

        LIVE.forget_error()
        try:
            steps = max(1, min(60, int(M["steps"])))
            cfg = float(M["cfg"])
            lstr = max(0.0, min(2.0, float(M["lstr"])))
            lstr2 = max(0.0, min(2.0, float(M["lstr2"])))
            denoise = max(0.1, min(1.0, float(M.get("denoise") or 1.0)))
            length = max(9, min(161, int(M["length"])))
            clips = max(1, min(8, int(fields.get("clips") or 1)))
            count = max(1, min(6, int(M.get("count") or 1)))
            typed = (M.get("seed") or "").strip()
            base_seed = int(typed) % 2**31 if typed else None
            w, _, h = (M["size"] or "832x480").lower().partition("x")
            width, height = int(w), int(h)
        except ValueError:
            return self.render('<p class="err">steps, CFG, strength and seed '
                               'must be numbers</p>')
        lora = M["lora"] or None
        lora2 = M["lora2"] or None
        mode = now
        M["size"] = "{}x{}".format(width, height)
        mm = self.models()
        for pick in (lora, lora2):
            if pick and pick not in mm["lora"]:
                return self.render('<p class="err">no such LoRA</p>')
            if not pick:
                continue
            low = pick.lower()
            wrong = (["qwen", "flux", "klein", "sdxl"] if mode == "video"
                     else ["wan"])
            hit = next((w for w in wrong if w in low), None)
            if hit:
                M["lora"] = M["lora2"] = ""
                return self.render(
                    '<p class="err">{} looks like a <b>{}</b> LoRA and this '
                    'is {} mode. ComfyUI will not refuse it - it loads what '
                    'matches, which is almost nothing, and degrades the rest. '
                    'That is what a ruined clip usually is. I have cleared '
                    'both LoRA slots; pick again.</p>'
                    .format(html.escape(pick), html.escape(hit), mode))

        # Both halves of one pair resolve to the same pair, so choosing
        # them in the two slots would apply it twice at double strength.
        # The dropdown lists both halves, so this is easy to do by
        # accident; the second one is dropped rather than compounded.
        chain, seen = [], set()
        for pick, strength in ((lora, lstr), (lora2, lstr2)):
            if not pick:
                continue
            hi, lo = lora_pair(pick, mm["lora"])
            if (hi, lo) in seen:
                continue
            seen.add((hi, lo))
            chain.append((hi, lo, strength))

        if mode == "video":
            if not self.video_ready(mm):
                return self.render('<p class="err">video is not set up on this '
                                   'box - run add-wan-video.sh</p>')
            # Wan wants both dimensions on a multiple of 16, and the frame
            # count on 4n+1; ComfyUI errors out unhelpfully otherwise.
            if steps < 8 and not any("light" in (n or "").lower()
                                     for n, _l, _st in chain):
                return self.render(
                    '<p class="err">{} steps is too few for Wan. It runs two '
                    'experts in sequence, so that is {} steps each, and the '
                    'result is mush. Use 20 - or 4 with a Lightning LoRA, '
                    'which is trained for it.</p>'
                    .format(steps, max(1, steps // 2)))
            width, height = (width // 16) * 16, (height // 16) * 16
            length = ((length - 1) // 4) * 4 + 1

            if clips > 1:
                # One line of instruction per clip; a single line means the
                # same thing happens throughout, which is usually what a
                # one-line answer meant.
                lines = [l.strip() for l in prompt.splitlines() if l.strip()]
                while len(lines) < clips:
                    lines.append(lines[-1])
                if LIVE.snapshot()["clips"] and not LIVE.snapshot()["done"]:
                    return self.render('<p class="err">a sequence is already '
                                       'running</p>')
                if not self.graph.info.get("ImageFromBatch"):
                    return self.render('<p class="err">this ComfyUI has no '
                                       'ImageFromBatch node, so a clip cannot '
                                       'hand its last frame to the next '
                                       'one</p>')
                LIVE.queued("", "starting %d clips" % clips, 1, clips)
                threading.Thread(target=sequence_worker, args=(dict(
                    graph=self.graph, comfy=self.comfy,
                    comfy_dir=os.path.dirname(self.outdir),
                    outdir=self.outdir, indir=self.indir,
                    high=mm["wan_high"][0], low=mm["wan_low"][0],
                    clip=mm["wan_clip"][0], vae=mm["wan_vae"][0],
                    image=image, prompts=lines[:clips], steps=steps, cfg=cfg,
                    seed=base_seed,
                    negative=H.last.get("negative", ""),
                    width=width, height=height, length=length, loras=chain,
                    timeout=7200),), daemon=True).start()
                return self.redirect("/")

            try:
                g = self.graph.build_video(
                    high=mm["wan_high"][0], low=mm["wan_low"][0],
                    clip=mm["wan_clip"][0], vae=mm["wan_vae"][0],
                    image=image, prompt=prompt,
                    negative=H.last.get("negative") or "static, still, blurry",
                    steps=steps, cfg=cfg,
                    seed=seed_for(base_seed),
                    width=width, height=height, length=length,
                    loras=chain, end_image=end_image or None)
            except Exception as e:
                return self.render('<p class="err">could not build the video '
                                   'graph: {}</p>'.format(html.escape(str(e))))
            try:
                r = api(self.comfy, "/prompt",
                        {"prompt": g, "client_id": CLIENT_ID})
            except urllib.error.HTTPError as e:
                return self.render('<p class="err">ComfyUI rejected it:</p>'
                                   '<pre class="note" style="white-space:pre-wrap">{}</pre>'
                                   .format(html.escape(e.read().decode("utf-8", "replace")[:800])),
                                   )
            except Exception as e:
                return self.render('<p class="err">{}</p>'.format(html.escape(str(e))))
            LIVE.queued(str(r.get("prompt_id", "")),
                        "{} frames at {}x{}{}{}".format(
                            length, width, height,
                            ", ending on your last frame" if end_image else "",
                            ", {} LoRA(s)".format(len(chain)) if chain else ""))
            return self.redirect("/")

        sel = fields.get("model", "")
        kind, _, name = sel.partition(":")
        if kind != "ckpt":
            if "qwen" not in name.lower():
                return self.render('<p class="err">{} is not a Qwen edit model. '
                                   'This form only drives Qwen editing; Klein '
                                   'lives in Forge on 7860.</p>'
                                   .format(html.escape(name)))
            if not (mm["clip"] and mm["vae"]):
                return self.render('<p class="err">no clip or vae installed</p>')

        # The graph differs only by prompt and seed between one image and
        # six, so it is built on demand rather than once.
        common = dict(images=[image, image2, image3], steps=steps, cfg=cfg,
                      sampler="euler", scheduler="simple", denoise=denoise,
                      width=width, height=height, mask=mask_name or None,
                      loras=[(hi, st) for hi, _lo, st in chain])

        def mk(text, seed):
            if kind == "ckpt":
                return self.graph.build(mode="checkpoint", ckpt=name,
                                        unet=None, clip=None, vae=None,
                                        prompt=text, seed=seed, **common)
            return self.graph.build(
                mode="separate", unet=name,
                clip=next((c for c in mm["clip"]
                           if "2.5_vl" in c or "2.5-vl" in c), mm["clip"][0]),
                vae=next((v for v in mm["vae"] if "qwen" in v.lower()),
                         mm["vae"][0]),
                ckpt=None, prompt=text, seed=seed, **common)

        if count > 1:
            lines = [l.strip() for l in prompt.splitlines() if l.strip()]
            while len(lines) < count:
                lines.append(lines[-1])     # one instruction, several rolls
            snap = LIVE.snapshot()
            if snap["clips"] and not snap["done"]:
                return self.render('<p class="err">a batch is already '
                                   'running</p>')
            try:
                mk(lines[0], 1)             # fail here, not in a thread
            except Exception as e:
                return self.render('<p class="err">could not build the graph: '
                                   '{}</p>'.format(html.escape(str(e))))
            LIVE.queued("", "starting %d images" % count, 1, count, "image")
            threading.Thread(target=batch_worker, args=(dict(
                build=mk, comfy=self.comfy, prompts=lines[:count],
                seed=base_seed, timeout=1800),), daemon=True).start()
            return self.redirect("/")

        try:
            g = mk(prompt, seed_for(base_seed))
        except Exception as e:
            return self.render('<p class="err">could not build the graph: {}</p>'
                               .format(html.escape(str(e))))

        try:
            r = api(self.comfy, "/prompt", {"prompt": g, "client_id": CLIENT_ID})
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:800]
            return self.render('<p class="err">ComfyUI rejected it:</p>'
                               '<pre class="note" style="white-space:pre-wrap">{}</pre>'
                               .format(html.escape(detail)))
        except Exception as e:
            return self.render('<p class="err">{}</p>'.format(html.escape(str(e))),
                               )

        # Straight to the progress page rather than back to the form: the
        # whole reason for queueing is to watch it happen.
        LIVE.queued(str(r.get("prompt_id", "")), prompt[:200])
        self.redirect("/")


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--port", type=int, default=7862)
    a.add_argument("--comfy", default="http://127.0.0.1:8288")
    a.add_argument("--indir", default=os.path.expanduser("~/ComfyUI/input"))
    a.add_argument("--outdir", default=os.path.expanduser("~/ComfyUI/output"))
    a.add_argument("--user", default="")
    a.add_argument("--password", default="")
    o = a.parse_args()

    H.comfy = o.comfy.rstrip("/")
    H.indir = os.path.abspath(os.path.expanduser(o.indir))
    H.outdir = os.path.abspath(os.path.expanduser(o.outdir))
    os.makedirs(H.indir, exist_ok=True)
    os.makedirs(H.outdir, exist_ok=True)
    if not (o.user and o.password):
        print("refusing to start without a password", file=sys.stderr)
        return 2
    H.credential = "{}:{}".format(o.user, o.password)

    print("reading", H.comfy + "/object_info")
    H.graph = Graph(api(H.comfy, "/object_info", timeout=120))
    print("  encoder:", H.graph.encoder)
    print("  its inputs:", H.graph.inputs_of(H.graph.encoder) if H.graph.encoder else "-")
    print("  image slots:", H.graph.image_fields() or "-")
    for n in H.graph.notes:
        print("  !!", n)
    try:
        args = os.popen("ps -eo args").read()
    except Exception:
        args = "--preview-method"
    if "main.py" in args and "--preview-method" not in args:
        print("  !! ComfyUI is running without --preview-method auto, so it\n"
              "     will not send preview frames. The progress page will show\n"
              "     step counts but no picture until it is restarted with it.",
              file=sys.stderr)

    threading.Thread(target=ws_listen, args=(H.comfy, LIVE),
                     daemon=True).start()
    print("serving on 0.0.0.0:{}".format(o.port))
    ThreadingHTTPServer(("0.0.0.0", o.port), H).serve_forever()


if __name__ == "__main__":
    sys.exit(main() or 0)
PYEOF
python3 -c "compile(open('$SCRIPT').read(),'$SCRIPT','exec')" || die "embedded server is broken"

if [ ! -s "$CREDS" ]; then
  printf 'user: edit\npass: %s\n' "$(openssl rand -base64 15 | tr -d '/+=')" > "$CREDS"
  chmod 600 "$CREDS"
fi
U=$(awk '/^user:/{print $2}' "$CREDS")
P=$(awk '/^pass:/{print $2}' "$CREDS")

# Photos uploaded through the inbox should be pickable here without a
# second upload, so ComfyUI's input folder is where the inbox writes.
if [ -d "$HOME/photo-inbox" ] && [ ! -L "$COMFY_DIR/input" ]; then
  say "pointing ComfyUI's input at the inbox"
  if [ -d "$COMFY_DIR/input" ] && [ -z "$(ls -A "$COMFY_DIR/input" 2>/dev/null)" ]; then
    rmdir "$COMFY_DIR/input" && ln -sfn "$HOME/photo-inbox" "$COMFY_DIR/input" \
      && echo "  input -> ~/photo-inbox"
  else
    echo "  (left alone; $COMFY_DIR/input is not empty)"
  fi
fi

say "starting"
tmux kill-session -t "=$SESSION" 2>/dev/null
tmux new-session -d -s "$SESSION" \
  "python3 -u '$SCRIPT' --port $PORT --comfy '$COMFY_URL' --user '$U' --password '$P' \
     --indir '$COMFY_DIR/input' --outdir '$COMFY_DIR/output' 2>&1 | tee -a '$LOG'"
sleep 4
tmux has-session -t "=$SESSION" 2>/dev/null || { tail -20 "$LOG"; die "did not start"; }
CODE=$(curl -sS -o /dev/null -w '%{http_code}' -u "$U:$P" "http://127.0.0.1:$PORT/" 2>/dev/null)

cat <<EOF

=== done ===

  local check   HTTP $CODE   (200 means serving)
  login         $U / $P    (also in $CREDS)
  session       $SESSION
  stop          tmux kill-session -t =$SESSION
  log           $LOG

Forward port $PORT in the Thunder console and open it.

While something is generating, /status shows a progress bar and a live
preview of the image forming. It refreshes itself with a meta tag, so it
works with JavaScript switched off.

What the log says about node discovery matters - if "encoder: None" or any
!! lines appear above, tell me and I will adjust the graph.
EOF
