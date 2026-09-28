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
CREDS="$HOME/edit-form-credentials.txt"
SCRIPT="$HOME/edit-form.py"
LOG="$HOME/edit-form.log"
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

import argparse, base64, hmac, html, json, os, re, shutil, socket, struct
import subprocess, sys, threading, time, urllib.parse, urllib.request, uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MAX_BYTES = 64 * 1024 * 1024
IMG_EXT = (".png", ".jpg", ".jpeg", ".webp", ".webm", ".mp4")
VID_EXT = (".webm", ".mp4")
MIME = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".webp": "image/webp", ".gif": "image/gif",
        ".webm": "video/webm", ".mp4": "video/mp4"}


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


def cell(rel, prompt=""):
    """One tile in a gallery: a still, or a player for a clip.

    A clip in an <img> renders as a broken image, which is what a video
    mode does by default if nothing here separates the two. <video
    controls> needs no JavaScript, so it survives Lockdown Mode.
    """
    q = urllib.parse.quote(rel)
    if rel.lower().endswith(VID_EXT):
        return ('<div class="card"><video controls playsinline preload="metadata"'
                ' src="/out?n={q}"></video>'
                '<a class="dl" href="/out?n={q}&amp;dl=1">save it</a></div>'
                .format(q=q))
    extra = ""
    if prompt:
        # Shown in full and selectable, because copying by hand is the only
        # way to get at the clipboard without JavaScript - and the link
        # beside it is the thing you actually wanted the clipboard for.
        extra = ('<p class="said">{p}</p>'
                 '<a class="dl" href="/useprompt?n={q}">use this prompt '
                 '&rarr;</a>'.format(p=html.escape(prompt), q=q))
    return ('<div class="card"><a href="/out?n={q}">'
            '<img loading="lazy" src="/out?n={q}" alt=""></a>'
            '<a class="dl" href="/reuse?n={q}">edit this one &rarr;</a>{e}'
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
        self.reel = ""
        self.node = ""
        self.value = 0
        self.max = 0
        self.done = not prompt_id
        self.error = ""
        self.outputs = []
        self.started = time.time()

    def queued(self, prompt_id, label, clip=0, clips=0):
        """Called when this form posts a job, before ComfyUI says anything."""
        with self.lock:
            self._clear(prompt_id, label)
            self.clip, self.clips = clip, clips
            self.preview = b""
            self.seq += 1

    def snapshot(self):
        with self.lock:
            return dict(connected=self.connected, prompt_id=self.prompt_id,
                        label=self.label, node=self.node, value=self.value,
                        max=self.max, queue=self.queue, done=self.done,
                        error=self.error, outputs=list(self.outputs),
                        started=self.started, seq=self.seq,
                        clip=self.clip, clips=self.clips, reel=self.reel,
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
            elif t in ("execution_error", "execution_interrupted"):
                self.error = str(d.get("exception_message")
                                 or d.get("exception_type") or "interrupted")
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


def wait_done(prompt_id, timeout=7200):
    """Block until ComfyUI finishes that job, and hand back its outputs."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        s = LIVE.snapshot()
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
                negative="static, still, blurry, distorted",
                steps=cfg["steps"], cfg=cfg["cfg"],
                seed=(int(time.time() * 1000) + i * 7919) % 2**31,
                width=cfg["width"], height=cfg["height"],
                length=cfg["length"], loras=cfg["loras"],
                end_image=None, save_last=(i < n - 1))
            r = api(cfg["comfy"], "/prompt",
                    {"prompt": g, "client_id": CLIENT_ID})
            pid = str(r.get("prompt_id", ""))
            LIVE.queued(pid, "clip %d of %d - %s" % (i + 1, n, prompt[:60]),
                        i + 1, n)
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

    def inputs_of(self, node):
        spec = self.info.get(node, {}).get("input", {})
        return list(spec.get("required", {})) + list(spec.get("optional", {}))

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
              steps, cfg, seed, sampler, scheduler, loras=()):
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
        # also what sets the size of the output.
        g["la"] = {"class_type": "VAEEncode",
                   "inputs": {"pixels": ["im0", 0], "vae": V}}
        g["ks"] = {"class_type": "KSampler", "inputs": {
            "model": M, "positive": ["po", 0], "negative": ["ne", 0],
            "latent_image": ["la", 0], "seed": seed, "steps": steps,
            "cfg": cfg, "sampler_name": sampler, "scheduler": scheduler,
            "denoise": 1.0}}
        g["de"] = {"class_type": "VAEDecode",
                   "inputs": {"samples": ["ks", 0], "vae": V}}
        g["sv"] = {"class_type": "SaveImage",
                   "inputs": {"images": ["de", 0], "filename_prefix": "edit"}}
        return g


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
            g["lf"] = {"class_type": "ImageFromBatch", "inputs": {
                "image": ["de", 0], "batch_index": length - 1, "length": 1}}
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
.row{{display:flex;gap:10px}} .row>div{{flex:1}}
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
<form method="post" enctype="multipart/form-data" action="/">
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

  <label>Instruction</label>
  <textarea name="prompt" placeholder="change the shirt to green">{last}</textarea>

  <label>What to do</label>
  <select name="mode">{modes}</select>

  <label>Model (image editing only)</label>
  <select name="model">{models}</select>

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
  <div class="row">
    <div><label>Frames (video)</label><input type="number" name="length" value="{length}" min="9" max="161"></div>
    <div><label>Size</label><input type="text" name="size" value="{size}"></div>
  </div>
  <label>Clips to chain &mdash; each carries on from the last frame of the one before</label>
  <input type="number" name="clips" value="{clips}" min="1" max="8">
  <p class="note">For a long video, raise Clips. At 81 frames each that is
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
            "existing": "", "existing2": "", "existing3": ""}

    # ...but the sampler settings are per mode, and sharing them was a
    # quiet way to ruin a video. Four steps at CFG 1 is right for the
    # merged image model and is two steps per expert on Wan, which comes
    # out as mush. Switching mode now brings that mode's numbers with it.
    modes = {
        "image": {"steps": "4", "cfg": "1.0", "size": "1024x1024",
                  "length": "81", "lora": "", "lstr": "1.0",
                  "lora2": "", "lstr2": "1.0"},
        "video": {"steps": "20", "cfg": "3.5", "size": "832x480",
                  "length": "81", "lora": "", "lstr": "1.0",
                  "lora2": "", "lstr2": "1.0"},
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
        return [r for _m, r in out[:limit]]

    def models(self):
        o = self.graph.info
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
        avail = self.listing(self.indir)
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
                           choices2=choices2, choices3=choices3,
                           extra=extra, third=third,
                           outs=outs, last=html.escape(L["prompt"]),
                           steps=steps, cfg=html.escape(str(cfg))).encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def cells(self, rels):
        for r in rels:
            yield cell(r, prompt_of(os.path.join(self.outdir, r)))

    def live_block(self):
        """What is happening right now, as a fragment above the form.

        This used to be its own page, which meant every run ended on a
        dead end: read the result, press back, retype. Putting it here
        costs a meta refresh while something is running and gives back a
        form that is still filled in when it finishes.
        """
        s = LIVE.snapshot()
        running = bool(s["prompt_id"]) and not s["done"]
        el = int(time.time() - s["started"])
        elapsed = "%d:%02d" % (el // 60, el % 60)
        out = []
        if running:
            # With previews off there is otherwise nothing on screen but a
            # bar, and no way to tell which of two similar attempts is the
            # one currently running.
            if s["clips"] > 1:
                out.append('<p class="note">clip {} of {}</p>'.format(
                    s["clip"], s["clips"]))
            if s["label"]:
                out.append('<p class="said">{}</p>'.format(html.escape(s["label"])))
            if s["max"]:
                pct = min(100, int(100.0 * s["value"] / s["max"]))
                out.append('<div class="bar"><i style="width:{}%"></i></div>'
                           '<p class="note">step {} of {} &middot; {} elapsed'
                           '</p>'.format(pct, s["value"], s["max"], elapsed))
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

    def do_POST(self):
        if not self.authed():
            return
        ctype = self.headers.get("Content-Type", "")
        m = re.search(r'boundary=(?:"([^"]+)"|([^;]+))', ctype)
        if not m:
            return self.render('<p class="err">not a form post</p>')
        n = int(self.headers.get("Content-Length") or 0)
        if n <= 0 or n > MAX_BYTES:
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

        # Remember the lot before anything can go wrong, so an error comes
        # back to a filled-in form rather than an empty one.
        for k in ("prompt", "model", "clips"):
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
                               ("length", "length"), ("size", "size")):
                if field in fields:
                    M[key] = fields[field]

        image = stash("photo", "existing")
        image2 = stash("photo2", "existing2")
        image3 = stash("photo3", "existing3")
        end_image = image2          # in video mode the second one is the last frame
        H.last["existing"] = image or ""
        H.last["existing2"] = image2 or ""
        H.last["existing3"] = image3 or ""
        if not image:
            return self.render('<p class="err">pick or upload a photo</p>')

        prompt = fields.get("prompt", "")
        if not prompt:
            return self.render('<p class="err">type an instruction</p>')
        if switched:
            return self.render(
                '<p class="ok">switched to {}, so the settings below are the '
                'ones that mode wants ({} steps, CFG {}). Check them and '
                'press Run.</p>'.format(now, M["steps"], M["cfg"]))

        try:
            steps = max(1, min(60, int(M["steps"])))
            cfg = float(M["cfg"])
            lstr = max(0.0, min(2.0, float(M["lstr"])))
            lstr2 = max(0.0, min(2.0, float(M["lstr2"])))
            length = max(9, min(161, int(M["length"])))
            clips = max(1, min(8, int(fields.get("clips") or 1)))
            w, _, h = (M["size"] or "832x480").lower().partition("x")
            width, height = int(w), int(h)
        except ValueError:
            return self.render('<p class="err">steps, CFG and strength must be numbers</p>')
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
                    width=width, height=height, length=length, loras=chain,
                    timeout=7200),), daemon=True).start()
                return self.redirect("/")

            try:
                g = self.graph.build_video(
                    high=mm["wan_high"][0], low=mm["wan_low"][0],
                    clip=mm["wan_clip"][0], vae=mm["wan_vae"][0],
                    image=image, prompt=prompt,
                    negative="static, still, blurry, distorted",
                    steps=steps, cfg=cfg,
                    seed=int(time.time() * 1000) % 2**31,
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
        try:
            if kind == "ckpt":
                g = self.graph.build(mode="checkpoint", ckpt=name, unet=None,
                                     clip=None, vae=None,
                                     images=[image, image2, image3],
                                     prompt=prompt, steps=steps, cfg=cfg,
                                     seed=int(time.time() * 1000) % 2**31,
                                     sampler="euler", scheduler="simple",
                                     loras=[(hi, st) for hi, _lo, st in chain])
            elif "qwen" not in name.lower():
                return self.render('<p class="err">{} is not a Qwen edit model. '
                                   'This form only drives Qwen editing; Klein '
                                   'lives in Forge on 7860.</p>'
                                   .format(html.escape(name)))
            else:
                if not (mm["clip"] and mm["vae"]):
                    return self.render('<p class="err">no clip or vae installed</p>')
                g = self.graph.build(mode="separate", unet=name,
                                     clip=next((c for c in mm["clip"] if "2.5_vl" in c or "2.5-vl" in c), mm["clip"][0]),
                                     vae=next((v for v in mm["vae"] if "qwen" in v.lower()), mm["vae"][0]),
                                     ckpt=None, prompt=prompt,
                                     images=[image, image2, image3],
                                     steps=steps, cfg=cfg,
                                     seed=int(time.time() * 1000) % 2**31,
                                     sampler="euler", scheduler="simple",
                                     loras=[(hi, st) for hi, _lo, st in chain])
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
tmux kill-session -t =editform 2>/dev/null
tmux new-session -d -s editform \
  "python3 -u '$SCRIPT' --port $PORT --comfy '$COMFY_URL' --user '$U' --password '$P' 2>&1 | tee -a '$LOG'"
sleep 4
tmux has-session -t =editform 2>/dev/null || { tail -20 "$LOG"; die "did not start"; }
CODE=$(curl -sS -o /dev/null -w '%{http_code}' -u "$U:$P" "http://127.0.0.1:$PORT/" 2>/dev/null)

cat <<EOF

=== done ===

  local check   HTTP $CODE   (200 means serving)
  login         $U / $P    (also in $CREDS)
  stop          tmux kill-session -t =editform
  log           $LOG

Forward port $PORT in the Thunder console and open it.

While something is generating, /status shows a progress bar and a live
preview of the image forming. It refreshes itself with a meta tag, so it
works with JavaScript switched off.

What the log says about node discovery matters - if "encoder: None" or any
!! lines appear above, tell me and I will adjust the graph.
EOF
