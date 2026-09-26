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

import argparse, base64, hmac, html, json, os, re, sys, time, urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MAX_BYTES = 64 * 1024 * 1024
IMG_EXT = (".png", ".jpg", ".jpeg", ".webp", ".webm", ".mp4")


def api(url, path, payload=None, timeout=30):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        url + path, data=data,
        headers={"Content-Type": "application/json"} if data else {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


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
        fields = self.inputs_of(self.encoder) if self.encoder else []
        for cand in ("image1", "image"):
            if cand in fields:
                return cand
        return None

    def build(self, *, mode, unet, clip, vae, ckpt, image, prompt,
              steps, cfg, seed, sampler, scheduler, lora=None, lora_strength=1.0):
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

        # A LoRA sits between the loaders and everything downstream: the
        # sampler takes its model, the encoder its clip. Inserting it here
        # means the rest of the graph is written once either way.
        if lora:
            g["lo"] = {"class_type": "LoraLoader", "inputs": {
                "model": M, "clip": C, "lora_name": lora,
                "strength_model": lora_strength, "strength_clip": lora_strength}}
            M, C = ["lo", 0], ["lo", 1]

        g["im"] = {"class_type": "LoadImage", "inputs": {"image": image}}
        imgf = self.image_field()

        def enc(text):
            ins = {"clip": C, "prompt": text}
            if "vae" in self.inputs_of(self.encoder):
                ins["vae"] = V
            if imgf:
                ins[imgf] = ["im", 0]
            return {"class_type": self.encoder, "inputs": ins}

        g["po"] = enc(prompt)
        g["ne"] = enc("")
        g["la"] = {"class_type": "VAEEncode",
                   "inputs": {"pixels": ["im", 0], "vae": V}}
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


    def build_video(self, *, high, low, clip, vae, image, prompt, negative,
                    steps, cfg, seed, width, height, length, lora=None,
                    lora_strength=1.0):
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
        MH, ML, C = ["uh", 0], ["ul", 0], ["cl", 0]
        if lora:
            # Both experts have to carry it, or the halves disagree halfway
            # through the same clip.
            g["lh"] = {"class_type": "LoraLoader", "inputs": {
                "model": MH, "clip": C, "lora_name": lora,
                "strength_model": lora_strength, "strength_clip": lora_strength}}
            g["ll"] = {"class_type": "LoraLoader", "inputs": {
                "model": ML, "clip": C, "lora_name": lora,
                "strength_model": lora_strength, "strength_clip": 1.0}}
            MH, ML, C = ["lh", 0], ["ll", 0], ["lh", 1]

        g["po"] = {"class_type": "CLIPTextEncode",
                   "inputs": {"clip": C, "text": prompt}}
        g["ne"] = {"class_type": "CLIPTextEncode",
                   "inputs": {"clip": C, "text": negative}}
        g["wv"] = {"class_type": self.i2v, "inputs": {
            "positive": ["po", 0], "negative": ["ne", 0], "vae": ["va", 0],
            "width": width, "height": height, "length": length,
            "batch_size": 1, "start_image": ["im", 0]}}

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


PAGE = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Edit</title><style>
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
.card img{{width:100%;border-radius:8px;display:block;background:#8882}}
.note{{color:#888;font-size:14px}} .err{{color:#c55}} .ok{{color:#4a4}}
</style></head><body>
<h1>Edit a photo</h1>
<p class="sub">{status}</p>
{msg}
<form method="post" enctype="multipart/form-data" action="/">
  <label>Photo — upload one</label>
  <input type="file" name="photo" accept="image/*">
  <label>…or pick one already on the box</label>
  <select name="existing">{choices}</select>

  <label>Instruction</label>
  <textarea name="prompt" placeholder="change the shirt to green">{last}</textarea>

  <label>What to do</label>
  <select name="mode">{modes}</select>

  <label>Model (image editing only)</label>
  <select name="model">{models}</select>

  <label>LoRA (optional)</label>
  <select name="lora">{loras}</select>
  <label>LoRA strength</label>
  <input type="text" name="lora_strength" value="{lstr}">

  <div class="row">
    <div><label>Steps</label><input type="number" name="steps" value="{steps}" min="1" max="60"></div>
    <div><label>CFG</label><input type="text" name="cfg" value="{cfg}"></div>
  </div>
  <div class="row">
    <div><label>Frames (video)</label><input type="number" name="length" value="{length}" min="9" max="161"></div>
    <div><label>Size</label><input type="text" name="size" value="{size}"></div>
  </div>
  <p class="note">Editing: the merged AIO wants 4 steps and CFG 1; the
  separate bf16 wants 20 and CFG 3. Video: 20 steps, CFG 3.5, 49 frames
  (about 3s). Identity holds over short clips and drifts over long ones.</p>
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


class H(BaseHTTPRequestHandler):
    comfy = "http://127.0.0.1:8288"
    indir = ""
    outdir = ""
    credential = ""
    graph = None
    last_prompt = ""

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

    def render(self, msg="", steps=4, cfg="1.0", lora="", lstr="1.0",
               mode="image", length=49, size="832x480"):
        m = self.models()
        opts = []
        for c in m["ckpt"]:
            opts.append('<option value="ckpt:{0}">{1} (merged, 4 steps)</option>'
                        .format(html.escape(c), html.escape(c)))
        # diffusion_models holds whatever else is installed - Klein, on this
        # box. Wrapping a Qwen graph around a FLUX model fails in a way that
        # looks like a broken pipeline, so only offer what belongs here.
        for u in [x for x in m["unet"] if "qwen" in x.lower()]:
            opts.append('<option value="unet:{0}">{1} (separate, 20 steps)</option>'
                        .format(html.escape(u), html.escape(u)))
        loras = '<option value="">(none)</option>' + "".join(
            '<option value="{0}"{2}>{1}</option>'.format(
                html.escape(l), html.escape(l), " selected" if l == lora else "")
            for l in m["lora"])
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
        choices = "".join(
            '<option value="{0}">{1}</option>'.format(html.escape(f), html.escape(f))
            for f in self.listing(self.indir)) or "<option value=''>(none)</option>"
        outs = "".join(
            '<div class="card"><a href="/out?n={q}"><img loading="lazy" '
            'src="/out?n={q}" alt=""></a></div>'.format(q=urllib.parse.quote(r))
            for r in self.listing(self.outdir)) or '<p class="note">nothing yet</p>'
        status = "ComfyUI at {} · encoder {}".format(
            self.comfy, self.graph.encoder or "NONE")
        if self.graph.notes:
            status += " · " + "; ".join(self.graph.notes)
        body = PAGE.format(status=html.escape(status), msg=msg, choices=choices,
                           models="".join(opts) or "<option value=''>(no Qwen edit model installed)</option>",
                           loras=loras, lstr=html.escape(str(lstr)),
                           modes=modes, length=length, size=html.escape(size),
                           outs=outs, last=html.escape(self.last_prompt),
                           steps=steps, cfg=html.escape(str(cfg))).encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    # --------------------------------------------------------------- http
    def do_GET(self):
        if not self.authed():
            return
        u = urllib.parse.urlsplit(self.path)
        if u.path == "/out":
            rel = urllib.parse.parse_qs(u.query).get("n", [""])[0]
            p = os.path.abspath(os.path.join(self.outdir, rel))
            if not p.startswith(os.path.abspath(self.outdir) + os.sep) \
               or not os.path.isfile(p):
                self.send_response(404); self.send_header("Content-Length", "0")
                self.end_headers(); return
            blob = open(p, "rb").read()
            ext = os.path.splitext(p)[1].lower().lstrip(".")
            self.send_response(200)
            self.send_header("Content-Type", "image/" + ("jpeg" if ext == "jpg" else ext))
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
            return self.render('<p class="err">too large</p>')
        body = self.rfile.read(n)

        fields, upload = {}, None
        for name, fn, data in parse_multipart(body, (m.group(1) or m.group(2)).strip().encode()):
            if fn and data:
                upload = (safe(fn), data)
            elif name:
                fields[name] = data.decode("utf-8", "replace").strip()

        image = ""
        if upload:
            os.makedirs(self.indir, exist_ok=True)
            image = time.strftime("%H%M%S_") + upload[0]
            with open(os.path.join(self.indir, image), "wb") as fh:
                fh.write(upload[1])
        elif fields.get("existing"):
            image = fields["existing"]
        if not image:
            return self.render('<p class="err">pick or upload a photo</p>')

        prompt = fields.get("prompt", "")
        if not prompt:
            return self.render('<p class="err">type an instruction</p>')
        self.last_prompt = prompt

        try:
            steps = max(1, min(60, int(fields.get("steps") or 4)))
            cfg = float(fields.get("cfg") or 1.0)
            lstr = max(0.0, min(2.0, float(fields.get("lora_strength") or 1.0)))
            length = max(9, min(161, int(fields.get("length") or 49)))
            w, _, h = (fields.get("size") or "832x480").lower().partition("x")
            width, height = int(w), int(h)
        except ValueError:
            return self.render('<p class="err">steps, CFG and strength must be numbers</p>')
        lora = fields.get("lora") or None
        mode = "video" if fields.get("mode") == "video" else "image"
        size = "{}x{}".format(width, height)
        keep = dict(steps=steps, cfg=cfg, lora=lora or "", lstr=lstr,
                    mode=mode, length=length, size=size)
        mm = self.models()
        if lora and lora not in mm["lora"]:
            return self.render('<p class="err">no such LoRA</p>', **keep)

        if mode == "video":
            if not self.video_ready(mm):
                return self.render('<p class="err">video is not set up on this '
                                   'box - run add-wan-video.sh</p>', **keep)
            # Wan wants both dimensions on a multiple of 16, and the frame
            # count on 4n+1; ComfyUI errors out unhelpfully otherwise.
            width, height = (width // 16) * 16, (height // 16) * 16
            length = ((length - 1) // 4) * 4 + 1
            try:
                g = self.graph.build_video(
                    high=mm["wan_high"][0], low=mm["wan_low"][0],
                    clip=mm["wan_clip"][0], vae=mm["wan_vae"][0],
                    image=image, prompt=prompt,
                    negative="static, still, blurry, distorted",
                    steps=steps, cfg=cfg,
                    seed=int(time.time() * 1000) % 2**31,
                    width=width, height=height, length=length,
                    lora=lora, lora_strength=lstr)
            except Exception as e:
                return self.render('<p class="err">could not build the video '
                                   'graph: {}</p>'.format(html.escape(str(e))), **keep)
            try:
                r = api(self.comfy, "/prompt", {"prompt": g})
            except urllib.error.HTTPError as e:
                return self.render('<p class="err">ComfyUI rejected it:</p>'
                                   '<pre class="note" style="white-space:pre-wrap">{}</pre>'
                                   .format(html.escape(e.read().decode("utf-8", "replace")[:800])),
                                   **keep)
            except Exception as e:
                return self.render('<p class="err">{}</p>'.format(html.escape(str(e))), **keep)
            return self.render('<p class="ok">queued {} frames at {}x{} ({}). '
                               'Video takes minutes, not seconds - reload in a '
                               'few.</p>'.format(length, width, height,
                                                 html.escape(str(r.get("prompt_id", "?")))),
                               **keep)

        sel = fields.get("model", "")
        kind, _, name = sel.partition(":")
        try:
            if kind == "ckpt":
                g = self.graph.build(mode="checkpoint", ckpt=name, unet=None,
                                     clip=None, vae=None, image=image,
                                     prompt=prompt, steps=steps, cfg=cfg,
                                     seed=int(time.time() * 1000) % 2**31,
                                     sampler="euler", scheduler="simple",
                                     lora=lora, lora_strength=lstr)
            elif "qwen" not in name.lower():
                return self.render('<p class="err">{} is not a Qwen edit model. '
                                   'This form only drives Qwen editing; Klein '
                                   'lives in Forge on 7860.</p>'
                                   .format(html.escape(name)), **keep)
            else:
                if not (mm["clip"] and mm["vae"]):
                    return self.render('<p class="err">no clip or vae installed</p>', **keep)
                g = self.graph.build(mode="separate", unet=name,
                                     clip=next((c for c in mm["clip"] if "2.5_vl" in c or "2.5-vl" in c), mm["clip"][0]),
                                     vae=next((v for v in mm["vae"] if "qwen" in v.lower()), mm["vae"][0]),
                                     ckpt=None, image=image, prompt=prompt,
                                     steps=steps, cfg=cfg,
                                     seed=int(time.time() * 1000) % 2**31,
                                     sampler="euler", scheduler="simple",
                                     lora=lora, lora_strength=lstr)
        except Exception as e:
            return self.render('<p class="err">could not build the graph: {}</p>'
                               .format(html.escape(str(e))), **keep)

        try:
            r = api(self.comfy, "/prompt", {"prompt": g})
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:800]
            return self.render('<p class="err">ComfyUI rejected it:</p>'
                               '<pre class="note" style="white-space:pre-wrap">{}</pre>'
                               .format(html.escape(detail)), **keep)
        except Exception as e:
            return self.render('<p class="err">{}</p>'.format(html.escape(str(e))),
                               **keep)

        self.render('<p class="ok">queued ({}). Reload in a few seconds — it '
                    'appears under Results.</p>'.format(html.escape(str(r.get("prompt_id", "?")))),
                    **keep)


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
    print("  image field:", H.graph.image_field())
    for n in H.graph.notes:
        print("  !!", n)
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

What the log says about node discovery matters - if "encoder: None" or any
!! lines appear above, tell me and I will adjust the graph.
EOF
