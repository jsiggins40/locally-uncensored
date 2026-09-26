#!/usr/bin/env bash
#
# Start the JavaScript-free photo inbox, for uploading from a phone in
# Lockdown Mode.
#
#   bash i.sh              start on port 7861
#   PORT=7862 bash i.sh    somewhere else
#
# Then forward that port in the Thunder console and open the URL.
#
# Why this exists: Safari's Lockdown Mode restricts the web APIs Gradio's
# uploader needs, so photos silently fail to reach Forge. A plain HTML form
# needs no JavaScript at all, so the browser does the upload itself.

set -uo pipefail

PORT="${PORT:-7861}"
INBOX="${INBOX:-$HOME/photo-inbox}"
# Forge Neo writes to output/, singular; A1111 and Forge classic use
# outputs/. Guessing wrong shows an empty gallery next to a folder full of
# images, so take whichever is actually there.
if [ -z "${OUTDIR:-}" ]; then
  for d in "$HOME/sd-webui-forge-neo/output" "$HOME/sd-webui-forge-neo/outputs" \
           "$HOME/ForgeNeo/output" "$HOME/ForgeNeo/outputs"; do
    if [ -d "$d" ]; then OUTDIR="$d"; break; fi
  done
  OUTDIR="${OUTDIR:-$HOME/sd-webui-forge-neo/output}"
fi
CREDS="$HOME/photo-inbox-credentials.txt"
SCRIPT="$HOME/photo-inbox.py"
LOG="$HOME/photo-inbox.log"

say() { printf '\n=== %s ===\n' "$*"; }

command -v python3 >/dev/null || { echo "no python3"; exit 1; }

# The server is written out from here rather than fetched separately: one
# file to copy onto the box, and one fewer URL to paste on a phone keyboard
# where a truncated paste silently downloads a web page instead.
say "writing the server"
cat > "$SCRIPT" <<'PYEOF'
#!/usr/bin/env python3
"""A JavaScript-free upload page, for getting photos onto the box.

Safari's Lockdown Mode disables JIT and restricts the web APIs that Gradio's
uploader relies on, so its file picker opens, accepts a photo, and then
silently fails to send it. A plain <form enctype="multipart/form-data">
predates all of that: the browser itself does the POST, no script involved.
So this serves one form, parses the multipart body by hand, and writes the
file to disk.

iPhone photos arrive as HEIC, which PIL cannot open unaided; if pillow-heif
is installed they are converted to PNG on arrival, and if it is not they are
saved untouched and the page says so rather than pretending.

  python3 photo-inbox.py [--port 7861] [--dir ~/photo-inbox] [--user u] [--pass p]

Stdlib only - no pip install on a box that is already full.
"""

import argparse
import base64
import hmac
import html
import os
import re
import sys
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MAX_BYTES = 64 * 1024 * 1024
ALLOWED = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif", ".bmp", ".tif", ".tiff"}

try:  # optional, only used to rescue HEIC
    import pillow_heif  # type: ignore
    from PIL import Image  # type: ignore
    pillow_heif.register_heif_opener()
    HEIC_OK = True
except Exception:
    HEIC_OK = False


def parse_multipart(body: bytes, boundary: bytes):
    """Yield (field_name, filename, content) for each part.

    Written out rather than using cgi.FieldStorage, which is deprecated in
    3.11 and gone in 3.13 - this has to keep working after the next upgrade.
    """
    sep = b"--" + boundary
    for chunk in body.split(sep):
        if not chunk or chunk in (b"--", b"--\r\n"):
            continue
        head, _, data = chunk.partition(b"\r\n\r\n")
        if not _:
            continue
        disp = b""
        for line in head.split(b"\r\n"):
            if line.lower().startswith(b"content-disposition:"):
                disp = line
                break
        if not disp:
            continue
        name = re.search(rb'name="([^"]*)"', disp)
        fname = re.search(rb'filename="([^"]*)"', disp)
        yield (
            name.group(1).decode("utf-8", "replace") if name else "",
            fname.group(1).decode("utf-8", "replace") if fname else None,
            data[:-2] if data.endswith(b"\r\n") else data,
        )


def safe_name(raw: str) -> str:
    base = os.path.basename(raw or "upload")
    base = re.sub(r"[^A-Za-z0-9._-]", "_", base)[:100]
    return base or "upload"


PAGE = """<!doctype html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Photo inbox</title>
<style>
  :root {{ color-scheme: dark light; }}
  body {{ font: 17px/1.5 -apple-system,system-ui,sans-serif; margin: 0;
         padding: 24px 16px; max-width: 640px; }}
  h1 {{ font-size: 22px; margin: 0 0 4px; }}
  p.sub {{ color: #888; margin: 0 0 24px; }}
  form {{ border: 1px solid #8884; border-radius: 12px; padding: 20px;
          margin-bottom: 28px; }}
  input[type=file] {{ width: 100%; margin-bottom: 16px; }}
  button {{ font-size: 17px; padding: 12px 20px; width: 100%;
            border-radius: 10px; border: 0; background: #d2691e; color: #fff; }}
  .grid {{ display: grid; gap: 12px;
           grid-template-columns: repeat(auto-fill, minmax(140px, 1fr)); }}
  .card {{ border: 1px solid #8884; border-radius: 10px; overflow: hidden; }}
  .card img {{ width: 100%; height: 140px; object-fit: cover; display: block;
               background: #8882; }}
  .card .row {{ display: flex; align-items: center; gap: 6px; padding: 6px 8px; }}
  .card .nm {{ font-size: 12px; color: #888; flex: 1; overflow: hidden;
               text-overflow: ellipsis; white-space: nowrap; }}
  .card button {{ width: auto; padding: 4px 10px; font-size: 14px;
                  background: #8883; color: inherit; }}
  code {{ background: #8882; padding: 2px 6px; border-radius: 4px;
          word-break: break-all; }}
  .note {{ color: #888; font-size: 15px; }}
</style></head><body>
<h1>Photo inbox</h1>
<p class="sub">{count} file(s) waiting in <code>{dir}</code></p>

<form method="post" enctype="multipart/form-data" action="/">
  <input type="file" name="photo" accept="image/*" multiple>
  <button type="submit">Upload</button>
</form>

{msg}

<p class="note">Point img2img &rarr; Batch &rarr; input directory at
<code>{dir}</code> to use these.</p>
{heic}

<h2 style="font-size:18px">Waiting</h2>
<div class="grid">{items}</div>

<h2 style="font-size:18px">Results</h2>
<p class="note">Newest first, from <code>{outdir}</code>. Tap an image to open
it full size, then press and hold to save it to your camera roll.</p>
<div class="grid">{outs}</div>
</body></html>
"""


class Handler(BaseHTTPRequestHandler):
    server_version = "photo-inbox"
    directory = os.path.expanduser("~/photo-inbox")
    outdir = os.path.expanduser("~/sd-webui-forge-neo/outputs")
    credential = ""

    def log_message(self, fmt, *args):
        sys.stderr.write("%s %s\n" % (self.address_string(), fmt % args))

    # ---------------------------------------------------------------- auth
    def authed(self) -> bool:
        if not self.credential:
            return True
        got = self.headers.get("Authorization", "")
        want = "Basic " + base64.b64encode(self.credential.encode()).decode()
        if hmac.compare_digest(got, want):
            return True
        self.send_response(401)
        self.send_header("WWW-Authenticate", 'Basic realm="photo inbox"')
        self.send_header("Content-Length", "0")
        self.end_headers()
        return False

    # ---------------------------------------------------------------- page
    def render(self, msg=""):
        try:
            names = sorted(
                f for f in os.listdir(self.directory) if not f.startswith(".")
            )
        except OSError:
            names = []
        # Thumbnails are plain <img> tags and the delete control is a plain
        # form, so the whole page still works with Lockdown Mode on.
        items = "".join(
            '<div class="card"><img loading="lazy" src="/img?n={q}" alt="">'
            '<div class="row"><span class="nm">{e}</span>'
            '<form method="post" action="/" style="border:0;padding:0;margin:0">'
            '<input type="hidden" name="delete" value="{q}">'
            '<button type="submit">delete</button></form></div></div>'.format(
                q=urllib.parse.quote(n), e=html.escape(n))
            for n in names
        ) or '<p class="note">nothing yet</p>' 
        outs = "".join(
            '<div class="card"><a href="/out?n={q}"><img loading="lazy" '
            'src="/out?n={q}" alt=""></a>'
            '<div class="row"><span class="nm">{e}</span></div></div>'.format(
                q=urllib.parse.quote(r), e=html.escape(os.path.basename(r)))
            for r in self.recent_outputs()
        ) or '<p class="note">nothing generated yet</p>'
        heic = "" if HEIC_OK else (
            '<p class="note">HEIC conversion is unavailable, so iPhone photos '
            "are stored as-is and Forge may not read them. Set Camera &rarr; "
            "Formats &rarr; Most Compatible, or upload a screenshot.</p>"
        )
        body = PAGE.format(
            count=len(names), dir=html.escape(self.directory),
            outdir=html.escape(self.outdir),
            msg=msg, items=items, outs=outs, heic=heic,
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def recent_outputs(self, limit=60):
        """Newest images anywhere under outdir, as paths relative to it.

        Forge nests by date (outputs/img2img-images/2026-09-25/...), so this
        walks rather than lists, and sorts by mtime because the date folders
        do not sort usefully once the month rolls over.
        """
        found = []
        for root, _dirs, files in os.walk(self.outdir):
            for f in files:
                if os.path.splitext(f)[1].lower() in (".png", ".jpg", ".jpeg", ".webp"):
                    full = os.path.join(root, f)
                    try:
                        found.append((os.path.getmtime(full), full))
                    except OSError:
                        pass
        found.sort(reverse=True)
        return [os.path.relpath(p, self.outdir) for _m, p in found[:limit]]

    def resolve_out(self, rel: str):
        p = os.path.abspath(os.path.join(self.outdir, rel))
        root = os.path.abspath(self.outdir)
        return p if p.startswith(root + os.sep) and os.path.isfile(p) else None

    def resolve(self, name: str):
        """Absolute path inside the inbox, or None if it tries to escape."""
        p = os.path.abspath(os.path.join(self.directory, safe_name(name)))
        root = os.path.abspath(self.directory)
        return p if p.startswith(root + os.sep) and os.path.isfile(p) else None

    def do_GET(self):
        if not self.authed():
            return
        parts = urllib.parse.urlsplit(self.path)
        if parts.path == "/out":
            rel = urllib.parse.parse_qs(parts.query).get("n", [""])[0]
            path = self.resolve_out(rel)
            if not path:
                self.send_response(404)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            ext = os.path.splitext(path)[1].lower().lstrip(".")
            with open(path, "rb") as fh:
                blob = fh.read()
            self.send_response(200)
            self.send_header("Content-Type",
                             "image/" + {"jpg": "jpeg"}.get(ext, ext or "png"))
            self.send_header("Content-Length", str(len(blob)))
            self.end_headers()
            self.wfile.write(blob)
            return
        if parts.path == "/img":
            q = urllib.parse.parse_qs(parts.query).get("n", [""])[0]
            path = self.resolve(q)
            if not path:
                self.send_response(404)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            ext = os.path.splitext(path)[1].lower().lstrip(".")
            ctype = {"jpg": "jpeg", "tif": "tiff"}.get(ext, ext)
            with open(path, "rb") as fh:
                blob = fh.read()
            self.send_response(200)
            self.send_header("Content-Type", "image/" + (ctype or "octet-stream"))
            self.send_header("Content-Length", str(len(blob)))
            self.end_headers()
            self.wfile.write(blob)
            return
        self.render()

    def do_POST(self):
        if not self.authed():
            return
        ctype = self.headers.get("Content-Type", "")

        # The delete button is an ordinary form, so it arrives urlencoded
        # rather than multipart - handle it before looking for a boundary.
        if "application/x-www-form-urlencoded" in ctype:
            n = int(self.headers.get("Content-Length") or 0)
            fields = urllib.parse.parse_qs(self.rfile.read(n).decode("utf-8", "replace"))
            target = (fields.get("delete") or [""])[0]
            path = self.resolve(target)
            if path:
                try:
                    os.remove(path)
                    return self.render('<p style="color:#4a4">Deleted {}.</p>'.format(
                        html.escape(os.path.basename(path))))
                except OSError as exc:
                    return self.render('<p style="color:#c55">{}</p>'.format(
                        html.escape(str(exc))))
            return self.render('<p style="color:#c55">No such file.</p>')

        m = re.search(r'boundary=(?:"([^"]+)"|([^;]+))', ctype)
        if not m:
            return self.render('<p style="color:#c55">Not a file upload.</p>')
        boundary = (m.group(1) or m.group(2)).strip().encode()

        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0 or length > MAX_BYTES:
            return self.render(
                '<p style="color:#c55">Too large (limit {} MB).</p>'.format(
                    MAX_BYTES // 2**20))

        body = self.rfile.read(length)
        os.makedirs(self.directory, exist_ok=True)
        saved, skipped = [], []

        for _, filename, data in parse_multipart(body, boundary):
            if not filename or not data:
                continue
            name = safe_name(filename)
            ext = os.path.splitext(name)[1].lower()
            if ext not in ALLOWED:
                skipped.append("{} (type {})".format(name, ext or "?"))
                continue
            stamp = time.strftime("%H%M%S")
            path = os.path.join(self.directory, "{}_{}".format(stamp, name))
            with open(path, "wb") as fh:
                fh.write(data)
            # HEIC is what an iPhone produces by default and what Forge cannot
            # read; convert here so the far end never has to care.
            if ext in (".heic", ".heif") and HEIC_OK:
                try:
                    png = os.path.splitext(path)[0] + ".png"
                    Image.open(path).convert("RGB").save(png)
                    os.remove(path)
                    path = png
                except Exception as exc:
                    skipped.append("{} (convert failed: {})".format(name, exc))
            saved.append(os.path.basename(path))

        msg = ""
        if saved:
            msg += '<p style="color:#4a4">Saved: {}</p>'.format(
                html.escape(", ".join(saved)))
        if skipped:
            msg += '<p style="color:#c55">Skipped: {}</p>'.format(
                html.escape(", ".join(skipped)))
        self.render(msg or '<p style="color:#c55">Nothing was sent.</p>')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=7861)
    ap.add_argument("--dir", default=os.path.expanduser("~/photo-inbox"))
    ap.add_argument("--outdir",
                    default=os.path.expanduser("~/sd-webui-forge-neo/outputs"))
    ap.add_argument("--user", default="")
    ap.add_argument("--password", default="")
    a = ap.parse_args()

    Handler.directory = os.path.abspath(os.path.expanduser(a.dir))
    Handler.outdir = os.path.abspath(os.path.expanduser(a.outdir))
    os.makedirs(Handler.directory, exist_ok=True)
    # The forwarded port is public, so refuse to run naked.
    Handler.credential = (
        "{}:{}".format(a.user, a.password) if a.user and a.password else ""
    )
    if not Handler.credential:
        print("refusing to start without --user and --password: "
              "this port is reachable from the internet", file=sys.stderr)
        return 2

    print("serving {} on 0.0.0.0:{}".format(Handler.directory, a.port))
    print("results from", Handler.outdir)
    print("HEIC conversion:", "on" if HEIC_OK else "off (pillow-heif missing)")
    ThreadingHTTPServer(("0.0.0.0", a.port), Handler).serve_forever()


if __name__ == "__main__":
    sys.exit(main() or 0)
PYEOF
python3 -c "compile(open('$SCRIPT').read(), '$SCRIPT', 'exec')" \
  || { echo "the embedded server did not survive the copy"; exit 1; }

# The forwarded port is public, so it never runs without a password.
if [ ! -s "$CREDS" ]; then
  printf 'user: forge\npass: %s\n' "$(openssl rand -base64 15 | tr -d '/+=')" > "$CREDS"
  chmod 600 "$CREDS"
fi
U=$(awk '/^user:/{print $2}' "$CREDS")
P=$(awk '/^pass:/{print $2}' "$CREDS")

# Optional, and only worth a try: without it iPhone HEIC files are stored
# as-is and Forge cannot open them.
if ! python3 -c 'import pillow_heif' 2>/dev/null; then
  say "trying to add HEIC support"
  pip install -q --break-system-packages pillow-heif pillow 2>/dev/null \
    || pip install -q pillow-heif pillow 2>/dev/null \
    || echo "  (no luck - upload screenshots, or set Camera > Formats > Most Compatible)"
fi

mkdir -p "$INBOX"
tmux kill-session -t =inbox 2>/dev/null
tmux new-session -d -s inbox \
  "python3 '$SCRIPT' --port $PORT --dir '$INBOX' --outdir '$OUTDIR' --user '$U' --password '$P' 2>&1 | tee -a '$LOG'"
sleep 2

if ! tmux has-session -t =inbox 2>/dev/null; then
  echo "failed to start; see $LOG"; tail -5 "$LOG"; exit 1
fi

CODE=$(curl -sS -o /dev/null -w '%{http_code}' -u "$U:$P" "http://127.0.0.1:$PORT/" 2>/dev/null)
say "running"
cat <<EOF
  local check   HTTP $CODE  (200 means it is serving)
  inbox         $INBOX
  results       $OUTDIR
  login         $U / $P      (also in $CREDS)
  stop          tmux kill-session -t =inbox
  log           $LOG

Next:
  1. Thunder console -> forward port $PORT -> open the URL it gives you
  2. Sign in with the credentials above, pick photos, Upload
  3. In Forge: img2img -> Batch -> input directory: $INBOX

No JavaScript is involved, so Lockdown Mode leaves it alone.
EOF
