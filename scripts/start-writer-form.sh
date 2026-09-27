#!/usr/bin/env bash
#
# A plain form that writes documents, in front of the local model.
#
#   bash w2.sh            start on port 7863
#   PORT=7864 bash w2.sh
#
# Asking a model for "a business plan" in one shot gets you two thousand
# words of throat-clearing. This asks for one section at a time, each with
# its own instruction and a summary of what came before, and writes the
# executive summary last - once there is something to summarise. That is
# most of the difference between a document you can send someone and a
# chat transcript you have to rewrite.
#
# No JavaScript, for the same reason as the photo form: Lockdown Mode.
# Progress arrives by meta refresh.

set -uo pipefail

PORT="${PORT:-7863}"
OLLAMA="${OLLAMA:-http://127.0.0.1:11434}"
CREDS="$HOME/writer-credentials.txt"
SCRIPT="$HOME/writer-form.py"
LOG="$HOME/writer-form.log"
OUTDIR="${OUTDIR:-$HOME/documents}"

say() { printf '\n=== %s ===\n' "$*"; }
die() { printf '\nFAILED: %s\n' "$*"; exit 1; }

curl -fsS -m 5 "$OLLAMA/api/tags" >/dev/null 2>&1 \
  || die "ollama is not answering at $OLLAMA (run add-writer.sh, or: tmux ls)"
MODEL="${MODEL:-$(cat "$HOME/.writer-model" 2>/dev/null)}"
[ -n "$MODEL" ] || MODEL=$(curl -fsS "$OLLAMA/api/tags" \
  | python3 -c 'import sys,json;m=json.load(sys.stdin).get("models") or [];print(m[0]["name"] if m else "")')
[ -n "$MODEL" ] || die "no model pulled yet (run add-writer.sh)"
mkdir -p "$OUTDIR"

say "writing the server"
cat > "$SCRIPT" <<'PYEOF'
#!/usr/bin/env python3
"""A JavaScript-free form that drives a local model, section by section."""

import argparse, base64, hmac, html, json, os, re, subprocess, sys, threading
import time, urllib.parse, urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MAX_BYTES = 4 * 1024 * 1024

# Each entry is a heading and what that section is actually for. The
# guidance matters more than the heading: "Market Analysis" alone gets
# generalities, the instruction is what asks for numbers and makes the
# model mark the ones it invented.
PLAN = [
    ("Company Description",
     "What the company does, how it is structured, where it operates, and "
     "why it exists. Concrete, not aspirational."),
    ("Services",
     "The specific services offered, how they are packaged, and how each is "
     "priced. Include a Markdown table of packages with prices."),
    ("Market Analysis",
     "Who the customers are, how many of them there are, and which way "
     "demand is moving. Give figures, and mark any you are estimating "
     "rather than citing."),
    ("Competitive Landscape",
     "Who else serves these customers, what they charge, and where this "
     "company beats them or loses to them. Be honest about the losses."),
    ("Marketing and Sales",
     "How customers are found and closed, the steps of the sales process, "
     "and what acquiring one customer costs."),
    ("Operations",
     "How the work actually gets delivered: process, tooling, "
     "subcontractors, quality control, and what happens when a project "
     "slips."),
    ("Team and Organisation",
     "The roles the business needs, who fills them today, and what has to "
     "be hired and when."),
    ("Financial Plan",
     "Revenue model, cost base, and a three-year projection as a Markdown "
     "table. State every assumption underneath the table."),
    ("Risks and Mitigations",
     "The handful of things most likely to sink this, each with what "
     "reduces it. As a two-column table."),
    ("Roadmap",
     "Milestones for the next 24 months as a table of date, milestone and "
     "how you know it is done."),
]

SITE_COPY = [
    ("Home", "The hero line, a short paragraph under it, and three benefit "
             "blocks. Write the copy itself, not a description of it."),
    ("About", "Who is behind this and why anyone should trust them."),
    ("Services", "Each service as a heading with two or three sentences and "
                 "a price or a starting price."),
    ("Process", "The steps a client goes through, from enquiry to launch."),
    ("Case Studies", "Two or three short case studies. If the brief gives "
                     "none, write them as clearly-labelled placeholders."),
    ("Contact", "The closing pitch and the call to action."),
]

KINDS = {
    "plan": ("Business Plan", PLAN, True),
    "sitecopy": ("Website Copy", SITE_COPY, False),
    "onepager": ("One Pager", [("Summary",
        "The whole business on one page: what it does, for whom, how it "
        "makes money, and what it needs next.")], False),
    "website": ("Website", None, False),      # one shot, raw HTML
    "custom": ("Document", None, False),      # outline supplied by the user
    "ask": ("Answer", None, False),           # one question, answered
}

SYSTEM = (
    "You are writing one section of a formal business document. Output "
    "Markdown only. Do not write the section heading - it is added for "
    "you. Do not introduce yourself, do not explain what you are about to "
    "do, and do not add closing remarks. Prefer concrete detail over "
    "adjectives. Where you use a figure the brief did not give you, mark "
    "it as an assumption in the text."
)

ASK_SYSTEM = (
    "You are advising someone who runs a business. Answer the question put "
    "to you, directly and in full. Do not open with a disclaimer. Do not "
    "substitute 'consult a lawyer' for an answer - they know that, and they "
    "are asking you. Do not hedge a clear answer into uselessness.\n\n"
    "Follow these rules:\n"
    "- Open by naming the jurisdiction your answer is for. If the question "
    "turns on jurisdiction and none was given, say so and name the one you "
    "are assuming.\n"
    "- Name the specific statute, regulation or case where you know it. If "
    "you are not certain of a name, a section number or a date, say you are "
    "not certain. Never invent a citation to sound authoritative - a "
    "plausible wrong reference is worse than no reference.\n"
    "- Keep settled law apart from what is contested or turns on the facts.\n"
    "- Where the answer changes with facts you were not given, say which "
    "facts and which way.\n"
    "- If a document was supplied, answer from its actual wording and quote "
    "the clauses you are relying on.\n"
    "- Finish with what to do next, concretely, and say plainly where a "
    "real solicitor is genuinely needed rather than as a reflex.\n"
    "Write in Markdown."
)

SITE_SYSTEM = (
    "You write complete single-file websites. Output one HTML document and "
    "nothing else: no Markdown fence, no commentary before or after. Put "
    "the CSS in a <style> tag in the head. It must be responsive, work "
    "without JavaScript, and look deliberate rather than templated."
)

CSS = """<style>
body{font:16px/1.65 Georgia,'Times New Roman',serif;max-width:46em;
  margin:2.5em auto;padding:0 1.2em;color:#1a1a1a}
h1{font-size:2em;border-bottom:2px solid #333;padding-bottom:.3em}
h2{font-size:1.4em;margin-top:2em;border-bottom:1px solid #ccc;padding-bottom:.2em}
table{border-collapse:collapse;width:100%;margin:1.2em 0;font-size:.92em}
th,td{border:1px solid #bbb;padding:.5em .7em;text-align:left}
th{background:#f0f0f0}
blockquote{border-left:3px solid #bbb;margin-left:0;padding-left:1em;color:#555}
code{background:#f4f4f4;padding:.1em .3em}
@page{margin:2cm}
</style>"""


def strip_think(text):
    """Reasoning models narrate before they answer; that is not the document."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S | re.I)
    text = re.sub(r"^.*?</think>", "", text, flags=re.S | re.I)
    return text.strip()


def ollama_chat(base, model, system, user, on_chunk, timeout=1800):
    body = json.dumps({
        "model": model, "stream": True,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "options": {"temperature": 0.7, "num_ctx": 16384},
    }).encode()
    req = urllib.request.Request(base + "/api/chat", data=body,
                                 headers={"Content-Type": "application/json"})
    out = []
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for line in r:
            line = line.strip()
            if not line:
                continue
            try:
                m = json.loads(line)
            except Exception:
                continue
            if m.get("error"):
                raise RuntimeError(str(m["error"]))
            piece = (m.get("message") or {}).get("content") or ""
            if piece:
                out.append(piece)
                on_chunk(piece)
            if m.get("done"):
                break
    return "".join(out)


class Job:
    """One document being written, and everything the status page shows."""

    def __init__(self):
        self.lock = threading.Lock()
        self.reset()

    def reset(self):
        self.active = False
        self.title = ""
        self.kind = ""
        self.sections = []          # [(heading, text)] finished
        self.planned = []           # [heading] all of them
        self.current = ""
        self.tail = ""              # the last of what is streaming in
        self.words = 0
        self.error = ""
        self.files = []             # [(label, filename)]
        self.started = 0.0
        self.note = ""

    def snapshot(self):
        with self.lock:
            return dict(active=self.active, title=self.title, kind=self.kind,
                        done=[h for h, _t in self.sections],
                        planned=list(self.planned), current=self.current,
                        tail=self.tail, words=self.words, error=self.error,
                        files=list(self.files), started=self.started,
                        note=self.note)


JOB = Job()


def run(cmd, cwd=None):
    p = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, timeout=300)
    if p.returncode != 0:
        raise RuntimeError("%s failed: %s" % (cmd[0],
                                              p.stdout.decode()[-400:]))


def convert(outdir, stem, md_path, title, want):
    """Markdown in, whichever of docx/pdf/html were asked for out."""
    made = [("Markdown", os.path.basename(md_path))]
    hdr = os.path.join(outdir, "_style.html")
    with open(hdr, "w") as fh:
        fh.write(CSS)
    html_path = os.path.join(outdir, stem + ".html")
    if "html" in want or "pdf" in want:
        run(["pandoc", md_path, "-s", "--toc", "-H", hdr,
             "--metadata", "title=" + title, "-o", html_path])
        if "html" in want:
            made.append(("Web page", os.path.basename(html_path)))
    if "docx" in want:
        d = os.path.join(outdir, stem + ".docx")
        run(["pandoc", md_path, "--toc", "-o", d])
        made.append(("Word", os.path.basename(d)))
    if "pdf" in want:
        from weasyprint import HTML as WHTML
        p = os.path.join(outdir, stem + ".pdf")
        WHTML(html_path).write_pdf(p)
        made.append(("PDF", os.path.basename(p)))
    if "html" not in want and os.path.exists(html_path):
        os.remove(html_path)
    return made


def worker(cfg):
    """Write the document, one section at a time, then convert it."""
    base, model, outdir = cfg["base"], cfg["model"], cfg["outdir"]
    kind, brief, company = cfg["kind"], cfg["brief"], cfg["company"]
    stem = re.sub(r"[^A-Za-z0-9_-]", "_", company or "document")[:40] \
        + time.strftime("_%m%d_%H%M")
    try:
        if kind == "website":
            def chunk(p):
                with JOB.lock:
                    JOB.tail = (JOB.tail + p)[-400:]
                    JOB.words += p.count(" ")
            with JOB.lock:
                JOB.current = "the whole site, in one pass"
            raw = ollama_chat(base, model, SITE_SYSTEM,
                              "Build the site for this business.\n\n" + brief,
                              chunk)
            body = strip_think(raw)
            body = re.sub(r"^```(?:html)?\s*|\s*```$", "", body.strip())
            path = os.path.join(outdir, stem + ".html")
            with open(path, "w") as fh:
                fh.write(body)
            with JOB.lock:
                JOB.files = [("Web page", os.path.basename(path))]
                JOB.current = ""
                JOB.active = False
            return

        if kind == "ask":
            def chunk(p):
                with JOB.lock:
                    JOB.tail = (JOB.tail + p)[-400:]
                    JOB.words += p.count(" ")
            with JOB.lock:
                JOB.current = "thinking it through"
            ask = ""
            if cfg["jurisdiction"]:
                ask += "JURISDICTION\n%s\n\n" % cfg["jurisdiction"]
            if brief:
                ask += "BACKGROUND ON THE BUSINESS\n%s\n\n" % brief
            if cfg["document"]:
                ask += "THE DOCUMENT IN QUESTION\n%s\n\n" % cfg["document"]
            ask += "THE QUESTION\n%s" % cfg["question"]
            answer = strip_think(ollama_chat(base, model, ASK_SYSTEM, ask, chunk))
            with JOB.lock:
                JOB.sections.append((cfg["question"][:70], answer))
                JOB.current = "saving"
            md_path = os.path.join(outdir, stem + ".md")
            with open(md_path, "w") as fh:
                fh.write("---\ntitle: %s\ndate: %s\n---\n\n# Question\n\n%s\n\n"
                         "# Answer\n\n%s\n" % (
                             (cfg["question"][:60] or "Question").replace(":", " -"),
                             time.strftime("%d %B %Y"), cfg["question"], answer))
            try:
                made = convert(outdir, stem, md_path,
                               cfg["question"][:60] or "Answer", cfg["formats"])
                note = ""
            except Exception as e:
                made = [("Markdown", os.path.basename(md_path))]
                note = "the text is fine, but converting it failed: %s" % e
            with JOB.lock:
                JOB.files = made
                JOB.note = note
                JOB.current = ""
                JOB.active = False
            return

        outline = cfg["outline"]
        for heading, guidance in outline:
            with JOB.lock:
                JOB.current = heading
                JOB.tail = ""

            def chunk(p):
                with JOB.lock:
                    JOB.tail = (JOB.tail + p)[-400:]
                    JOB.words += p.count(" ")

            so_far = ""
            with JOB.lock:
                if JOB.sections:
                    # Enough for continuity, not so much that a long
                    # document pushes the brief out of the context window.
                    so_far = "\n\n".join(
                        "## %s\n%s" % (h, t[:600]) for h, t in JOB.sections)
            ask = ("THE BUSINESS\n%s\n\n" % brief)
            if so_far:
                ask += "SECTIONS ALREADY WRITTEN (openings only)\n%s\n\n" % so_far
            ask += ("NOW WRITE THIS SECTION\n%s\n\n%s\n\nLength: %s."
                    % (heading, guidance, cfg["length"]))
            text = strip_think(ollama_chat(base, model, SYSTEM, ask, chunk))
            with JOB.lock:
                JOB.sections.append((heading, text))

        if cfg["exec_summary"]:
            with JOB.lock:
                JOB.current = "Executive Summary"
                JOB.tail = ""
                body = "\n\n".join("## %s\n%s" % (h, t[:900])
                                   for h, t in JOB.sections)

            def chunk2(p):
                with JOB.lock:
                    JOB.tail = (JOB.tail + p)[-400:]
                    JOB.words += p.count(" ")

            # Last, not first: a summary written before the thing it
            # summarises is just the brief again.
            summ = strip_think(ollama_chat(
                base, model, SYSTEM,
                "THE BUSINESS\n%s\n\nTHE PLAN AS WRITTEN\n%s\n\nNOW WRITE "
                "THIS SECTION\nExecutive Summary\n\nOne page. The business, "
                "the market, the model, the money, and the ask. It must "
                "agree with the sections above - do not introduce a figure "
                "that is not in them." % (brief, body), chunk2))
            with JOB.lock:
                JOB.sections.insert(0, ("Executive Summary", summ))

        with JOB.lock:
            JOB.current = "converting"
            title = "%s - %s" % (company or "Untitled", cfg["title"])
            parts = ["---", "title: %s" % title.replace(":", " -"),
                     "date: %s" % time.strftime("%d %B %Y"), "---", ""]
            for h, t in JOB.sections:
                parts.append("# " + h)
                parts.append("")
                parts.append(t)
                parts.append("")
            md = "\n".join(parts)
        md_path = os.path.join(outdir, stem + ".md")
        with open(md_path, "w") as fh:
            fh.write(md)
        try:
            made = convert(outdir, stem, md_path, title, cfg["formats"])
            note = ""
        except Exception as e:
            made = [("Markdown", os.path.basename(md_path))]
            note = "the text is fine, but converting it failed: %s" % e
        with JOB.lock:
            JOB.files = made
            JOB.note = note
            JOB.current = ""
            JOB.active = False
    except Exception as e:
        with JOB.lock:
            JOB.error = "%s: %s" % (type(e).__name__, e)
            JOB.current = ""
            JOB.active = False
PYEOF
echo "  (part 1 written)"

cat >> "$SCRIPT" <<'PYEOF'


PAGE = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Write</title><style>
:root{{color-scheme:dark light}}
body{{font:17px/1.5 -apple-system,system-ui,sans-serif;margin:0;padding:20px 16px;max-width:640px}}
h1{{font-size:22px;margin:0 0 2px}} p.sub{{color:#888;margin:0 0 18px;font-size:15px}}
form{{border:1px solid #8884;border-radius:12px;padding:18px;margin-bottom:24px}}
label{{display:block;font-size:14px;color:#888;margin:14px 0 4px}}
select,input[type=text],textarea{{width:100%;box-sizing:border-box;font-size:17px;
  padding:10px;border-radius:8px;border:1px solid #8886;background:#8881;color:inherit}}
textarea{{min-height:150px}}
button{{font-size:17px;padding:13px;width:100%;margin-top:18px;border:0;
  border-radius:10px;background:#2f6f4f;color:#fff}}
.fmt label{{display:inline-block;color:inherit;font-size:16px;margin:6px 16px 0 0}}
.fmt input{{width:auto;margin-right:5px}}
.note{{color:#888;font-size:14px}} .err{{color:#c55}} .ok{{color:#4a4}}
a.file{{display:block;padding:12px;margin-top:8px;border:1px solid #8884;
  border-radius:9px;text-decoration:none;color:inherit}}
</style></head><body>
<h1>Write a document</h1>
<p class="sub">{status}</p>
{msg}
<p class="note"><a href="/status">watch it being written &rarr;</a></p>
<form method="post" action="/">
  <label>Company name</label>
  <input type="text" name="company" value="{company}" placeholder="Acme Web">

  <label>What kind of document</label>
  <select name="kind">{kinds}</select>

  <label>The brief &mdash; the more you put here, the less it invents</label>
  <textarea name="brief" placeholder="What the business does, who it sells to, what it charges, who else is in the market, what you already know about costs and revenue, where you are based, who is on the team.">{brief}</textarea>

  <label>Your question (for &ldquo;Answer&rdquo;)</label>
  <textarea name="question" style="min-height:80px" placeholder="Who owns the copyright in a site I build for a client if the contract is silent on it?">{question}</textarea>

  <label>Jurisdiction &mdash; name it, or it will guess and not tell you</label>
  <input type="text" name="jurisdiction" value="{jurisdiction}" placeholder="England and Wales">

  <label>A document to answer from (paste a contract, a clause, a letter)</label>
  <textarea name="document" style="min-height:80px">{document}</textarea>

  <label>Your own outline (one heading per line, for &ldquo;custom&rdquo; only)</label>
  <textarea name="outline" style="min-height:80px">{outline}</textarea>

  <label>How long should each section be</label>
  <select name="length">{lengths}</select>

  <label>Formats</label>
  <div class="fmt">
    <label><input type="checkbox" name="fmt_docx" checked> Word</label>
    <label><input type="checkbox" name="fmt_pdf" checked> PDF</label>
    <label><input type="checkbox" name="fmt_html"> Web page</label>
  </div>
  <p class="note">Markdown is always kept. A business plan runs eleven
  sections and takes a while &mdash; you can close the page, it carries on.
  <br><br>
  On &ldquo;Answer&rdquo;: the model is sharp on a document you paste in and
  unreliable recalling law from memory, where it will produce citations that
  look right and are not. Check every section number before you rely on it.</p>
  <button type="submit">Write it</button>
</form>
<h2 style="font-size:18px">Documents</h2>
{files}
</body></html>"""

STATUS = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
{refresh}<title>{head}</title><style>
:root{{color-scheme:dark light}}
body{{font:17px/1.5 -apple-system,system-ui,sans-serif;margin:0;padding:20px 16px;max-width:640px}}
h1{{font-size:22px;margin:0 0 2px}} p.sub{{color:#888;margin:0 0 16px;font-size:15px}}
.bar{{height:10px;border-radius:5px;background:#8883;overflow:hidden;margin:16px 0 6px}}
.bar>i{{display:block;height:100%;background:#2f6f4f}}
ol{{padding-left:1.2em}} li.done{{color:#4a4}} li.now{{font-weight:600}}
li.todo{{color:#888}}
pre{{white-space:pre-wrap;background:#8881;border-radius:9px;padding:12px;
  font-size:14px;color:#888;max-height:12em;overflow:hidden}}
a.btn{{display:block;text-align:center;font-size:17px;padding:13px;margin-top:12px;
  border-radius:10px;background:#2f6f4f;color:#fff;text-decoration:none}}
a.btn.plain{{background:#8883;color:inherit}}
a.file{{display:block;padding:12px;margin-top:8px;border:1px solid #8884;
  border-radius:9px;text-decoration:none;color:inherit}}
.note{{color:#888;font-size:14px}} .err{{color:#c55}} .ok{{color:#4a4}}
</style></head><body>
<h1>{head}</h1>
<p class="sub">{sub}</p>
{body}
<a class="btn plain" href="/">back to the form</a>
</body></html>"""

LENGTHS = [("short", "short - a few paragraphs"),
           ("medium", "medium - half a page"),
           ("long", "long - a full page or more")]


class H(BaseHTTPRequestHandler):
    base = "http://127.0.0.1:11434"
    model = ""
    outdir = ""
    credential = ""
    last = {"company": "", "brief": "", "outline": "", "question": "",
            "jurisdiction": "", "document": ""}

    def log_message(self, f, *a):
        sys.stderr.write("%s %s\n" % (self.address_string(), f % a))

    def authed(self):
        if not self.credential:
            return True
        want = "Basic " + base64.b64encode(self.credential.encode()).decode()
        if hmac.compare_digest(self.headers.get("Authorization", ""), want):
            return True
        self.send_response(401)
        self.send_header("WWW-Authenticate", 'Basic realm="write"')
        self.send_header("Content-Length", "0")
        self.end_headers()
        return False

    def on_disk(self, limit=30):
        out = []
        for f in os.listdir(self.outdir):
            if f.startswith("_") or not f.lower().endswith(
                    (".md", ".docx", ".pdf", ".html")):
                continue
            p = os.path.join(self.outdir, f)
            try:
                out.append((os.path.getmtime(p), f, os.path.getsize(p)))
            except OSError:
                pass
        out.sort(reverse=True)
        return out[:limit]

    def render(self, msg=""):
        kinds = "".join('<option value="{0}">{1}</option>'.format(k, v[0])
                        for k, v in KINDS.items())
        lengths = "".join('<option value="{0}"{2}>{1}</option>'.format(
            k, v, " selected" if k == "medium" else "") for k, v in LENGTHS)
        files = "".join(
            '<a class="file" href="/file?n={q}">{n}<br>'
            '<span class="note">{kb} KB &middot; {when}</span></a>'.format(
                q=urllib.parse.quote(n), n=html.escape(n), kb=sz // 1024 or 1,
                when=time.strftime("%d %b %H:%M", time.localtime(m)))
            for m, n, sz in self.on_disk()) or '<p class="note">none yet</p>'
        body = PAGE.format(
            status=html.escape("%s via %s" % (self.model, self.base)),
            msg=msg, kinds=kinds, lengths=lengths, files=files,
            company=html.escape(self.last["company"]),
            brief=html.escape(self.last["brief"]),
            outline=html.escape(self.last["outline"]),
            question=html.escape(self.last["question"]),
            jurisdiction=html.escape(self.last["jurisdiction"]),
            document=html.escape(self.last["document"])).encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def status_page(self):
        s = JOB.snapshot()
        el = int(time.time() - s["started"]) if s["started"] else 0
        elapsed = "%d:%02d" % (el // 60, el % 60)
        parts = []
        if s["active"]:
            head, refresh = "Writing", '<meta http-equiv="refresh" content="4">'
            n, total = len(s["done"]), max(1, len(s["planned"]))
            parts.append('<div class="bar"><i style="width:{}%"></i></div>'
                         .format(min(100, int(100.0 * n / total))))
            parts.append('<p class="note">{} of {} sections &middot; about {} '
                         'words &middot; {} elapsed</p>'
                         .format(n, total, s["words"], elapsed))
            if s["planned"]:
                lis = []
                for h in s["planned"]:
                    cls = ("done" if h in s["done"]
                           else "now" if h == s["current"] else "todo")
                    lis.append('<li class="{}">{}</li>'.format(cls, html.escape(h)))
                parts.append("<ol>" + "".join(lis) + "</ol>")
            if s["current"]:
                parts.append('<p class="note">writing: {}</p>'
                             .format(html.escape(s["current"])))
            if s["tail"]:
                parts.append("<pre>" + html.escape(s["tail"]) + "</pre>")
        elif s["error"]:
            head, refresh = "Failed", ""
            parts.append('<p class="err">{}</p>'.format(html.escape(s["error"][:600])))
        elif s["files"]:
            head, refresh = "Done", ""
            parts.append('<p class="ok">{} words in {}</p>'
                         .format(s["words"], elapsed))
            if s["note"]:
                parts.append('<p class="err">{}</p>'.format(html.escape(s["note"])))
            for label, fn in s["files"]:
                parts.append('<a class="file" href="/file?n={q}">{l}<br>'
                             '<span class="note">{n}</span></a>'.format(
                                 q=urllib.parse.quote(fn), l=html.escape(label),
                                 n=html.escape(fn)))
        else:
            head, refresh = "Idle", ""
            parts.append('<p class="note">Nothing being written. Start one '
                         'from the form.</p>')
        body = STATUS.format(refresh=refresh, head=head,
                             sub=html.escape(s["title"] or "no document yet"),
                             body="".join(parts)).encode()
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
        if u.path == "/status":
            return self.status_page()
        if u.path == "/file":
            rel = urllib.parse.parse_qs(u.query).get("n", [""])[0]
            p = os.path.abspath(os.path.join(self.outdir, rel))
            if not p.startswith(os.path.abspath(self.outdir) + os.sep) \
               or not os.path.isfile(p):
                self.send_response(404); self.send_header("Content-Length", "0")
                self.end_headers(); return
            ext = os.path.splitext(p)[1].lower()
            mime = {".pdf": "application/pdf", ".html": "text/html",
                    ".md": "text/markdown; charset=utf-8",
                    ".docx": "application/vnd.openxmlformats-officedocument"
                             ".wordprocessingml.document"}.get(ext,
                                                               "application/octet-stream")
            blob = open(p, "rb").read()
            self.send_response(200)
            self.send_header("Content-Type", mime)
            # Word and PDF are for keeping, so hand them over as downloads
            # rather than letting the browser try to render them inline.
            if ext in (".docx", ".pdf", ".md"):
                self.send_header("Content-Disposition",
                                 'attachment; filename="%s"' % os.path.basename(p))
            self.send_header("Content-Length", str(len(blob)))
            self.end_headers(); self.wfile.write(blob); return
        self.render()

    def do_POST(self):
        if not self.authed():
            return
        n = int(self.headers.get("Content-Length") or 0)
        if n <= 0 or n > MAX_BYTES:
            return self.render('<p class="err">nothing submitted</p>')
        fields = {k: v[0] for k, v in urllib.parse.parse_qs(
            self.rfile.read(n).decode("utf-8", "replace")).items()}

        company = fields.get("company", "").strip()
        brief = fields.get("brief", "").strip()
        kind = fields.get("kind", "plan")
        outline_raw = fields.get("outline", "").strip()
        question = fields.get("question", "").strip()
        jurisdiction = fields.get("jurisdiction", "").strip()
        document = fields.get("document", "").strip()
        H.last = {"company": company, "brief": brief, "outline": outline_raw,
                  "question": question, "jurisdiction": jurisdiction,
                  "document": document}

        if kind not in KINDS:
            return self.render('<p class="err">unknown document type</p>')
        if kind == "ask":
            if len(question) < 15:
                return self.render('<p class="err">Type the question you want '
                                   'answered.</p>')
        elif len(brief) < 40:
            return self.render('<p class="err">The brief is the whole input. '
                               'A couple of sentences is not enough to write '
                               'from - it will invent the rest.</p>')
        # Read the flag and let go before rendering: JOB's lock is not
        # reentrant, and anything render() later wants from JOB would
        # hang the request instead of answering it.
        with JOB.lock:
            busy = JOB.active
        if busy:
            return self.render('<p class="err">something is already being '
                               'written. <a href="/status">Watch it</a>, '
                               'or wait for it to finish.</p>')

        title, outline, exec_summary = KINDS[kind]
        if kind == "custom":
            heads = [l.strip() for l in outline_raw.splitlines() if l.strip()]
            if not heads:
                return self.render('<p class="err">"custom" needs an outline: '
                                   'one heading per line.</p>')
            outline = [(h, "Write this section of the document.") for h in heads]
        want = [f for f in ("docx", "pdf", "html")
                if fields.get("fmt_" + f)]
        cfg = dict(base=self.base, model=self.model, outdir=self.outdir,
                   kind=kind, brief=brief, company=company, title=title,
                   question=question, jurisdiction=jurisdiction,
                   document=document,
                   outline=outline or [], exec_summary=exec_summary,
                   length=dict(LENGTHS).get(fields.get("length", "medium"),
                                            "medium"),
                   formats=want)
        with JOB.lock:
            JOB.reset()
            JOB.active = True
            JOB.title = (question[:70] if kind == "ask"
                         else "%s - %s" % (company or "Untitled", title))
            JOB.kind = kind
            JOB.started = time.time()
            JOB.planned = ([h for h, _g in (outline or [])]
                           + (["Executive Summary"] if exec_summary else [])
                           or ["the answer" if kind == "ask" else "the whole site"])
        threading.Thread(target=worker, args=(cfg,), daemon=True).start()
        self.send_response(303)
        self.send_header("Location", "/status")
        self.send_header("Content-Length", "0")
        self.end_headers()


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--port", type=int, default=7863)
    a.add_argument("--ollama", default="http://127.0.0.1:11434")
    a.add_argument("--model", required=True)
    a.add_argument("--outdir", default=os.path.expanduser("~/documents"))
    a.add_argument("--user", default="")
    a.add_argument("--password", default="")
    o = a.parse_args()
    H.base = o.ollama.rstrip("/")
    H.model = o.model
    H.outdir = os.path.abspath(os.path.expanduser(o.outdir))
    os.makedirs(H.outdir, exist_ok=True)
    if not (o.user and o.password):
        print("refusing to start without a password", file=sys.stderr)
        return 2
    H.credential = "{}:{}".format(o.user, o.password)
    for tool, why in (("pandoc", "Word and web pages"),):
        if subprocess.call(["which", tool], stdout=subprocess.DEVNULL) != 0:
            print("  !! no %s - %s will not be produced" % (tool, why),
                  file=sys.stderr)
    try:
        import weasyprint  # noqa: F401
    except Exception:
        print("  !! no weasyprint - PDF will not be produced", file=sys.stderr)
    print("model %s, documents in %s" % (H.model, H.outdir))
    print("serving on 0.0.0.0:%d" % o.port)
    ThreadingHTTPServer(("0.0.0.0", o.port), H).serve_forever()


if __name__ == "__main__":
    sys.exit(main() or 0)
PYEOF

python3 -c "compile(open('$SCRIPT').read(),'$SCRIPT','exec')" || die "embedded server is broken"

if [ ! -s "$CREDS" ]; then
  printf 'user: write\npass: %s\n' "$(openssl rand -base64 15 | tr -d '/+=')" > "$CREDS"
  chmod 600 "$CREDS"
fi
U=$(awk '/^user:/{print $2}' "$CREDS")
P=$(awk '/^pass:/{print $2}' "$CREDS")

say "starting"
tmux kill-session -t =writer 2>/dev/null
tmux new-session -d -s writer \
  "python3 -u '$SCRIPT' --port $PORT --ollama '$OLLAMA' --model '$MODEL' --outdir '$OUTDIR' --user '$U' --password '$P' 2>&1 | tee -a '$LOG'"
sleep 2
tmux has-session -t =writer 2>/dev/null || { tail -20 "$LOG"; die "did not start"; }
CODE=$(curl -sS -o /dev/null -w '%{http_code}' -u "$U:$P" "http://127.0.0.1:$PORT/" 2>/dev/null)

cat <<EOF

=== done ===

  port        $PORT   (http $CODE)
  model       $MODEL
  documents   $OUTDIR
  log         tail -n 40 $LOG
  stop        tmux kill-session -t =writer

$(cat "$CREDS")

Forward port $PORT in the Thunder console and open it.

Fill the brief properly - it is the whole input, and whatever you leave
out the model will invent and then build on. Then /status shows each
section landing as it is written; you can close the page, it carries on.
EOF
