# Serves web/edit.html at /extensions/rv6_mobile/edit.html, on the same
# origin as the ComfyUI API, and adds POST /rv6m/segment: tap-to-select
# masks from Segment Anything (SAM) for the edit page. Adds no nodes.
import asyncio
import io
import logging
import os
import threading

from aiohttp import web

import folder_paths
from server import PromptServer

NODE_CLASS_MAPPINGS = {}
WEB_DIRECTORY = "./web"

SAM_ID = os.environ.get("RV6M_SAM_MODEL", "facebook/sam-vit-huge")
_lock = threading.Lock()
_sam = {}          # "model", "processor", "device"
_embeds = {}       # image name -> (embeddings, original_sizes, reshaped_input_sizes); last few only


def _load():
    if not _sam:
        import torch
        from transformers import SamModel, SamProcessor
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info("[rv6_mobile] loading %s on %s", SAM_ID, device)
        _sam["processor"] = SamProcessor.from_pretrained(SAM_ID)
        _sam["model"] = SamModel.from_pretrained(SAM_ID).to(device).eval()
        _sam["device"] = device
    return _sam["model"], _sam["processor"], _sam["device"]


def _segment(name, points, labels, grow):
    import numpy as np
    import torch
    from PIL import Image

    with _lock:
        model, processor, device = _load()
        path = folder_paths.get_annotated_filepath(name)
        image = Image.open(path).convert("RGB")
        pts = [[[float(x), float(y)] for x, y in points]]
        lbl = [[int(v) for v in labels]]
        with torch.no_grad():
            if name not in _embeds:
                inputs = processor(image, return_tensors="pt").to(device)
                emb = model.get_image_embeddings(inputs["pixel_values"])
                _embeds.clear()  # one image at a time is all the page needs
                _embeds[name] = (emb, inputs["original_sizes"], inputs["reshaped_input_sizes"])
            emb, orig, reshaped = _embeds[name]
            pin = processor(image, input_points=pts, input_labels=lbl, return_tensors="pt").to(device)
            out = model(input_points=pin["input_points"], input_labels=pin["input_labels"],
                        image_embeddings=emb, multimask_output=True)
            masks = processor.image_processor.post_process_masks(out.pred_masks.cpu(), orig.cpu(), reshaped.cpu())
            scores = out.iou_scores[0, 0]
            m = masks[0][0, int(scores.argmax())].float()
            if grow > 0:
                k = 2 * grow + 1
                m = torch.nn.functional.max_pool2d(m[None, None], k, stride=1, padding=grow)[0, 0]
        alpha = (m.numpy() > 0.5).astype(np.uint8) * 255
        rgba = np.zeros((*alpha.shape, 4), np.uint8)
        rgba[..., 0] = 255
        rgba[..., 3] = alpha
        buf = io.BytesIO()
        Image.fromarray(rgba, "RGBA").save(buf, "PNG")
        return buf.getvalue()


@PromptServer.instance.routes.post("/rv6m/segment")
async def rv6m_segment(request):
    try:
        body = await request.json()
        points, labels = body["points"], body["labels"]
        if not points or len(points) != len(labels):
            return web.json_response({"error": "need at least one point"}, status=400)
        png = await asyncio.get_running_loop().run_in_executor(
            None, _segment, body["image"], points, labels, int(body.get("grow", 8)))
        return web.Response(body=png, content_type="image/png")
    except Exception as e:
        logging.exception("[rv6_mobile] segment failed")
        return web.json_response({"error": f"{type(e).__name__}: {e}"}, status=500)


# --- Seedream (Atlas Cloud) -------------------------------------------------
# POST /rv6m/cloud sends the photos in ComfyUI's input folder plus the prompt
# to Atlas Cloud, waits for the result and saves it in the output folder so it
# shows up like a local edit. The API key stays on this VM: ~/.atlascloud_key
# or the ATLASCLOUD_API_KEY environment variable. It never reaches the page.
ATLAS = "https://api.atlascloud.ai/api/v1/model"
CLOUD_MODEL = os.environ.get("RV6M_CLOUD_MODEL", "bytedance/seedream-v5.0-pro/edit")
# Pro takes a resolution tier instead of Lite's exact "W*H" size
KEY_FILE = os.path.expanduser("~/.atlascloud_key")
# Atlas sits behind Cloudflare, which refuses Python's default "Python-urllib"
# user agent with "error code: 1010"; send an ordinary one instead.
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36 rv6-mobile"


def _cloud_key():
    key = os.environ.get("ATLASCLOUD_API_KEY", "")
    if not key and os.path.exists(KEY_FILE):
        key = open(KEY_FILE).read().strip()
    return key


def _find(obj, keys):
    """First value under any of keys, searched depth-first (Atlas may nest it in "data")."""
    if isinstance(obj, dict):
        for k in keys:
            if obj.get(k) not in (None, "", []):
                return obj[k]
        for v in obj.values():
            r = _find(v, keys)
            if r is not None:
                return r
    elif isinstance(obj, list):
        for v in obj:
            r = _find(v, keys)
            if r is not None:
                return r
    return None


def _data(reply):
    """Atlas wraps results as {"code": ..., "data": {...}}; look there first."""
    return reply["data"] if isinstance(reply, dict) and isinstance(reply.get("data"), dict) else reply


def _atlas(method, url, key, body=None):
    import json
    import urllib.error
    import urllib.request
    req = urllib.request.Request(url, method=method, data=json.dumps(body).encode() if body else None,
                                 headers={"Authorization": "Bearer " + key, "Content-Type": "application/json",
                                          "Accept": "application/json", "User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"Atlas said {e.code}: {e.read().decode(errors='replace')[:600]}")


LOG_FILE = os.path.expanduser("~/rv6m_cloud_last.json")


def _save_log(log):
    """The last cloud request (without the photo data) and Atlas's replies, for troubleshooting."""
    import json
    try:
        with open(LOG_FILE, "w") as f:
            json.dump(log, f, indent=1, default=str)
    except OSError:
        pass


def _cloud(names, prompt, size, model):
    import base64
    import time
    import urllib.request
    key = _cloud_key()
    if not key:
        raise RuntimeError("No Atlas API key on the VM (~/.atlascloud_key)")
    images = []
    for n in names:
        path = folder_paths.get_annotated_filepath(n)
        low = path.lower()
        mime = "image/png" if low.endswith(".png") else "image/webp" if low.endswith(".webp") else "image/jpeg"
        images.append(f"data:{mime};base64," + base64.b64encode(open(path, "rb").read()).decode())
    body = {"model": model, "prompt": prompt, "images": images, "enable_base64_output": False}
    if "pro" in model.lower():
        # No aspect_ratio: like Atlas's playground, let the model keep picture 1's shape
        body["resolution"] = "2k"
    else:
        body["size"] = size
    sent = {k: v for k, v in body.items() if k != "images"}
    sent["images"] = [f"{n} ({len(i) * 3 // 4 // 1024} KB)" for n, i in zip(names, images)]
    log = {"sent": sent}
    try:
        sub = _atlas("POST", ATLAS + "/generateImage", key, body)
        log["submit_reply"] = sub
    except Exception as e:
        log["error"] = str(e)
        raise
    finally:
        _save_log(log)
    pid = _find(_data(sub), ["id", "prediction_id", "request_id"])
    if not pid:
        raise RuntimeError("Atlas reply had no job id: " + str(sub)[:600])
    t = time.time()
    while True:
        res = _data(_atlas("GET", f"{ATLAS}/prediction/{pid}", key))
        log["last_reply"] = res
        status = str(_find(res, ["status"]) or "").lower()
        if status in ("completed", "succeeded", "success", "done"):
            break
        if status in ("failed", "error", "canceled", "cancelled"):
            _save_log(log)
            raise RuntimeError("Atlas job failed: " + str(_find(res, ["error", "message"]) or res)[:600])
        if time.time() - t > 300:
            raise RuntimeError("Atlas took over 5 minutes; last reply: " + str(res)[:300])
        time.sleep(1.5)
    _save_log(log)
    out = _find(res, ["outputs", "output", "images", "url"])
    url = out[0] if isinstance(out, list) else out
    if isinstance(url, dict):
        url = _find(url, ["url"])
    if not isinstance(url, str):
        raise RuntimeError("Atlas reply had no image: " + str(res)[:600])
    if url.startswith("http"):
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=120) as r:
            data = r.read()
    else:
        data = base64.b64decode(url.split(",", 1)[-1])
    ext = ".jpg" if data[:3] == b"\xff\xd8\xff" else ".webp" if data[8:12] == b"WEBP" else ".png"
    name = f"rv6_cloud_{int(time.time() * 1000)}{ext}"
    with open(os.path.join(folder_paths.get_output_directory(), name), "wb") as f:
        f.write(data)
    return {"filename": name, "sent": sent}


_jobs = {}  # job id -> {"status": "running"|"done"|"error", "filename", "error", "seconds"}


async def _run_cloud(jid, names, prompt, size, model):
    loop = asyncio.get_running_loop()
    t = loop.time()
    try:
        out = await loop.run_in_executor(None, _cloud, names, prompt, size, model)
        _jobs[jid].update(status="done", **out)
    except Exception as e:
        logging.exception("[rv6_mobile] cloud edit failed")
        _jobs[jid].update(status="error", error=str(e))
    _jobs[jid]["seconds"] = round(loop.time() - t, 1)
    _finished(_jobs[jid]["status"] == "done", _jobs[jid]["seconds"], _jobs[jid].get("error", ""))


@PromptServer.instance.routes.get("/rv6m/cloud/status")
async def rv6m_cloud_status(request):
    return web.json_response({"configured": bool(_cloud_key()), "model": CLOUD_MODEL})


# Starts the edit in the background and returns a job id right away, so the
# phone can switch apps (iOS drops long requests) and ask for it again later.
@PromptServer.instance.routes.post("/rv6m/cloud")
async def rv6m_cloud(request):
    import uuid
    body = await request.json()
    names = [n for n in body.get("images", []) if n]
    if not names or not str(body.get("prompt", "")).strip():
        return web.json_response({"error": "need a photo and a prompt"}, status=400)
    if not _cloud_key():
        return web.json_response({"error": "No Atlas API key on the VM (~/.atlascloud_key)"}, status=400)
    jid = uuid.uuid4().hex
    _jobs[jid] = {"status": "running"}
    asyncio.ensure_future(_run_cloud(jid, names, body["prompt"], str(body.get("size") or "2048*2048"),
                                     body.get("model") or CLOUD_MODEL))
    return web.json_response({"job": jid})


@PromptServer.instance.routes.get("/rv6m/cloud/job/{jid}")
async def rv6m_cloud_job(request):
    job = _jobs.get(request.match_info["jid"])
    if job is None:
        return web.json_response({"status": "error", "error": "unknown job (ComfyUI was restarted?)"}, status=404)
    return web.json_response(job)


# ---- "Notify me when done": a push to the ntfy app on the phone when the
# page's edits finish, so you can leave the page while the VM works. Safari
# can't run in the background on iOS, so the VM sends it. The page says when it
# is open and visible; nothing is sent then. Messages only say done/failed.
import json
import secrets
import time
import urllib.request

NOTIFY_FILE = os.path.expanduser("~/.rv6m_notify.json")
_note = {"seen": 0.0, "pending": [], "known": set()}


def _notify_cfg():
    try:
        with open(NOTIFY_FILE) as f:
            return json.load(f)
    except Exception:
        return {}


def _notify_save(cfg):
    with open(NOTIFY_FILE, "w") as f:
        json.dump(cfg, f)
    os.chmod(NOTIFY_FILE, 0o600)


def _push(title, body, tags):
    cfg = _notify_cfg()
    if not cfg.get("topic"):
        raise RuntimeError("notifications are not set up")
    headers = {"Title": title, "Tags": tags, "User-Agent": "rv6-mobile"}
    if cfg.get("click"):
        headers["Click"] = cfg["click"]
    req = urllib.request.Request(cfg.get("server", "https://ntfy.sh").rstrip("/") + "/" + cfg["topic"],
                                 data=body.encode(), headers=headers, method="POST")
    urllib.request.urlopen(req, timeout=15).read()


def _finished(ok, seconds, error=""):
    _note["pending"].append((ok, seconds, str(error)[:200], time.time()))


def _is_ours(prompt):
    return any(isinstance(n, dict) and n.get("class_type") in ("SaveImage", "SaveVideo")
               and str(n.get("inputs", {}).get("filename_prefix", "")).startswith("rv6_edit")
               for n in (prompt or {}).values())


def _watch():
    first = True
    while True:
        time.sleep(2)
        try:
            q = getattr(PromptServer.instance, "prompt_queue", None)
            if q is None:
                continue
            hist = q.get_history(max_items=40)
            for pid, h in hist.items():
                if pid in _note["known"]:
                    continue
                _note["known"].add(pid)
                if first:
                    continue  # finished before this ComfyUI started watching
                pr = h.get("prompt") or ()
                if len(pr) < 3 or not _is_ours(pr[2]):
                    continue
                st = h.get("status") or {}
                msgs = {m[0]: (m[1] if len(m) > 1 else {}) for m in st.get("messages", [])}
                if "execution_interrupted" in msgs:
                    continue  # Stop was pressed
                ok = st.get("status_str") == "success"
                t0, t1 = msgs.get("execution_start", {}).get("timestamp"), \
                    (msgs.get("execution_success") or msgs.get("execution_error") or {}).get("timestamp")
                secs = (t1 - t0) / 1000 if t0 and t1 else 0
                _finished(ok, secs, "" if ok else msgs.get("execution_error", {}).get("exception_message", ""))
            first = False
            if len(_note["known"]) > 2000:
                _note["known"] = set(hist)
            p = _note["pending"]
            # One message per batch: wait until nothing else is queued (or 2 min)
            if p and (q.get_tasks_remaining() == 0 or time.time() - p[0][3] > 120):
                _note["pending"] = []
                cfg = _notify_cfg()
                if not cfg.get("enabled") or not cfg.get("topic") or time.time() - _note["seen"] < 8:
                    continue  # off, or the page is open and showing it
                good = [x for x in p if x[0]]
                bad = [x for x in p if not x[0]]
                if good:
                    n = len(good)
                    _push("Photo edit ready", (f"{n} images are" if n > 1 else "Your image is") +
                          " ready. Tap to open.", "white_check_mark")
                if bad:
                    _push("Photo edit failed", "An edit failed: " + (bad[0][2] or "see the page") , "x")
        except Exception:
            logging.exception("[rv6_mobile] notify watcher")
            time.sleep(10)


threading.Thread(target=_watch, daemon=True, name="rv6m-notify").start()


@PromptServer.instance.routes.post("/rv6m/seen")
async def rv6m_seen(request):
    try:
        body = await request.json()
    except Exception:
        body = {}
    _note["seen"] = time.time() if body.get("visible", True) else 0.0
    return web.json_response({"ok": True})


@PromptServer.instance.routes.get("/rv6m/notify")
async def rv6m_notify_status(request):
    cfg = _notify_cfg()
    return web.json_response({"enabled": bool(cfg.get("enabled")), "topic": cfg.get("topic", ""),
                              "server": cfg.get("server", "https://ntfy.sh")})


@PromptServer.instance.routes.post("/rv6m/notify")
async def rv6m_notify(request):
    body = await request.json()
    cfg, action = _notify_cfg(), body.get("action")
    if action in ("on", "off", "new"):
        if action == "new" or not cfg.get("topic"):
            cfg["topic"] = "photoedit-" + secrets.token_hex(10)  # hard to guess = private
        cfg.setdefault("server", "https://ntfy.sh")
        cfg["enabled"] = action != "off"
        if body.get("click"):
            cfg["click"] = str(body["click"])[:300]
        _notify_save(cfg)
    elif action == "test":
        try:
            await asyncio.get_running_loop().run_in_executor(
                None, _push, "Photo edit test", "Notifications work. You'll get one when an edit is ready.", "bell")
        except Exception as e:
            return web.json_response({"error": f"Could not send: {e}"}, status=502)
    else:
        return web.json_response({"error": "unknown action"}, status=400)
    return web.json_response({"enabled": bool(cfg.get("enabled")), "topic": cfg.get("topic", ""),
                              "server": cfg.get("server", "https://ntfy.sh")})
