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


# Upscaled pictures are 4K-8K PNGs (30-100 MB): too slow for a phone over
# Tailscale. GET /rv6m/jpeg?filename=...&subfolder=... gives the same output
# picture as a high-quality JPEG (a few MB), made once and kept in temp.
# &max=600 gives a small one for the Recent edits thumbnails.
def _jpeg(src, size):
    from PIL import Image
    st = os.stat(src)
    out = os.path.join(folder_paths.get_temp_directory(), "rv6m_jpeg",
                       f"{os.path.basename(src)}.{int(st.st_mtime)}.{st.st_size}.{size}.jpg")
    if not os.path.exists(out):
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with Image.open(src) as im:
            im = im.convert("RGB")
            if size:
                im.thumbnail((size, size), Image.LANCZOS)
            im.save(out + ".part", "JPEG", quality=95 if not size else 85, subsampling=0 if not size else 2)
        os.replace(out + ".part", out)
    return out


@PromptServer.instance.routes.get("/rv6m/jpeg")
async def rv6m_jpeg(request):
    root = os.path.realpath(folder_paths.get_output_directory())
    src = os.path.realpath(os.path.join(root, request.query.get("subfolder", ""), request.query.get("filename", "")))
    if not src.startswith(root + os.sep) or not os.path.isfile(src):
        return web.json_response({"error": "no such output picture"}, status=404)
    try:
        size = max(64, min(4096, int(request.query["max"]))) if request.query.get("max", "").isdigit() else 0
        out = await asyncio.get_running_loop().run_in_executor(None, _jpeg, src, size)
    except Exception as e:
        logging.exception("[rv6_mobile] jpeg failed")
        return web.json_response({"error": f"{type(e).__name__}: {e}"}, status=500)
    return web.FileResponse(out, headers={"Content-Type": "image/jpeg", "Cache-Control": "max-age=86400"})


# --- Remember my work, kept on the VM ----------------------------------------
# The page keeps the chosen photos, extra photos and selection in the phone's
# IndexedDB so they survive Safari reloading the tab. iPhone Lockdown Mode
# turns IndexedDB off (iOS 17+), so then the page stores the same things here:
# POST /rv6m/work (multipart: key, meta JSON, files b0, b1...), GET
# /rv6m/work?key=..., GET /rv6m/work/blob?key=...&name=b0. Only these keys, in
# ~/.rv6m_work/<key>/; a new save replaces the old one.
WORK_DIR = os.path.expanduser("~/.rv6m_work")
WORK_KEYS = {"photo", "mask", "more", "batch"}


@PromptServer.instance.routes.post("/rv6m/work")
async def rv6m_work_set(request):
    import json
    import re
    import shutil
    import uuid
    form = await request.post()
    key = str(form.get("key", ""))
    if key not in WORK_KEYS:
        return web.json_response({"error": "unknown key"}, status=400)
    meta = json.loads(str(form.get("meta", "null")))
    final = os.path.join(WORK_DIR, key)
    tmp = os.path.join(WORK_DIR, f".{key}.{uuid.uuid4().hex}")
    os.makedirs(tmp)
    types = {}
    for name, field in form.items():
        if re.fullmatch(r"b\d{1,3}", name) and hasattr(field, "file"):
            with open(os.path.join(tmp, name), "wb") as f:
                shutil.copyfileobj(field.file, f)
            types[name] = field.content_type or "application/octet-stream"
    with open(os.path.join(tmp, "meta.json"), "w") as f:
        json.dump({"meta": meta, "types": types}, f)
    old = final + ".old"
    shutil.rmtree(old, ignore_errors=True)
    if os.path.exists(final):
        os.replace(final, old)
    os.replace(tmp, final)
    shutil.rmtree(old, ignore_errors=True)
    logging.info("[rv6_mobile] saved %s for the page (%d file(s), %d KB)", key, len(types),
                 sum(os.path.getsize(os.path.join(final, n)) for n in types) // 1024)
    return web.json_response({"ok": True})


@PromptServer.instance.routes.get("/rv6m/work")
async def rv6m_work_get(request):
    import json
    key = request.query.get("key", "")
    path = os.path.join(WORK_DIR, key, "meta.json")
    if key not in WORK_KEYS or not os.path.exists(path):
        return web.json_response({"meta": None})
    with open(path) as f:
        return web.json_response({"meta": json.load(f)["meta"]})


@PromptServer.instance.routes.get("/rv6m/work/blob")
async def rv6m_work_blob(request):
    import json
    import re
    key, name = request.query.get("key", ""), request.query.get("name", "")
    if key not in WORK_KEYS or not re.fullmatch(r"b\d{1,3}", name) or not os.path.exists(os.path.join(WORK_DIR, key, name)):
        return web.json_response({"error": "not found"}, status=404)
    with open(os.path.join(WORK_DIR, key, "meta.json")) as f:
        ctype = json.load(f)["types"].get(name, "application/octet-stream")
    return web.FileResponse(os.path.join(WORK_DIR, key, name), headers={"Content-Type": ctype, "Cache-Control": "no-store"})


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


PRO_SIZES = ["2048*2048", "2368*1776", "1776*2368", "2816*1584", "1584*2816", "2496*1664", "1664*2496", "3136*1344"]


PRO_ALL = PRO_SIZES + ["1024*1024", "1536*1536", "1776*1328", "1328*1776", "2048*1152", "1152*2048"]


def _pro_size(size):
    """size as is when it is one of Seedream 5.0 Pro's presets, else the 2K preset closest in shape."""
    import math
    if str(size) in PRO_ALL:
        return str(size)
    try:
        w, h = (int(x) for x in str(size).lower().replace("x", "*").split("*"))
        r = math.log(w / h)
    except Exception:
        return "2048*2048"
    return min(PRO_SIZES, key=lambda p: abs(math.log(int(p.split("*")[0]) / int(p.split("*")[1])) - r))


SEEDREAM_THINKING = os.environ.get("RV6M_SEEDREAM_THINKING", "disabled")      # enabled | disabled
SEEDREAM_OPTIMIZE = os.environ.get("RV6M_SEEDREAM_OPTIMIZE", "fast")          # standard | fast


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
        # Seedream 5.0 Pro (Atlas's API reference): size must be one of its presets
        # (there is no "resolution" field; anything else falls back to a square
        # 2048*2048). Pick the 2K preset closest to picture 1's shape. Prompt
        # rewriting: thinking off and the "fast" mode (your choice; the playground
        # uses enabled + standard). Change with RV6M_SEEDREAM_THINKING / _OPTIMIZE.
        body.update(size=_pro_size(size), output_format="jpeg", thinking=SEEDREAM_THINKING,
                    prompt_optimization_mode=SEEDREAM_OPTIMIZE)
    else:
        body["size"] = size
    sent = {k: v for k, v in body.items() if k != "images"}
    sent["images"] = [f"{n} ({len(i) * 3 // 4 // 1024} KB)" for n, i in zip(names, images)]
    data = _atlas_job("/generateImage", key, body, sent, 300)
    ext = ".jpg" if data[:3] == b"\xff\xd8\xff" else ".webp" if data[8:12] == b"WEBP" else ".png"
    name = f"rv6_cloud_{int(time.time() * 1000)}{ext}"
    with open(os.path.join(folder_paths.get_output_directory(), name), "wb") as f:
        f.write(data)
    return {"filename": name, "sent": sent}


def _atlas_job(endpoint, key, body, sent, limit, meta=None):
    """Submit to Atlas, poll until done (up to limit seconds) and return the result file's bytes.
    meta (a dict), if given, gets the prediction id and any draft_id from Atlas's reply."""
    import base64
    import time
    import urllib.request
    log = {"sent": sent}
    try:
        sub = _atlas("POST", ATLAS + endpoint, key, body)
        log["submit_reply"] = sub
    except Exception as e:
        log["error"] = str(e)
        # Say which step and which model, so a 404 "not found" can be told apart
        raise RuntimeError(f"sending the job (model {body.get('model')}): {e}") from None
    finally:
        _save_log(log)
    pid = _find(_data(sub), ["id", "prediction_id", "request_id"])
    if not pid:
        raise RuntimeError("Atlas reply had no job id: " + str(sub)[:600])
    t = time.time()
    while True:
        try:
            res = _data(_atlas("GET", f"{ATLAS}/prediction/{pid}", key))
        except RuntimeError as e:
            # A job that was just created can briefly be "not found"; keep asking for a minute
            if "said 404" in str(e) and time.time() - t < 60:
                log["early_404"] = str(e)
                time.sleep(3)
                continue
            log["error"] = str(e)
            _save_log(log)
            raise RuntimeError(f"checking on job {pid}: {e}") from None
        log["last_reply"] = res
        status = str(_find(res, ["status"]) or "").lower()
        if status in ("completed", "succeeded", "success", "done"):
            break
        if status in ("failed", "error", "canceled", "cancelled"):
            _save_log(log)
            raise RuntimeError("Atlas job failed: " + str(_find(res, ["error", "message"]) or res)[:600])
        if time.time() - t > limit:
            raise RuntimeError(f"Atlas took over {limit // 60} minutes; last reply: " + str(res)[:300])
        time.sleep(1.5)
    _save_log(log)
    if meta is not None:
        meta.update(id=pid, draft_id=_find(res, ["draft_id"]) or _find(_data(sub), ["draft_id"]))
    out = _find(res, ["outputs", "output", "images", "videos", "video", "url"])
    url = out[0] if isinstance(out, list) else out
    if isinstance(url, dict):
        url = _find(url, ["url"])
    if not isinstance(url, str):
        raise RuntimeError("Atlas reply had no result file: " + str(res)[:600])
    if url.startswith("http"):
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=300) as r:
            return r.read()
    return base64.b64decode(url.split(",", 1)[-1])


# ☁️ Seedance on Atlas (the page's 🎬 Video, "Seedance" engine), two kinds:
#   image-to-video: picture 1 is the first frame, picture 2 (optional) the last;
#     the video keeps the photo's shape
#   reference-to-video: up to 30 photos as references (people, clothes, places),
#     cited in the prompt as @Image1, @Image2...; a new shot, any shape
# 4-30 s, optional sound. Model ids can be changed with RV6M_CLOUD_VIDEO_MODEL
# and RV6M_CLOUD_REF_MODEL.
CLOUD_VIDEO_MODEL = os.environ.get("RV6M_CLOUD_VIDEO_MODEL", "bytedance/seedance-2.5/image-to-video")
CLOUD_REF_MODEL = os.environ.get("RV6M_CLOUD_REF_MODEL", "bytedance/seedance-2.5/reference-to-video")
# From Atlas's Seedance 2.5 API references (the same for both kinds). 480p/720p/1080p are made
# natively; -sr (FlashVSR) and -esr (Atlas Video Enhance) make the nearest
# native size and then upscale it (4k-esr: from 1080p to 3840x2160 at 16:9).
REF_RESOLUTIONS = ["480p", "720p", "720p-sr", "720p-esr", "1080p", "1080p-sr", "1080p-esr", "1080p-esr & 60fps",
                   "1440p-sr", "1440p-esr", "4k-esr"]
REF_RATIOS = ["adaptive", "16:9", "4:3", "1:1", "3:4", "9:16", "21:9"]


def _ref_image(path):
    """A photo as Seedance takes it (both kinds): sides 300-6000 px, shape (W/H) 0.4-2.5, under 30 MB."""
    import base64
    from PIL import Image
    with Image.open(path) as im:
        w, h = im.size
        if not 0.4 <= w / h <= 2.5:
            raise RuntimeError(f"{os.path.basename(path)} is {w}x{h}: Seedance only takes photos between 2.5:1 and 1:2.5. Crop it first.")
        if min(w, h) < 300:
            s = 300 / min(w, h)
            buf = io.BytesIO()
            im.convert("RGB").resize((round(w * s), round(h * s)), Image.LANCZOS).save(buf, "JPEG", quality=95)
            return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    return _data_uri(path)


def _data_uri(path):
    import base64
    mime = "image/png" if path.lower().endswith(".png") else "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(open(path, "rb").read()).decode()


# 📝 Drafts: a Seedance video made with draft=true is a 480p preview (billed as
# 480p). Atlas's bytedance/seedance-2.5/draft-complete renders the same video at
# 1080p from just its draft_id, within 7 days. The VM remembers which saved
# video came from which draft: ~/.rv6m_drafts.json
CLOUD_FINISH_MODEL = os.environ.get("RV6M_CLOUD_FINISH_MODEL", "bytedance/seedance-2.5/draft-complete")
DRAFTS_FILE = os.path.expanduser("~/.rv6m_drafts.json")
DRAFT_DAYS = 7
_drafts_lock = threading.Lock()


def _drafts():
    import json
    import time
    try:
        with open(DRAFTS_FILE) as f:
            d = json.load(f)
    except Exception:
        return {}
    return {k: v for k, v in d.items() if time.time() - v.get("t", 0) < DRAFT_DAYS * 86400}


def _draft_add(filename, draft_id):
    import json
    import time
    with _drafts_lock:
        d = _drafts()
        d[filename] = {"draft_id": str(draft_id), "t": time.time()}
        with open(DRAFTS_FILE + ".tmp", "w") as f:
            json.dump(d, f)
        os.replace(DRAFTS_FILE + ".tmp", DRAFTS_FILE)


def _save_video(data):
    import time
    ext = ".mov" if data[4:12] == b"ftypqt  " else ".mp4"
    out = f"rv6_cloud_{int(time.time() * 1000)}{ext}"
    with open(os.path.join(folder_paths.get_output_directory(), out), "wb") as f:
        f.write(data)
    return out


def _cloud_finish(filename):
    key = _cloud_key()
    if not key:
        raise RuntimeError("No Atlas API key on the VM (~/.atlascloud_key)")
    d = _drafts().get(filename)
    if not d:
        raise RuntimeError("That video isn't a draft from the last 7 days")
    body = {"model": CLOUD_FINISH_MODEL, "draft_id": d["draft_id"], "output_format": "mp4", "watermark": False}
    out = _save_video(_atlas_job("/generateVideo", key, body, {**body, "draft_of": filename}, 2400))
    return {"filename": out, "sent": {"model": CLOUD_FINISH_MODEL, "resolution": "1080p", "finished": filename}}


def _cloud_video(names, prompt, model, duration, resolution, audio, ratio="adaptive", draft=False):
    key = _cloud_key()
    if not key:
        raise RuntimeError("No Atlas API key on the VM (~/.atlascloud_key)")
    ref = "reference" in model
    images = [_ref_image(folder_paths.get_annotated_filepath(n)) for n in names]
    body = {"model": model, "prompt": prompt, "duration": int(duration), "resolution": resolution,
            "generate_audio": bool(audio), "watermark": False}
    if ref:
        # omni_reference_task_type left unset (= auto), as Atlas recommends
        body.update(reference_images=images, ratio=ratio, output_format="mp4")
    else:
        # image-to-video only takes "adaptive": the video keeps the photo's shape
        body.update(image=images[0], ratio="adaptive", output_format="mp4")
        if len(images) > 1:
            body["last_image"] = images[1]
    if draft:
        body["draft"] = True  # 480p preview; Atlas ignores resolution
    sent = {k: v for k, v in body.items() if k not in ("image", "last_image", "reference_images")}
    sent["images"] = [f"{n} ({len(i) * 3 // 4 // 1024} KB)" for n, i in zip(names, images)]
    if draft:
        sent["resolution"] = "480p draft"
    meta = {}
    # Upscaled (-sr/-esr) and long videos can take a while
    out = _save_video(_atlas_job("/generateVideo", key, body, sent, 2400, meta))
    if draft:
        # Atlas: the draft_id is the draft's prediction id (also returned as draft_id)
        _draft_add(out, meta.get("draft_id") or meta["id"])
    return {"filename": out, "sent": sent, "draft": bool(draft)}


_jobs = {}  # job id -> {"status": "running"|"done"|"error", "filename", "error", "seconds"}


async def _run_cloud(jid, fn, *args):
    loop = asyncio.get_running_loop()
    t = loop.time()
    try:
        out = await loop.run_in_executor(None, fn, *args)
        _jobs[jid].update(status="done", **out)
    except Exception as e:
        logging.exception("[rv6_mobile] cloud edit failed")
        _jobs[jid].update(status="error", error=str(e))
    _jobs[jid]["seconds"] = round(loop.time() - t, 1)
    _finished(_jobs[jid]["status"] == "done", _jobs[jid]["seconds"], _jobs[jid].get("error", ""))


@PromptServer.instance.routes.get("/rv6m/cloud/status")
async def rv6m_cloud_status(request):
    return web.json_response({"configured": bool(_cloud_key()), "model": CLOUD_MODEL, "video_model": CLOUD_VIDEO_MODEL,
                              "ref_model": CLOUD_REF_MODEL, "ref_resolutions": REF_RESOLUTIONS})


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
    asyncio.ensure_future(_run_cloud(jid, _cloud, names, body["prompt"], str(body.get("size") or "2048*2048"),
                                     body.get("model") or CLOUD_MODEL))
    return web.json_response({"job": jid})


@PromptServer.instance.routes.post("/rv6m/cloud/video")
async def rv6m_cloud_video(request):
    import uuid
    body = await request.json()
    ref = body.get("kind") == "reference"
    names = [n for n in (body.get("images") or [body.get("image")]) if n][:30 if ref else 2]  # first (+ last) frame
    if not names or not str(body.get("prompt", "")).strip():
        return web.json_response({"error": "need a photo and a prompt"}, status=400)
    if not _cloud_key():
        return web.json_response({"error": "No Atlas API key on the VM (~/.atlascloud_key)"}, status=400)
    res = str(body.get("resolution") or "720p")
    if res not in REF_RESOLUTIONS:
        return web.json_response({"error": f"Seedance doesn't make {res} videos"}, status=400)
    ratio = str(body.get("ratio") or "adaptive")
    if ratio not in REF_RATIOS:
        ratio = "adaptive"
    duration = int(body.get("duration") or 5)
    duration = -1 if duration == -1 else min(30, max(4, duration))
    jid = uuid.uuid4().hex
    _jobs[jid] = {"status": "running"}
    asyncio.ensure_future(_run_cloud(jid, _cloud_video, names, body["prompt"],
                                     body.get("model") or (CLOUD_REF_MODEL if ref else CLOUD_VIDEO_MODEL),
                                     duration, res, body.get("audio", True), ratio, bool(body.get("draft"))))
    return web.json_response({"job": jid})


@PromptServer.instance.routes.get("/rv6m/drafts")
async def rv6m_drafts(request):
    import time
    return web.json_response({k: {"days_left": round(DRAFT_DAYS - (time.time() - v["t"]) / 86400, 1)} for k, v in _drafts().items()})


@PromptServer.instance.routes.post("/rv6m/cloud/finish")
async def rv6m_cloud_finish(request):
    import uuid
    body = await request.json()
    name = os.path.basename(str(body.get("filename") or ""))
    if name not in _drafts():
        return web.json_response({"error": "That video isn't a draft from the last 7 days"}, status=400)
    if not _cloud_key():
        return web.json_response({"error": "No Atlas API key on the VM (~/.atlascloud_key)"}, status=400)
    jid = uuid.uuid4().hex
    _jobs[jid] = {"status": "running"}
    asyncio.ensure_future(_run_cloud(jid, _cloud_finish, name))
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
