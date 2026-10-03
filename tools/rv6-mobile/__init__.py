# Serves web/edit.html at /extensions/rv6_mobile/edit.html, on the same
# origin as the ComfyUI API, and adds POST /rv6m/segment: tap-to-select
# masks from Segment Anything (SAM) for the edit page. Adds no nodes.
import asyncio
import io
import logging
import math
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
# Pro takes a resolution tier plus an aspect ratio instead of Lite's exact "W*H" size
RATIOS = ["1:1", "4:3", "3:4", "3:2", "2:3", "16:9", "9:16"]
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
        mime = "image/jpeg" if path.lower().endswith((".jpg", ".jpeg")) else "image/png"
        images.append(f"data:{mime};base64," + base64.b64encode(open(path, "rb").read()).decode())
    body = {"model": model, "prompt": prompt, "images": images, "enable_base64_output": False}
    if "pro" in model.lower():
        w, h = (int(v) for v in size.split("*"))
        body["resolution"] = "2k"
        body["aspect_ratio"] = min(RATIOS, key=lambda r: abs(math.log(w / h) - math.log(int(r.split(":")[0]) / int(r.split(":")[1]))))
    else:
        body["size"] = size
    sub = _atlas("POST", ATLAS + "/generateImage", key, body)
    pid = _find(_data(sub), ["id", "prediction_id", "request_id"])
    if not pid:
        raise RuntimeError("Atlas reply had no job id: " + str(sub)[:600])
    t = time.time()
    while True:
        res = _data(_atlas("GET", f"{ATLAS}/prediction/{pid}", key))
        status = str(_find(res, ["status"]) or "").lower()
        if status in ("completed", "succeeded", "success", "done"):
            break
        if status in ("failed", "error", "canceled", "cancelled"):
            raise RuntimeError("Atlas job failed: " + str(_find(res, ["error", "message"]) or res)[:600])
        if time.time() - t > 300:
            raise RuntimeError("Atlas took over 5 minutes; last reply: " + str(res)[:300])
        time.sleep(1.5)
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
    return name


_jobs = {}  # job id -> {"status": "running"|"done"|"error", "filename", "error", "seconds"}


async def _run_cloud(jid, names, prompt, size, model):
    loop = asyncio.get_running_loop()
    t = loop.time()
    try:
        name = await loop.run_in_executor(None, _cloud, names, prompt, size, model)
        _jobs[jid].update(status="done", filename=name)
    except Exception as e:
        logging.exception("[rv6_mobile] cloud edit failed")
        _jobs[jid].update(status="error", error=str(e))
    _jobs[jid]["seconds"] = round(loop.time() - t, 1)


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
