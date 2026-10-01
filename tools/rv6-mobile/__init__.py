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
