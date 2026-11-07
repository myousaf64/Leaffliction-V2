#!/usr/bin/env python3
"""Leaffliction v2 backend. POST /predict -> top-3, Grad-CAM overlay, inference ms."""

from __future__ import annotations

import base64
import io
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image, UnidentifiedImageError

from pipeline import (CAM_CMAP, IMGSZ, ROOT, UNCERTAIN_BELOW, _crop224, gradcam,
                      load_model, predict)

MAX_BYTES = 10 * 1024 * 1024
STATE: dict = {}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    model, names, device = load_model()
    cam_layer = "head"
    cam_json = ROOT / "assets" / "cam" / "cam.json"
    if cam_json.exists():
        import json
        cam_layer = json.loads(cam_json.read_text()).get("target_layer", "head")
    STATE.update(model=model, names=names, device=device, cam_layer=cam_layer)
    # Warm up once. The first forward pays for kernel autotuning (~170 ms), and a demo
    # that reports that as its inference time is lying about the model.
    warm = Image.new("RGB", (IMGSZ, IMGSZ), (90, 130, 70))
    predict(model, warm, device)
    gradcam(model, warm, device, class_idx=0, which=cam_layer)
    print(f"model ready on {device}, {len(names)} classes, cam layer {cam_layer}")
    yield
    STATE.clear()


app = FastAPI(title="Leaffliction", lifespan=lifespan)


def _png(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _heat_png(heat: np.ndarray) -> Image.Image:
    """Heat as standalone RGBA: colour from CAM_CMAP, alpha from intensity."""
    rgba = (CAM_CMAP(heat) * 255).astype(np.uint8)
    rgba[..., 3] = (np.clip(heat, 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(rgba, mode="RGBA")


@app.get("/")
def index():
    return FileResponse(ROOT / "index.html")


@app.post("/predict")
async def predict_route(file: UploadFile = File(...)):
    raw = await file.read()
    if not raw:
        raise HTTPException(400, "Empty upload. Choose an image file.")
    if len(raw) > MAX_BYTES:
        raise HTTPException(400, f"File is {len(raw) / 1e6:.1f} MB. The limit is 10 MB.")
    try:
        img = Image.open(io.BytesIO(raw))
        img.load()                      # forces a decode, so a truncated JPEG fails here
        img = img.convert("RGB")
    except UnidentifiedImageError as exc:
        raise HTTPException(400, "That file is not an image. Use a JPEG or PNG photo.") from exc
    except (OSError, ValueError) as exc:
        raise HTTPException(400, f"The image is damaged or incomplete: {exc}.") from exc
    if min(img.size) < 32:
        raise HTTPException(400, f"Image is {img.size[0]}x{img.size[1]}. The minimum is 32x32.")

    model, names, device = STATE["model"], STATE["names"], STATE["device"]
    probs, ms = predict(model, img, device)
    order = probs.argsort()[::-1][:3]
    top3 = [{"label": names[i], "confidence": round(float(probs[i]), 4)} for i in order]
    heat = gradcam(model, img, device, class_idx=int(order[0]), which=STATE["cam_layer"])

    return JSONResponse({
        "top3": top3,
        "healthy": names[order[0]].endswith("_healthy"),
        "uncertain": bool(probs[order[0]] < UNCERTAIN_BELOW),
        "inference_ms": round(ms, 2),
        "device": device,
        "imgsz": IMGSZ,
        "image_png": _png(_crop224(img)),
        "heatmap_png": _png(_heat_png(heat)),
    })
