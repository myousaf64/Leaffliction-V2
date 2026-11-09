#!/usr/bin/env python3
"""Leaffliction v2 backend.

POST /predict        one image  -> top-3, Grad-CAM overlay, inference ms
POST /predict-batch  many images -> the same per image, plus a summary
"""

from __future__ import annotations

import base64
import io
import time
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image, UnidentifiedImageError

from pipeline import (CAM_CMAP, IMGSZ, ROOT, UNCERTAIN_BELOW, _crop224, gradcam,
                      load_model, predict)

MAX_BYTES = 10 * 1024 * 1024
MAX_FILES = 32
MAX_TOTAL_BYTES = 64 * 1024 * 1024
STATE: dict = {}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    import json

    model, names, device = load_model()
    cam_layer = "head"
    cam_json = ROOT / "assets" / "cam" / "cam.json"
    if cam_json.exists():
        cam_layer = json.loads(cam_json.read_text()).get("target_layer", "head")

    meta = None
    metrics = ROOT / "metrics.json"
    if metrics.exists():
        m = json.loads(metrics.read_text())
        meta = {
            "accuracy": m["test"]["accuracy"],
            "n": m["test"]["n"],
            "gpu_ms": m["latency_ms"]["gpu"],
            "cpu_ms": m["latency_ms"]["cpu"],
            "size_mb": m["model"]["size_mb"],
            "classes": len(m["classes"]),
        }
    STATE.update(model=model, names=names, device=device, cam_layer=cam_layer, meta=meta)
    # Warm up once. The first forward pays for kernel autotuning (~170 ms), and a demo
    # that reports that as its inference time is lying about the model.
    warm = Image.new("RGB", (IMGSZ, IMGSZ), (90, 130, 70))
    predict(model, warm, device)
    gradcam(model, warm, device, class_idx=0, which=cam_layer)
    print(f"model ready on {device}, {len(names)} classes, cam layer {cam_layer}")
    yield
    STATE.clear()


app = FastAPI(title="Leaffliction", lifespan=lifespan)


def _data_url(img: Image.Image, mime: str, **opts) -> str:
    buf = io.BytesIO()
    img.save(buf, format=mime.split("/")[1].upper(), **opts)
    return f"data:{mime};base64," + base64.b64encode(buf.getvalue()).decode()


def _photo_url(img: Image.Image) -> str:
    """The leaf crop as JPEG: 0.3 ms and 26 KB, against 19 ms and 132 KB as a PNG.

    A 32-image batch sends 32 of these, so the difference is the batch feeling instant
    or not. The heatmap cannot follow - it needs the alpha channel.
    """
    return _data_url(img, "image/jpeg", quality=88)


def _png(img: Image.Image) -> str:
    return _data_url(img, "image/png", compress_level=1)


def _heat_png(heat: np.ndarray) -> Image.Image:
    """Heat as standalone RGBA: colour from CAM_CMAP, alpha from intensity."""
    rgba = (CAM_CMAP(heat) * 255).astype(np.uint8)
    rgba[..., 3] = (np.clip(heat, 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(rgba, mode="RGBA")


@app.get("/")
def index():
    return FileResponse(ROOT / "index.html")


@app.get("/meta")
def meta():
    """Measured headline numbers for the intro cover.

    404 when metrics.json is absent, so the page omits the tiles rather than inventing them.
    """
    if not STATE.get("meta"):
        raise HTTPException(404, "No metrics.json. Run: uv run python pipeline.py eval")
    return JSONResponse(STATE["meta"])


def _analyse(raw: bytes) -> dict:
    """Guard the bytes, then classify and explain them.

    Raises HTTPException(400) with a message meant for a person to read.
    """
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

    return {
        "top3": top3,
        "healthy": names[order[0]].endswith("_healthy"),
        "uncertain": bool(probs[order[0]] < UNCERTAIN_BELOW),
        "inference_ms": round(ms, 2),
        "device": device,
        "imgsz": IMGSZ,
        "image_png": _photo_url(_crop224(img)),
        "heatmap_png": _png(_heat_png(heat)),
    }


@app.post("/predict")
async def predict_route(file: UploadFile = File(...)):
    return JSONResponse(_analyse(await file.read()))


@app.post("/predict-batch")
async def predict_batch(files: list[UploadFile] = File(...)):
    if not files:
        raise HTTPException(400, "No files. Choose at least one image.")
    if len(files) > MAX_FILES:
        raise HTTPException(
            400, f"{len(files)} files. The limit is {MAX_FILES} images at a time.")

    started = time.perf_counter()
    results, budget = [], MAX_TOTAL_BYTES
    for upload in files:
        raw = await upload.read()
        budget -= len(raw)
        if budget < 0:
            raise HTTPException(
                400, f"The batch is over {MAX_TOTAL_BYTES // 1024 // 1024} MB. Send fewer photos.")
        name = upload.filename or "photo"
        try:
            # One unreadable photo must not sink the other 31.
            results.append({"ok": True, "name": name, **_analyse(raw)})
        except HTTPException as exc:
            results.append({"ok": False, "name": name, "error": exc.detail})

    done = [r for r in results if r["ok"]]
    times = [r["inference_ms"] for r in done]
    return JSONResponse({
        "results": results,
        "summary": {
            "n": len(results),
            "ok": len(done),
            "failed": len(results) - len(done),
            "healthy": sum(r["healthy"] and not r["uncertain"] for r in done),
            "diseased": sum(not r["healthy"] and not r["uncertain"] for r in done),
            "uncertain": sum(r["uncertain"] for r in done),
            "total_ms": round((time.perf_counter() - started) * 1000, 1),
            "mean_ms": round(sum(times) / len(times), 2) if times else None,
        },
        "device": STATE["device"],
    })
