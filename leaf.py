#!/usr/bin/env python3
"""One inference path for the leaf classifier.

app.py and predict.py both import this module. Neither one loads the model
or installs a hook itself.

Two traps this module handles:

1. model.predict() runs inside torch.no_grad(). A hook fires there but no
   gradient arrives, so Grad-CAM gets nothing. This module calls the raw
   nn.Module instead.
2. The Classify head returns softmax scores in eval mode. A gradient taken
   from a score near 1.0 is almost zero and the heatmap turns to noise. This
   module takes the gradient from the linear layer output instead, before the
   softmax.
"""

import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF
from ultralytics import YOLO
from ultralytics.data.augment import classify_transforms

IMGSZ = 224
CAM_LAYER = "model.10.conv.act"
BATCH_CAP = 64
ALPHA = 0.4

PRETTY = {
    "Apple_Black_rot": "Apple - Black rot",
    "Apple_healthy": "Apple - Healthy",
    "Apple_rust": "Apple - Rust",
    "Apple_scab": "Apple - Scab",
    "Grape_Black_rot": "Grape - Black rot",
    "Grape_Esca": "Grape - Esca",
    "Grape_healthy": "Grape - Healthy",
    "Grape_spot": "Grape - Leaf spot",
}

_state = {}


def pretty(name):
    """Return the plain-words name of one class."""
    return PRETTY.get(name, name.replace("_", " "))


def is_healthy(name):
    """Return True if the class name means a healthy leaf."""
    return name.lower().endswith("healthy")


def weights_path():
    """Return the path of the trained weights."""
    override = os.environ.get("LEAF_WEIGHTS")
    if override:
        return Path(override)
    found = sorted(
        Path("runs/classify").glob("train*/weights/best.pt"),
        key=lambda p: p.stat().st_mtime,
    )
    if not found:
        sys.exit("no weights found. Run this first:\n\n    .venv/bin/python train.py\n")
    return found[-1]


def load(verbose=True):
    """Load the model once, install the hooks, and warm up the GPU."""
    if _state:
        return _state

    device = "cuda" if torch.cuda.is_available() else "cpu"
    wrapper = YOLO(str(weights_path()))
    net = wrapper.model.to(device).eval()
    for param in net.parameters():
        param.requires_grad_(True)

    modules = dict(net.named_modules())
    if CAM_LAYER not in modules:
        sys.exit(f"layer {CAM_LAYER} is not in this model. Modules: {list(modules)[-8:]}")
    linear_name, linear = [
        (n, m) for n, m in net.named_modules() if isinstance(m, torch.nn.Linear)
    ][-1]

    cache = {}

    def on_activation(_module, _inputs, output):
        cache["act"] = output
        if output.requires_grad:
            output.register_hook(lambda grad: cache.__setitem__("grad", grad))

    def on_logits(_module, _inputs, output):
        cache["logits"] = output

    modules[CAM_LAYER].register_forward_hook(on_activation)
    linear.register_forward_hook(on_logits)

    _state.update(
        net=net,
        device=device,
        cache=cache,
        names=wrapper.names,
        transform=classify_transforms(IMGSZ),
    )

    noise = np.random.randint(0, 255, (IMGSZ + 40, IMGSZ + 60, 3), dtype=np.uint8)
    warm = Image.fromarray(noise)
    answer(warm)
    batch_answer([warm])

    if verbose:
        print(f"weights:   {weights_path()}")
        print(f"device:    {device}")
        print(f"cam layer: {CAM_LAYER}  (gradient from {linear_name})")
    return _state


def to_pil(image):
    """Return image as an RGB PIL image. Accept a path, an array, or a PIL image."""
    if isinstance(image, (str, Path)):
        return Image.open(image).convert("RGB")
    if isinstance(image, np.ndarray):
        return Image.fromarray(image).convert("RGB")
    return image.convert("RGB")


def model_view(pil):
    """Return the 224x224 crop the model sees, as an RGB array.

    The transform resizes the short side and then crops the centre. The
    heatmap must sit on this crop. On the full photo it lands misaligned.
    """
    return np.array(TF.center_crop(TF.resize(pil, IMGSZ), [IMGSZ, IMGSZ]))


def _heatmap(cache, height, width):
    """Return the Grad-CAM heatmap, scaled to 0...1."""
    act = cache["act"][0].detach().float()
    grad = cache["grad"][0].detach().float()
    weights = grad.mean(dim=(1, 2))
    cam = torch.relu((weights[:, None, None] * act).sum(0)).cpu().numpy()
    cam -= cam.min()
    if cam.max() > 0:
        cam /= cam.max()
    return cv2.resize(cam, (width, height))


def _blend(view, heat):
    """Return the heatmap drawn over the crop."""
    colour = cv2.applyColorMap((heat * 255).astype("uint8"), cv2.COLORMAP_JET)
    colour = cv2.cvtColor(colour, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(view, 1 - ALPHA, colour, ALPHA, 0)


def answer(image):
    """Classify one image and build its heatmap. Return every fact about it."""
    state = load()
    pil = to_pil(image)
    view = model_view(pil)
    tensor = state["transform"](pil).unsqueeze(0).to(state["device"])

    start = time.perf_counter()
    state["net"].zero_grad(set_to_none=True)
    state["cache"].clear()
    state["net"](tensor)
    logits = state["cache"]["logits"]
    index = int(logits.argmax(1))
    logits[0, index].backward()
    scores = logits.detach().softmax(1)[0]
    heat = _heatmap(state["cache"], view.shape[0], view.shape[1])
    if state["device"] == "cuda":
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) * 1000

    names = state["names"]
    return {
        "name": names[index],
        "pretty": pretty(names[index]),
        "confidence": float(scores[index]),
        "scores": {pretty(names[i]): float(scores[i]) for i in range(len(scores))},
        "ms": elapsed,
        "view": view,
        "overlay": _blend(view, heat),
    }


def batch_answer(images):
    """Classify a list of images in one forward for each chunk.

    Return the verdicts and the total milliseconds. This path has no heatmap,
    because speed is the point of it.
    """
    state = load()
    rows = []
    start = time.perf_counter()
    with torch.no_grad():
        for first in range(0, len(images), BATCH_CAP):
            chunk = images[first:first + BATCH_CAP]
            tensor = torch.stack(
                [state["transform"](to_pil(image)) for image in chunk]
            ).to(state["device"])
            scores = state["net"](tensor).detach()
            for row in scores:
                index = int(row.argmax())
                name = state["names"][index]
                rows.append({
                    "name": name,
                    "pretty": pretty(name),
                    "confidence": float(row[index]),
                    "healthy": is_healthy(name),
                })
    if state["device"] == "cuda":
        torch.cuda.synchronize()
    return rows, (time.perf_counter() - start) * 1000


def side_by_side(result):
    """Return the crop and the heatmap in one image."""
    return np.hstack([result["view"], result["overlay"]])
