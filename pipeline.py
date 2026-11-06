#!/usr/bin/env python3
"""Leaffliction v2 pipeline.

Subcommands:
    split   stratified 70/15/15 of images/ into dataset/, then offline balance of train only
    train   fine-tune yolo11n-cls
    eval    test split only -> metrics.json
    cam     Grad-CAM target-layer probe + 6 example overlays
    export  ONNX export

Also the shared module: load_model, preprocess, predict, gradcam, leaf_mask, CAM_CMAP.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image, ImageEnhance

ROOT = Path(__file__).resolve().parent
IMAGES = ROOT / "images"
DATASET = ROOT / "dataset"
RUNS = ROOT / "runs"
WEIGHTS = RUNS / "cls" / "weights" / "best.pt"
METRICS = ROOT / "metrics.json"
SPLITS_REPORT = ROOT / "splits_report.txt"
ASSETS = ROOT / "assets"

SEED = 42
IMGSZ = 224
SPLIT_FRACTIONS = (0.70, 0.15, 0.15)
BALANCE_CAP = 1200
UNCERTAIN_BELOW = 0.40

# images/ still carries augmented copies left over from Leaffliction v1, named
# "image (100)_Rotate.JPG" beside their parent "image (100).JPG". A copy and its parent
# must share a split, so the split is stratified over these groups, not over single files.
V1_OPS = ("_Crop", "_Distortion", "_Flip", "_Rotate", "_Shear", "_Skew")


def group_key(stem: str) -> str:
    """Parent stem of a v1 augmented copy, else the stem itself."""
    for op in V1_OPS:
        if stem.endswith(op):
            return stem[:-len(op)]
    return stem

# Design tokens. Grad-CAM colormap is ivory -> lime -> terracotta, never jet.
FOREST = "#173F35"
LIME = "#A8CF45"
SAGE = "#E9F0E6"
IVORY = "#F8F7F0"
TERRACOTTA = "#D77B50"
TERRACOTTA_TEXT = "#A8522E"
CHARCOAL = "#26332E"
CAM_CMAP = LinearSegmentedColormap.from_list("leaf", [IVORY, LIME, TERRACOTTA])


# --------------------------------------------------------------------------- #
# shared model helpers
# --------------------------------------------------------------------------- #

class _Logits(nn.Module):
    """Ultralytics Classify returns (probs, logits) in eval mode. Grad-CAM needs logits."""

    def __init__(self, inner: nn.Module):
        super().__init__()
        self.inner = inner

    def forward(self, x):
        y = self.inner(x)
        return y[1] if isinstance(y, (tuple, list)) else y


def load_model(weights: Path = WEIGHTS, device: str | None = None):
    """Return (logits_module, names, device). names is an index-ordered list."""
    from ultralytics import YOLO

    if not Path(weights).exists():
        sys.exit(f"missing weights: {weights}\nrun: python pipeline.py train")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    yolo = YOLO(str(weights))
    names = yolo.names
    names = [names[i] for i in range(len(names))] if isinstance(names, dict) else list(names)
    core = yolo.model.float().to(device).eval()
    for p in core.parameters():
        p.requires_grad_(True)
    return _Logits(core), names, device


def preprocess(img: Image.Image) -> torch.Tensor:
    """Match ultralytics classify_transforms(224): resize, center crop, /255, no mean/std shift."""
    w, h = img.size
    scale = IMGSZ / min(w, h)
    img = img.convert("RGB").resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.BILINEAR)
    w, h = img.size
    left, top = (w - IMGSZ) // 2, (h - IMGSZ) // 2
    img = img.crop((left, top, left + IMGSZ, top + IMGSZ))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def predict(model, img: Image.Image, device: str) -> tuple[np.ndarray, float]:
    """Return (probabilities, inference milliseconds)."""
    x = preprocess(img).to(device)
    with torch.no_grad():
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        logits = model(x)
        if device == "cuda":
            torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000.0
    return logits.softmax(1)[0].cpu().numpy(), ms


def target_layer(model, which: str):
    """Candidate Grad-CAM layers. 'head' is the 1280-ch conv feeding the GAP; 'backbone' is the block before it."""
    seq = model.inner.model
    return seq[-1].conv if which == "head" else seq[-2]


def gradcam(model, img: Image.Image, device: str, class_idx: int | None = None,
            which: str = "head") -> np.ndarray:
    """Return a 224x224 float heatmap in [0, 1]."""
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    x = preprocess(img).to(device)
    if class_idx is None:
        with torch.no_grad():
            class_idx = int(model(x).argmax(1).item())
    with GradCAM(model=model, target_layers=[target_layer(model, which)]) as cam:
        grid = cam(input_tensor=x, targets=[ClassifierOutputTarget(class_idx)])[0]
    span = float(grid.max() - grid.min())
    return (grid - grid.min()) / span if span > 1e-8 else np.zeros_like(grid)


def leaf_mask(img: Image.Image) -> np.ndarray:
    """Leaf-vs-background boolean mask, 224x224. HSV threshold, largest component, holes filled."""
    from scipy import ndimage

    hsv = np.asarray(img.convert("RGB").resize((IMGSZ, IMGSZ), Image.BILINEAR).convert("HSV"))
    sat, val = hsv[..., 1].astype(np.int16), hsv[..., 2].astype(np.int16)
    raw = (sat > 40) & (val > 25)
    if raw.sum() < 0.02 * raw.size:  # threshold found nothing usable
        return np.ones_like(raw, dtype=bool)
    labels, n = ndimage.label(raw)  # 4-connected by default
    if n > 1:
        sizes = ndimage.sum(raw, labels, range(1, n + 1))
        raw = labels == (int(np.argmax(sizes)) + 1)
    return ndimage.binary_fill_holes(raw)


def overlay(img: Image.Image, heat: np.ndarray, alpha: float = 0.68) -> Image.Image:
    """Blend the heatmap over the 224x224 crop using CAM_CMAP."""
    base = np.asarray(_crop224(img), dtype=np.float32) / 255.0
    rgba = CAM_CMAP(heat)[..., :3].astype(np.float32)
    w = (alpha * heat)[..., None]
    return Image.fromarray((np.clip(base * (1 - w) + rgba * w, 0, 1) * 255).astype(np.uint8))


def _crop224(img: Image.Image) -> Image.Image:
    t = preprocess(img)[0].permute(1, 2, 0).numpy()
    return Image.fromarray((t * 255).astype(np.uint8))


# --------------------------------------------------------------------------- #
# step 1: split
# --------------------------------------------------------------------------- #

def _augment(img: Image.Image, rng: random.Random) -> Image.Image:
    """At least one geometric op, so a copy is never a brightness-only near-duplicate."""
    hflip, vflip = rng.random() < 0.5, rng.random() < 0.5
    rot = rng.choice([0, 90, 180, 270])
    if not (hflip or vflip or rot):
        rot = rng.choice([90, 180, 270])
    if hflip:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    if vflip:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
    if rot:
        img = img.rotate(rot)
    img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.8, 1.2))
    return ImageEnhance.Contrast(img).enhance(rng.uniform(0.8, 1.2))


def cmd_split(_args) -> None:
    if not IMAGES.is_dir():
        sys.exit(f"missing dataset directory: {IMAGES}")
    rng = random.Random(SEED)
    classes = sorted(d.name for d in IMAGES.iterdir() if d.is_dir())
    if not classes:
        sys.exit(f"no class folders in {IMAGES}")

    if DATASET.exists():
        shutil.rmtree(DATASET)

    before: dict[str, dict[str, int]] = {}
    stems: dict[str, str] = {}
    for cls in classes:
        files = sorted(p for p in (IMAGES / cls).iterdir() if p.is_file())
        groups: dict[str, list[Path]] = {}
        for src in files:
            groups.setdefault(group_key(src.stem), []).append(src)
        keys = sorted(groups)
        rng.shuffle(keys)

        n = len(files)
        want_train = int(n * SPLIT_FRACTIONS[0])
        want_val = int(n * SPLIT_FRACTIONS[1])
        parts: dict[str, list[Path]] = {"train": [], "val": [], "test": []}
        for key in keys:  # whole group at a time, so no copy is split off its parent
            if len(parts["train"]) < want_train:
                parts["train"] += groups[key]
            elif len(parts["val"]) < want_val:
                parts["val"] += groups[key]
            else:
                parts["test"] += groups[key]

        before[cls] = {k: len(v) for k, v in parts.items()}
        for split, group in parts.items():
            out = DATASET / split / cls
            out.mkdir(parents=True, exist_ok=True)
            for src in group:
                key = f"{cls}/{group_key(src.stem)}"
                if stems.get(key, split) != split:
                    sys.exit(f"leak: group {key} in both {stems[key]} and {split}")
                stems[key] = split
                (out / src.name).symlink_to(src.resolve())
        print(f"{cls:<22} train {len(parts['train']):>5}  val {len(parts['val']):>4}  test {len(parts['test']):>4}")

    # offline balance, train split only
    target = min(BALANCE_CAP, max(v["train"] for v in before.values()))
    after: dict[str, int] = {}
    for cls in classes:
        out = DATASET / "train" / cls
        originals = sorted(p for p in out.iterdir())
        need = target - len(originals)
        made = 0
        while made < need:
            src = originals[made % len(originals)]
            with Image.open(src) as im:
                _augment(im.convert("RGB"), rng).save(out / f"{src.stem}_aug{made}.JPG", quality=95)
            made += 1
        after[cls] = len(originals) + made
        print(f"{cls:<22} balanced {before[cls]['train']:>5} -> {after[cls]:>5}  (+{made})")

    _verify_split()
    _write_splits_report(classes, before, after, target)
    print(f"\nwrote {SPLITS_REPORT}")


def _verify_split() -> None:
    seen: dict[str, str] = {}
    originals = 0
    for split in ("train", "val", "test"):
        for path in (DATASET / split).rglob("*.JPG"):
            if "_aug" in path.stem:
                assert split == "train", f"augmented copy outside train: {path}"
                continue
            originals += 1
            key = f"{path.parent.name}/{group_key(path.stem)}"
            assert seen.get(key, split) == split, \
                f"leak: group {key} in {seen[key]} and {split}"
            seen[key] = split
    print(f"\nleak check ok: {originals} originals in {len(seen)} parent groups, "
          "every group confined to one split")


def _write_splits_report(classes, before, after, target) -> None:
    lines = [
        "Leaffliction v2 - splits report",
        f"source          : {IMAGES}",
        f"seed            : {SEED}",
        f"fractions       : train {SPLIT_FRACTIONS[0]:.2f} / val {SPLIT_FRACTIONS[1]:.2f} / test {SPLIT_FRACTIONS[2]:.2f}",
        "split order     : split first, augment second. Augmented copies exist in train only.",
        "grouping        : v1 leftover copies (image (N)_Rotate.JPG) share a split with their",
        "                  parent image (N).JPG. The split is stratified over parent groups.",
        f"balance target  : {target} per train class (cap {BALANCE_CAP})",
        "",
        f"{'class':<22}{'total':>8}{'train':>8}{'val':>7}{'test':>7}{'train_aug':>11}{'added':>8}",
        "-" * 71,
    ]
    tot = {"total": 0, "train": 0, "val": 0, "test": 0, "train_aug": 0}
    for cls in classes:
        b = before[cls]
        total = b["train"] + b["val"] + b["test"]
        lines.append(
            f"{cls:<22}{total:>8}{b['train']:>8}{b['val']:>7}{b['test']:>7}"
            f"{after[cls]:>11}{after[cls] - b['train']:>8}"
        )
        tot["total"] += total
        for k in ("train", "val", "test"):
            tot[k] += b[k]
        tot["train_aug"] += after[cls]
    lines += [
        "-" * 71,
        f"{'TOTAL':<22}{tot['total']:>8}{tot['train']:>8}{tot['val']:>7}{tot['test']:>7}"
        f"{tot['train_aug']:>11}{tot['train_aug'] - tot['train']:>8}",
        "",
        "Augmentation ops: horizontal flip, vertical flip, rotate 90/180/270,",
        "brightness 0.8-1.2, contrast 0.8-1.2. Every copy gets at least one geometric op.",
    ]
    SPLITS_REPORT.write_text("\n".join(lines) + "\n")


# --------------------------------------------------------------------------- #
# step 2: train
# --------------------------------------------------------------------------- #

def cmd_train(args) -> None:
    from ultralytics import YOLO

    if not (DATASET / "train").is_dir():
        sys.exit("missing dataset/. run: python pipeline.py split")
    for batch in (args.batch, args.batch // 2, args.batch // 4):
        print(f"\n=== train attempt, batch={batch} ===")
        try:
            YOLO("yolo11n-cls.pt").train(
                data=str(DATASET), imgsz=IMGSZ, batch=batch, epochs=args.epochs,
                patience=10, amp=True, workers=4, seed=SEED, deterministic=False,
                project=str(RUNS), name="cls", exist_ok=True, plots=True,
            )
            print(f"\ntrained at batch={batch}, weights at {WEIGHTS}")
            return
        except torch.cuda.OutOfMemoryError:
            print(f"CUDA OOM at batch={batch}, halving")
            torch.cuda.empty_cache()
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            print(f"CUDA OOM at batch={batch}, halving")
            torch.cuda.empty_cache()
    sys.exit("OOM at batch=8. Lower imgsz or free VRAM.")


# --------------------------------------------------------------------------- #
# step 3: eval
# --------------------------------------------------------------------------- #

def _test_files(names: list[str]) -> list[tuple[Path, int]]:
    index = {n: i for i, n in enumerate(names)}
    items = []
    for cls_dir in sorted((DATASET / "test").iterdir()):
        if cls_dir.name not in index:
            sys.exit(f"test class {cls_dir.name!r} is not in the model's names: {names}")
        items += [(p, index[cls_dir.name]) for p in sorted(cls_dir.iterdir()) if p.is_file()]
    return items


def _latency(model, device: str, sample: Path, warmup: int = 10, runs: int = 100) -> float:
    with Image.open(sample) as im:
        x = preprocess(im).to(device)
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(runs):
            model(x)
        if device == "cuda":
            torch.cuda.synchronize()
        return (time.perf_counter() - t0) * 1000.0 / runs


def cmd_eval(_args) -> None:
    model, names, device = load_model()
    if device != "cuda":
        print("warning: CUDA unavailable, gpu latency will be null")
    items = _test_files(names)
    k = len(names)
    confusion = np.zeros((k, k), dtype=int)
    top5_hits = 0

    with torch.no_grad():
        for i in range(0, len(items), 64):
            chunk = items[i:i + 64]
            batch = torch.cat([preprocess(Image.open(p)) for p, _ in chunk]).to(device)
            probs = model(batch).softmax(1).cpu().numpy()
            for (_, truth), row in zip(chunk, probs):
                confusion[truth, int(row.argmax())] += 1
                top5_hits += int(truth in row.argsort()[-5:])
            print(f"\reval {min(i + 64, len(items))}/{len(items)}", end="", flush=True)
    print()

    correct = int(np.trace(confusion))
    n = int(confusion.sum())
    per_class = {}
    for i, name in enumerate(names):
        tp = int(confusion[i, i])
        predicted = int(confusion[:, i].sum())
        support = int(confusion[i, :].sum())
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[name] = {
            "precision": round(precision, 4), "recall": round(recall, 4),
            "f1": round(f1, 4), "support": support,
        }

    gpu_ms = round(_latency(model, "cuda", items[0][0]), 3) if device == "cuda" else None
    cpu_model = _Logits(model.inner.cpu())
    cpu_ms = round(_latency(cpu_model, "cpu", items[0][0]), 3)

    params = sum(p.numel() for p in model.inner.parameters())
    metrics = {
        "classes": names,
        "test": {"accuracy": round(correct / n, 4), "top5": round(top5_hits / n, 4),
                 "correct": correct, "n": n},
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
        "latency_ms": {"gpu": gpu_ms, "cpu": cpu_ms, "batch": 1, "warmup": 10, "runs": 100,
                       "gpu_device": torch.cuda.get_device_name(0) if device == "cuda" else None},
        "model": {"weights": str(WEIGHTS.relative_to(ROOT)), "arch": "yolo11n-cls",
                  "size_mb": round(WEIGHTS.stat().st_size / 1e6, 2),
                  "params": params, "imgsz": IMGSZ},
        "split_seed": SEED,
    }
    METRICS.write_text(json.dumps(metrics, indent=2) + "\n")

    print(f"\ntest accuracy : {metrics['test']['accuracy']:.4f}  ({correct}/{n})")
    print(f"gpu ms/image  : {gpu_ms}")
    print(f"cpu ms/image  : {cpu_ms}")
    print(f"model size    : {metrics['model']['size_mb']} MB, {params:,} params")
    print(f"\nwrote {METRICS}")
    if metrics["test"]["accuracy"] < 0.95:
        print("NOTE: below the 95% target. The real number is published unchanged.")


# --------------------------------------------------------------------------- #
# step 4: cam
# --------------------------------------------------------------------------- #

def _mask_overlap(heat: np.ndarray, mask: np.ndarray) -> float:
    total = float(heat.sum())
    return float(heat[mask].sum() / total) if total > 1e-8 else 0.0


def cmd_cam(_args) -> None:
    model, names, device = load_model()
    items = _test_files(names)
    rng = random.Random(SEED)

    # probe: which target layer puts heatmap mass on the leaf?
    probe = [p for p, t in items if "healthy" not in names[t]]
    rng.shuffle(probe)
    probe = probe[:12]
    scores = {}
    for which in ("head", "backbone"):
        vals = []
        for path in probe:
            with Image.open(path) as im:
                im = im.convert("RGB")
                vals.append(_mask_overlap(gradcam(model, im, device, which=which), leaf_mask(im)))
        scores[which] = float(np.mean(vals))
        print(f"target layer {which:<9} mask overlap {scores[which]:.3f}  (n={len(vals)})")
    best = max(scores, key=scores.get)
    print(f"\nchosen layer: {best}  ({target_layer(model, best).__class__.__name__})")
    if scores[best] < 0.5:
        print("WARNING: under 0.5 - the heatmap sits largely on the background. Reported, not hidden.")

    # 6 examples: 5 correct across classes, 1 real failure
    out = ASSETS / "cam"
    if out.exists():
        shutil.rmtree(out)          # stale examples from an earlier run must not survive
    out.mkdir(parents=True)
    chosen, wrong, used = [], None, set()
    order = list(items)
    rng.shuffle(order)
    for path, truth in order:
        with Image.open(path) as im:
            im = im.convert("RGB")
            probs, _ = predict(model, im, device)
            pred = int(probs.argmax())
            if pred != truth and wrong is None:
                wrong = (path, truth, pred, float(probs[pred]))
            elif pred == truth and names[truth] not in used and len(chosen) < 5:
                used.add(names[truth])
                chosen.append((path, truth, pred, float(probs[pred])))
        if len(chosen) == 5 and wrong is not None:
            break

    manifest = []
    for i, (path, truth, pred, conf) in enumerate(chosen + ([wrong] if wrong else [])):
        tag = "correct" if truth == pred else "failure"
        with Image.open(path) as im:
            im = im.convert("RGB")
            heat = gradcam(model, im, device, class_idx=pred, which=best)
            mask = leaf_mask(im)
            stem = f"{i}_{tag}_{names[truth]}"
            _crop224(im).save(out / f"{stem}_leaf.png")
            Image.fromarray((mask * 255).astype(np.uint8)).save(out / f"{stem}_mask.png")
            overlay(im, heat).save(out / f"{stem}_cam.png")
        manifest.append({
            "stem": stem, "source": str(path.relative_to(ROOT)), "truth": names[truth],
            "pred": names[pred], "confidence": round(conf, 4),
            "mask_overlap": round(_mask_overlap(heat, mask), 4), "kind": tag,
        })
        print(f"{stem:<36} conf {conf:.3f}  overlap {manifest[-1]['mask_overlap']:.3f}")

    (out / "cam.json").write_text(json.dumps(
        {"target_layer": best, "probe_scores": scores, "examples": manifest}, indent=2) + "\n")
    if wrong is None:
        print("\nno misclassification in the test set, so no failure example exists")
    print(f"\nwrote {out}")


# --------------------------------------------------------------------------- #
# step 5: export
# --------------------------------------------------------------------------- #

def cmd_export(_args) -> None:
    try:
        import onnx  # noqa: F401
        import onnxslim  # noqa: F401
    except ImportError as exc:
        sys.exit(f"ONNX export needs onnx and onnxslim: {exc}\n"
                 "They are pinned in requirements.txt. Install them, or skip this step.")
    from ultralytics import YOLO

    if not WEIGHTS.exists():
        sys.exit(f"missing weights: {WEIGHTS}")
    path = YOLO(str(WEIGHTS)).export(format="onnx", imgsz=IMGSZ, simplify=True)
    print(f"\nexported {path}  ({Path(path).stat().st_size / 1e6:.2f} MB)")
    print("PyTorch best.pt stays as the weights the GUI serves.")


# --------------------------------------------------------------------------- #

def _self_check() -> None:
    """Smallest runnable check of the non-trivial pure logic: augment, mask, preprocess."""
    rng = random.Random(0)
    img = Image.new("RGB", (256, 256), (90, 140, 70))
    for x in range(100, 160):
        for y in range(100, 160):
            img.putpixel((x, y), (150, 90, 40))

    out = _augment(img, rng)
    assert out.size == (256, 256), out.size
    flat = Image.new("RGB", (256, 256), (10, 10, 10))
    assert _augment(flat, rng).size == (256, 256)

    mask = leaf_mask(img)
    assert mask.shape == (IMGSZ, IMGSZ), mask.shape
    assert mask.dtype == bool and mask.all(), "a fully saturated frame is all leaf"

    grey = Image.new("RGB", (256, 256), (128, 128, 128))  # zero saturation background
    assert leaf_mask(grey).all(), "unusable threshold falls back to all-true, never empty"

    tensor = preprocess(Image.new("RGB", (640, 480)))
    assert tuple(tensor.shape) == (1, 3, IMGSZ, IMGSZ), tensor.shape
    assert 0.0 <= float(tensor.min()) and float(tensor.max()) <= 1.0

    heat = np.zeros((IMGSZ, IMGSZ), dtype=np.float32)
    heat[:112] = 1.0
    half = np.zeros((IMGSZ, IMGSZ), dtype=bool)
    half[:112] = True
    assert abs(_mask_overlap(heat, half) - 1.0) < 1e-6
    assert _mask_overlap(np.zeros_like(heat), half) == 0.0

    assert group_key("image (100)_Rotate") == "image (100)"
    assert group_key("image (100)") == "image (100)"
    assert group_key("image (100)_aug3") == "image (100)_aug3", "my own copies are not v1 copies"
    assert group_key("Rotate") == "Rotate", "a bare op name is not a suffix"

    r, g, b, _ = CAM_CMAP(0.0)
    assert (round(r * 255), round(g * 255), round(b * 255)) == (248, 247, 240), "cmap starts at ivory"
    print("self-check ok")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("split").set_defaults(fn=cmd_split)
    train = sub.add_parser("train")
    train.add_argument("--batch", type=int, default=32)
    train.add_argument("--epochs", type=int, default=50)
    train.set_defaults(fn=cmd_train)
    sub.add_parser("eval").set_defaults(fn=cmd_eval)
    sub.add_parser("cam").set_defaults(fn=cmd_cam)
    sub.add_parser("export").set_defaults(fn=cmd_export)
    sub.add_parser("self-check").set_defaults(fn=lambda _a: _self_check())
    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
