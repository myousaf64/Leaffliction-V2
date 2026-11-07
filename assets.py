#!/usr/bin/env python3
"""Leaffliction v2 LinkedIn assets. Six 1080x1350 PNGs plus carousel.pdf.

Every number comes from metrics.json or from counting dataset/. Nothing is hardcoded.
Run after: pipeline.py split, train, eval, cam.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from PIL import Image

from pipeline import (ASSETS, CAM_CMAP, CHARCOAL, DATASET, FOREST, IVORY, LIME, METRICS,
                      ROOT, SAGE, SEED, TERRACOTTA, TERRACOTTA_TEXT, _augment)

W_IN, H_IN, DPI = 7.2, 9.0, 150          # 1080 x 1350
CAM_DIR = ASSETS / "cam"

plt.rcParams.update({
    "font.family": "DejaVu Sans",        # matplotlib-bundled, present everywhere
    "figure.facecolor": IVORY,
    "axes.facecolor": IVORY,
    "text.color": CHARCOAL,
    "axes.labelcolor": CHARCOAL,
    "xtick.color": FOREST,
    "ytick.color": FOREST,
    "axes.edgecolor": "#C3D1BD",
    "savefig.facecolor": IVORY,
})


# --------------------------------------------------------------------------- #
# data, all measured
# --------------------------------------------------------------------------- #

def load_metrics() -> dict:
    if not METRICS.exists():
        sys.exit(f"missing {METRICS}\nrun: python pipeline.py eval")
    return json.loads(METRICS.read_text())


def count_splits(classes: list[str]) -> dict[str, dict[str, int]]:
    """Ground truth from dataset/ itself: originals per split, plus train after augmentation."""
    if not DATASET.is_dir():
        sys.exit(f"missing {DATASET}\nrun: python pipeline.py split")
    out = {}
    for cls in classes:
        files = sorted((DATASET / "train" / cls).iterdir())
        out[cls] = {
            "train": sum("_aug" not in f.stem for f in files),
            "train_aug": len(files),
            "val": len(list((DATASET / "val" / cls).iterdir())),
            "test": len(list((DATASET / "test" / cls).iterdir())),
        }
    return out


def short(name: str) -> str:
    crop, *rest = name.split("_")
    return f"{crop}\n{' '.join(rest).lower()}"


def inline(name: str) -> str:
    crop, *rest = name.split("_")
    return f"{crop} {' '.join(rest).lower()}"


def header(fig: Figure, kicker: str, title: str, sub: str = "") -> None:
    """Forest band across the top, lime rule beneath it."""
    fig.patches.append(plt.Rectangle((0, 0.885), 1, 0.115, transform=fig.transFigure,
                                     facecolor=FOREST, zorder=0))
    fig.patches.append(plt.Rectangle((0, 0.879), 1, 0.006, transform=fig.transFigure,
                                     facecolor=LIME, zorder=0))
    fig.text(0.07, 0.957, kicker.upper(), color=LIME, fontsize=11.5, fontweight="bold",
             family="DejaVu Sans")
    fig.text(0.07, 0.908, title, color=IVORY, fontsize=25, fontweight="bold", va="center")
    if sub:
        fig.text(0.07, 0.845, sub, color=CHARCOAL, fontsize=12.5, va="top", wrap=True)


MASK_CMAP = LinearSegmentedColormap.from_list("mask", [FOREST, SAGE])


def panel(fig: Figure, path: Path, box, title: str, caption: str, title_color=FOREST,
          caption_color=CHARCOAL, title_size=13, caption_size=10) -> None:
    """One image with its title above and caption below, at explicit figure coordinates."""
    x, bottom, w, h = box
    ax = fig.add_axes([x, bottom, w, h])
    with Image.open(path) as im:
        arr = np.asarray(im)
    ax.imshow(arr, cmap=MASK_CMAP) if arr.ndim == 2 else ax.imshow(arr)
    ax.set_axis_off()
    fig.text(x + w / 2, bottom + h + 0.012, title, ha="center", fontsize=title_size,
             color=title_color, fontweight="bold")
    fig.text(x + w / 2, bottom - 0.025, caption, ha="center", fontsize=caption_size,
             color=caption_color, alpha=0.9)


def footer(fig: Figure, text: str) -> None:
    fig.text(0.07, 0.028, text, color=CHARCOAL, fontsize=9.5, alpha=0.65, va="bottom")


# --------------------------------------------------------------------------- #
# 1. class balance, before and after offline augmentation
# --------------------------------------------------------------------------- #

def fig_balance(m: dict, counts: dict) -> Figure:
    classes = m["classes"]
    before = [counts[c]["train"] for c in classes]
    after = [counts[c]["train_aug"] for c in classes]
    ratio = max(before) / min(before)

    fig = plt.figure(figsize=(W_IN, H_IN), dpi=DPI)
    header(fig, "step 1 - data", "The dataset was lopsided",
           f"Train split before balancing: {min(before)} to {max(before)} images per class, "
           f"a {ratio:.1f}x gap. Offline augmentation lifts every class to {max(after)}. "
           "The split runs first, so no augmented copy ever reaches val or test.")
    ax = fig.add_axes([0.30, 0.115, 0.64, 0.625])
    y = np.arange(len(classes))
    ax.barh(y + 0.21, after, height=0.4, color=LIME, label=f"after  ({sum(after):,})",
            edgecolor=FOREST, linewidth=0.7)
    ax.barh(y - 0.21, before, height=0.4, color=SAGE, label=f"before ({sum(before):,})",
            edgecolor=FOREST, linewidth=0.7)
    for yi, (b, a) in enumerate(zip(before, after)):
        ax.text(b + 18, yi - 0.21, str(b), va="center", fontsize=9.5, color=CHARCOAL)
        ax.text(a + 18, yi + 0.21, str(a), va="center", fontsize=9.5,
                color=FOREST, fontweight="bold")
    ax.set_yticks(y, [inline(c) for c in classes], fontsize=11)
    ax.invert_yaxis()
    ax.set_xlabel("training images", fontsize=11)
    ax.set_xlim(0, max(after) * 1.16)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.015), ncol=2,
          frameon=False, fontsize=11.5)
    footer(fig, f"Counted from dataset/train. Split seed {m['split_seed']}, 70/15/15.")
    return fig


# --------------------------------------------------------------------------- #
# 2. one leaf, six augmentations
# --------------------------------------------------------------------------- #

def fig_augment(m: dict) -> Figure:
    rng = random.Random(SEED)
    cls = m["classes"][0]
    src = sorted(p for p in (DATASET / "train" / cls).iterdir() if "_aug" not in p.stem)[0]
    with Image.open(src) as im:
        leaf = im.convert("RGB")

    fig = plt.figure(figsize=(W_IN, H_IN), dpi=DPI)
    header(fig, "step 2 - augmentation", "One leaf, six variants",
           "Flip, rotate, brightness and contrast. Every copy gets at least one geometric op, "
           "so no variant is a brightness-only near-duplicate of its source.")
    ax = fig.add_axes([0.355, 0.525, 0.29, 0.235])
    ax.imshow(leaf); ax.set_axis_off()
    ax.set_title("original", fontsize=11, color=FOREST, fontweight="bold", pad=6)

    for i in range(6):
        r, c = divmod(i, 3)
        ax = fig.add_axes([0.075 + c * 0.295, 0.285 - r * 0.215, 0.26, 0.19])
        ax.imshow(_augment(leaf, rng)); ax.set_axis_off()
        ax.set_title(f"variant {i + 1}", fontsize=10, color=CHARCOAL, pad=4)
    footer(fig, f"{inline(cls)} - {src.name}. Augmentation applies to the train split only.")
    return fig


# --------------------------------------------------------------------------- #
# 3. leaf -> mask -> heatmap
# --------------------------------------------------------------------------- #

def _cam_examples() -> list[dict]:
    meta = CAM_DIR / "cam.json"
    if not meta.exists():
        sys.exit(f"missing {meta}\nrun: python pipeline.py cam")
    return json.loads(meta.read_text())["examples"]


def fig_pipeline(m: dict) -> Figure:
    ex = next((e for e in _cam_examples() if e["kind"] == "correct"), _cam_examples()[0])
    fig = plt.figure(figsize=(W_IN, H_IN), dpi=DPI)
    header(fig, "step 3 - pipeline", "From photo to evidence",
           "The leaf is segmented by an HSV threshold, then Grad-CAM shows which pixels "
           "drove the prediction. A heatmap on the background would mean a model reading "
           "the pot, not the plant.")
    panels = [("photo", "_leaf.png", "224x224 centre crop"),
              ("leaf mask", "_mask.png", "HSV threshold, largest component"),
              ("Grad-CAM", "_cam.png", f"{ex['mask_overlap'] * 100:.0f}% of the heat is on the leaf")]
    for i, (title, suffix, caption) in enumerate(panels):
        panel(fig, CAM_DIR / f"{ex['stem']}{suffix}",
              (0.40, 0.585 - i * 0.235, 0.20, 0.16), title, caption)
    footer(fig, f"{inline(ex['truth'])}, predicted {inline(ex['pred'])} "
                f"at {ex['confidence'] * 100:.1f}%. Colormap ivory to lime to terracotta.")
    return fig


# --------------------------------------------------------------------------- #
# 4. six Grad-CAM examples
# --------------------------------------------------------------------------- #

def fig_heatmap(m: dict) -> Figure:
    ex = _cam_examples()
    n_fail = sum(e["kind"] == "failure" for e in ex)
    fig = plt.figure(figsize=(W_IN, H_IN), dpi=DPI)
    header(fig, "step 4 - explainability", "Where the model looks",
           f"{len(ex) - n_fail} correct predictions and {n_fail} real failure from the "
           "held-out test set. The heat marks the pixels that moved the logit.")
    for i, e in enumerate(ex[:6]):
        r, c = divmod(i, 2)
        ok = e["kind"] == "correct"
        panel(fig, CAM_DIR / f"{e['stem']}_cam.png",
              (0.19 + c * 0.42, 0.565 - r * 0.22, 0.20, 0.16),
              inline(e["pred"]),
              f"{e['confidence'] * 100:.0f}%" if ok else f"wrong, truly {inline(e['truth'])}",
              title_color=FOREST if ok else TERRACOTTA_TEXT,
              caption_color=CHARCOAL if ok else TERRACOTTA_TEXT,
              title_size=11, caption_size=9.5)
    overlaps = [e["mask_overlap"] for e in ex]
    footer(fig, f"Heat inside the leaf mask: {min(overlaps) * 100:.0f}% to "
                f"{max(overlaps) * 100:.0f}%, mean {np.mean(overlaps) * 100:.0f}%.")
    return fig


# --------------------------------------------------------------------------- #
# 5. confusion matrix
# --------------------------------------------------------------------------- #

def fig_confusion(m: dict) -> Figure:
    classes, cm = m["classes"], np.array(m["confusion_matrix"])
    fig = plt.figure(figsize=(W_IN, H_IN), dpi=DPI)
    errors = int(cm.sum() - np.trace(cm))
    header(fig, "step 5 - results", "Confusion matrix, test set",
           f"{cm.sum():,} held-out images, {errors} wrong. Rows are the truth, columns are the "
           f"prediction. Accuracy {m['test']['accuracy'] * 100:.2f}%.")
    ax = fig.add_axes([0.235, 0.265, 0.70, 0.515])
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    ax.imshow(norm, cmap=matplotlib.colors.LinearSegmentedColormap.from_list(
        "cm", [IVORY, SAGE, LIME, FOREST]), vmin=0, vmax=1)
    for i in range(len(classes)):
        for j in range(len(classes)):
            v = int(cm[i, j])
            if not v:
                continue
            ax.text(j, i, str(v), ha="center", va="center", fontsize=13,
                    fontweight="bold" if i == j else "normal",
                    color=IVORY if norm[i, j] > 0.55 else (CHARCOAL if i == j else TERRACOTTA_TEXT))
    ax.set_xticks(range(len(classes)), [short(c) for c in classes], fontsize=9.5)
    ax.set_yticks(range(len(classes)), [short(c) for c in classes], fontsize=9.5)
    ax.set_xlabel("predicted", fontsize=12, fontweight="bold", color=FOREST, labelpad=8)
    ax.set_ylabel("actual", fontsize=12, fontweight="bold", color=FOREST, labelpad=8)
    ax.set_xticks(np.arange(-.5, len(classes), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(classes), 1), minor=True)
    ax.grid(which="minor", color=IVORY, linewidth=2)
    ax.tick_params(which="minor", length=0)

    worst = min(m["per_class"].items(), key=lambda kv: kv[1]["recall"])
    fig.text(0.07, 0.115, f"Weakest class: {inline(worst[0])}, recall "
                          f"{worst[1]['recall'] * 100:.1f}% on {worst[1]['support']} test images.",
             fontsize=11, color=TERRACOTTA_TEXT, fontweight="bold")
    footer(fig, f"yolo11n-cls, {m['model']['imgsz']}px input, seed {m['split_seed']}. "
                "Test split only, never seen in training or validation.")
    return fig


# --------------------------------------------------------------------------- #
# 6. pitch card
# --------------------------------------------------------------------------- #

def fig_pitch(m: dict, counts: dict) -> Figure:
    lat, mod = m["latency_ms"], m["model"]
    originals = sum(v["train"] + v["val"] + v["test"] for v in counts.values())
    fig = plt.figure(figsize=(W_IN, H_IN), dpi=DPI)
    fig.patches.append(plt.Rectangle((0, 0), 1, 1, transform=fig.transFigure,
                                     facecolor=FOREST, zorder=0))
    fig.patches.append(plt.Rectangle((0.07, 0.845), 0.17, 0.007, transform=fig.transFigure,
                                     facecolor=LIME, zorder=1))
    fig.text(0.07, 0.885, "CROP DISEASE DETECTION", color=LIME, fontsize=12.5,
             fontweight="bold", zorder=2)
    fig.text(0.07, 0.775, "Early crop\ndisease detection\nfrom a phone photo.",
             color=IVORY, fontsize=35, fontweight="bold", va="top", linespacing=1.2, zorder=2)
    fig.text(0.07, 0.565,
             "A leaf disease spotted a week early is a crop saved.\n"
             "A grower holds up a phone; the model names the disease\n"
             "and shows the lesions it used to decide.",
             color=SAGE, fontsize=13.5, va="top", zorder=2, linespacing=1.6)

    gpu = f"{lat['gpu']:.1f} ms" if lat["gpu"] else "n/a"
    tiles = [("TEST ACCURACY", f"{m['test']['accuracy'] * 100:.2f}%", f"on {m['test']['n']:,} held-out images"),
             ("GPU", gpu, "per image, RTX 3050 Ti"),
             ("CPU", f"{lat['cpu']:.1f} ms", "per image, batch 1"),
             ("MODEL", f"{mod['size_mb']:.1f} MB", f"{mod['params'] / 1e6:.2f}M params, {mod['arch']}")]
    for i, (k, v, note) in enumerate(tiles):
        r, c = divmod(i, 2)
        x, y = 0.07 + c * 0.45, 0.285 - r * 0.175
        fig.patches.append(plt.Rectangle((x, y), 0.41, 0.145, transform=fig.transFigure,
                                         facecolor=SAGE, zorder=1, linewidth=0))
        fig.text(x + 0.03, y + 0.112, k, color=FOREST, fontsize=10, fontweight="bold", zorder=2)
        fig.text(x + 0.03, y + 0.055, v, color=CHARCOAL, fontsize=27, fontweight="bold",
                 va="center", zorder=2)
        fig.text(x + 0.03, y + 0.018, note, color=FOREST, fontsize=9.5, zorder=2)
    fig.text(0.07, 0.035, f"{len(m['classes'])} classes - apple and grape - "
                          f"{originals:,} source images - Grad-CAM explainability",
             color=SAGE, fontsize=10.5, alpha=0.9, zorder=2)
    return fig


# --------------------------------------------------------------------------- #

def main() -> None:
    m = load_metrics()
    counts = count_splits(m["classes"])
    ASSETS.mkdir(exist_ok=True)

    pages = [
        ("pitch", lambda: fig_pitch(m, counts)),
        ("balance", lambda: fig_balance(m, counts)),
        ("pipeline", lambda: fig_pipeline(m)),
        ("heatmap", lambda: fig_heatmap(m)),
        ("confusion", lambda: fig_confusion(m)),
        ("augment", lambda: fig_augment(m)),
    ]
    figures = {}
    for name, make in pages:
        fig = make()
        out = ASSETS / f"{name}.png"
        fig.savefig(out, dpi=DPI)
        with Image.open(out) as im:
            assert im.size == (1080, 1350), f"{out.name} is {im.size}, expected (1080, 1350)"
        print(f"{out.relative_to(ROOT)}  {im.size[0]}x{im.size[1]}  {out.stat().st_size // 1024} KB")
        figures[name] = fig

    order = ["pitch", "balance", "pipeline", "heatmap", "confusion"]
    pdf_path = ASSETS / "carousel.pdf"
    with PdfPages(pdf_path) as pdf:
        for name in order:
            pdf.savefig(figures[name])
    for fig in figures.values():
        plt.close(fig)
    print(f"{pdf_path.relative_to(ROOT)}  {len(order)} pages: {', '.join(order)}")


if __name__ == "__main__":
    main()
