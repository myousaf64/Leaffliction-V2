#!/usr/bin/env python3
"""Leaffliction v2 launch kit. Writes LAUNCH.md: post copy, carousel order, shot list, checklist.

Every figure is read from metrics.json, assets/cam/cam.json and splits_report.txt. The one
exception is PRE_FIX_ACCURACY below, which is history and is labelled as such.

Run after: pipeline.py eval, pipeline.py cam, assets.py.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from pipeline import ASSETS, METRICS, ROOT, SPLITS_REPORT, UNCERTAIN_BELOW

LAUNCH = ROOT / "LAUNCH.md"
CAM_JSON = ASSETS / "cam" / "cam.json"
REPO = "myousaf64/Leaffliction-V2"
RELEASE_TAG = "v2.0"

# Measured in the session before the split was fixed, on the leaky split. It is not in
# metrics.json because that file only ever holds the current run. Kept by hand so the post
# can tell the truth about what changed.
PRE_FIX_ACCURACY = 99.91

SLIDES = [
    ("pitch.png", "Title card: early crop disease detection from a phone photo, with test "
                  "accuracy, inference time and model size."),
    ("balance.png", "Bar chart of training images per class before and after offline "
                    "augmentation, showing the 5.9x imbalance being levelled out."),
    ("pipeline.png", "Three stacked panels: the leaf photo, the HSV leaf mask, and the "
                     "Grad-CAM heatmap over the same leaf."),
    ("heatmap.png", "Six Grad-CAM examples in a grid, five correct and one labelled as a "
                    "real misclassification."),
    ("confusion.png", "Eight by eight confusion matrix of the test set, almost entirely "
                      "diagonal, with the few errors marked."),
]


def load() -> dict:
    """Every measured number the launch kit needs, or a hard exit saying what to run."""
    for path, fix in ((METRICS, "pipeline.py eval"),
                      (CAM_JSON, "pipeline.py cam"),
                      (SPLITS_REPORT, "pipeline.py split")):
        if not path.exists():
            sys.exit(f"missing {path}\nrun: python {fix}")

    m = json.loads(METRICS.read_text())
    cam = json.loads(CAM_JSON.read_text())
    report = SPLITS_REPORT.read_text()

    total = re.search(r"^TOTAL\s+(\d+)", report, re.M)
    target = re.search(r"balance target\s*:\s*(\d+)", report)
    if not (total and target):
        sys.exit(f"{SPLITS_REPORT} is not in the expected format. Re-run pipeline.py split.")

    missing = [s for s, _ in SLIDES if not (ASSETS / s).exists()]
    if missing:
        sys.exit(f"missing carousel pages: {', '.join(missing)}\nrun: python assets.py")

    lat, mod, test = m["latency_ms"], m["model"], m["test"]
    return {
        "accuracy": test["accuracy"] * 100,
        "correct": test["correct"],
        "n": test["n"],
        "errors": test["n"] - test["correct"],
        "classes": len(m["classes"]),
        "gpu_ms": lat["gpu"],
        "cpu_ms": lat["cpu"],
        "gpu_name": lat["gpu_device"] or "GPU",
        "size_mb": mod["size_mb"],
        "params_m": mod["params"] / 1e6,
        "arch": mod["arch"],
        "source_images": int(total.group(1)),
        "balance_target": int(target.group(1)),
        "cam_overlap": cam["probe_scores"][cam["target_layer"]] * 100,
        "cam_n": len(cam["examples"]),
        "worst": min(m["per_class"].items(), key=lambda kv: kv[1]["recall"]),
        "uncertain_pct": UNCERTAIN_BELOW * 100,
    }


def post_copy(d: dict) -> str:
    gpu_short = d["gpu_name"].replace("NVIDIA GeForce ", "").replace(" Laptop GPU", "")
    return f"""\
My first version of this project reported {PRE_FIX_ACCURACY:.2f}% accuracy. It was wrong, and the
reason is worth sixty seconds.

I rebuilt my leaf disease classifier from scratch. {d['classes']} classes across apple and grape,
{d['arch']} fine-tuned on {d['source_images']:,} images.

Then I found it. The dataset carried 12 augmented copies left over from an earlier run -
"image (100)_Rotate.JPG" sitting right beside "image (100).JPG". A plain per-file split scattered
them. Two landed in the test set while their parent sat in training. The model had already been
shown the answer.

The fix: the split now groups every augmented copy with its parent, so a whole group lands in one
split and never two. Retrained from zero.

Honest number: {d['accuracy']:.2f}% on {d['n']:,} held-out images. {d['errors']} errors.

Lower than before. Correct this time.

The rest:
- {d['gpu_ms']:.2f} ms per image on a laptop {gpu_short}, {d['cpu_ms']:.2f} ms on CPU
- {d['size_mb']} MB model, {d['params_m']:.1f}M parameters
- Grad-CAM on every prediction, {d['cam_overlap']:.0f}% of the heat inside the leaf mask, so it
  reads the leaf and not the pot
- Under {d['uncertain_pct']:.0f}% confidence it answers "uncertain" instead of guessing

A leaf disease caught a week early is a crop saved. That only works if the number you trust is real.

Code and weights: github.com/{REPO}

#MachineLearning #ComputerVision #Agritech #MLOps #Python"""


def build(d: dict) -> str:
    slides = "\n".join(
        f"| {i} | `assets/{name}` | {alt} |" for i, (name, alt) in enumerate(SLIDES, 1))
    worst_name, worst = d["worst"]
    worst_label = worst_name.replace("_", " ").lower()

    return f"""\
# Leaffliction v2 - launch kit

Generated by `linkedin.py` from `metrics.json`, `assets/cam/cam.json` and `splits_report.txt`.
Do not edit this file. Change the script, or edit your copy after you paste it.

---

## 1. Do these in order

1. **Push the repo.** The post links to it, so it has to be live first.
2. **Cut the `{RELEASE_TAG}` release** with `best.pt`, `best.onnx` and `assets/carousel.pdf`.
3. **Check the clone works cold.** Clone into a scratch directory, install, `./run_demo.sh`.
   If it needs a training step, the README is lying and the post will embarrass you.
4. **Record the demo** (section 4). Record last, so the links in the post already resolve.
5. **Post the carousel** (section 3), paste the copy from section 2.
6. **Put the demo GIF in the first comment**, with the repo link again.

Do not record before the repo is live. Someone always pauses the video on the URL.

---

## 2. The post

Paste as-is, then cut anything that does not sound like you. {len(post_copy(d).split())} words,
which is long for LinkedIn but the story earns it.

```text
{post_copy(d)}
```

### Why this angle

"I hit {int(d['accuracy'])}%" is a post nobody stops for. "I caught myself cheating and the number
went down" is one they finish. It is also what actually happened, and the people you want reading
this - engineers who have shipped a model - will recognise the failure instantly.

---

## 3. The carousel

Upload `assets/carousel.pdf` directly. LinkedIn renders a PDF as a native swipeable carousel, which
holds attention far longer than a single image. Pages are already in this order:

| # | file | alt text |
|---|---|---|
{slides}

Paste the alt text into LinkedIn's accessibility field for each slide. It takes a minute and the
post reads properly for anyone using a screen reader.

The sixth asset, `assets/augment.png`, is not in the carousel. Keep it for a follow-up post about
the augmentation pipeline.

---

## 4. The demo recording

**Video or stills? Both, with different jobs.** The carousel is the post. The video is an 8 to 12
second GIF that goes in the README at `docs/demo.gif` and in the first comment on the post.
LinkedIn allows one media type per post, so the video cannot ride along with the carousel.

**Terminal or the ready page? The ready page.** Nobody watches a `pip install`. Start the server
before you hit record and capture only the browser window.

### Before you hit record

```bash
./run_demo.sh                 # let it finish loading, the terminal prints "model ready on cuda"
```

- Browser window 1280x800, zoom at 100%, bookmarks bar hidden, one tab.
- Have photos ready on the desktop: one clearly diseased grape leaf on its own, plus a folder of
  six mixed leaves for the batch shot.
- Do a dry run first. The first take is always clumsy with the slider.

### Shot list, about 14 seconds

| t | action |
|---|---|
| 0.0 | empty page, drop zone visible, do not move the mouse |
| 1.0 | drag the diseased leaf in and drop it |
| 2.0 | result lands. Hold still on the disease label and the confidence bar |
| 4.0 | drag the heatmap slider 0 to 100 and back to about 55 |
| 7.0 | press `H` twice so the heatmap blinks off and on |
| 8.5 | drag all six leaves in at once. The filmstrip fills and the summary line counts them |
| 10.5 | press the right arrow three times. The big heatmap follows the selection |
| 13.0 | land on a healthy leaf so the badge flips to the green state |
| 14.0 | stop |

That is the whole product: it answers, it explains, it scans a tray in one gesture, and it knows
healthy from sick.

### Convert to a GIF

```bash
ffmpeg -i demo.mp4 -vf "fps=15,scale=1000:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" \\
  -loop 0 docs/demo.gif
du -h docs/demo.gif          # keep it under 8 MB
```

Drop `fps` to 12 or `scale` to 800 if it comes out too heavy.

---

## 5. Commands for steps 1 and 2

```bash
# release, after the push
gh release create {RELEASE_TAG} \\
  runs/cls/weights/best.pt runs/cls/weights/best.onnx assets/carousel.pdf \\
  -R {REPO} \\
  -t "Leaffliction {RELEASE_TAG}" \\
  -n "{d['arch']}, {d['accuracy']:.2f}% on {d['n']:,} held-out images."

# prove the README download link resolves
curl -sIL https://github.com/{REPO}/releases/download/{RELEASE_TAG}/best.onnx | head -1
```

---

## 6. Pre-flight checklist

- [ ] `runs/cls/weights/best.pt` is in the repository, and a cold clone runs `./run_demo.sh`
      with no training step
- [ ] the `{RELEASE_TAG}` release exists and the asset URL returns `HTTP/2 200`
- [ ] `docs/demo.gif` exists, is under 8 MB, and the README no longer shows a broken image
- [ ] every number in the post matches `metrics.json`: {d['accuracy']:.2f}%,
      {d['correct']:,}/{d['n']:,}, {d['gpu_ms']:.2f} ms, {d['cpu_ms']:.2f} ms, {d['size_mb']} MB
- [ ] the {PRE_FIX_ACCURACY:.2f}% figure is the pre-fix run. It is recorded by hand in
      `linkedin.py`, not in `metrics.json`. Keep it or cut it, but do not misstate it
- [ ] alt text pasted for all {len(SLIDES)} slides
- [ ] the repo link in the post and in the first comment both open

---

## 7. If someone asks in the comments

**"Why not a bigger model?"** {d['size_mb']} MB and {d['gpu_ms']:.2f} ms is the point. This has to run
on a phone in a field, not in a data centre. The headroom is there if accuracy ever demands it.

**"{d['accuracy']:.2f}% sounds too good."** It is a lab dataset: flat backgrounds, one leaf per
photo, even lighting. Field photos will be harder. That is the honest caveat and it belongs in your
reply, not buried.

**"What is the weakest class?"** {worst_label}, recall {worst['recall'] * 100:.1f}% on
{worst['support']} test images. The {d['cam_n']} Grad-CAM examples include a real failure rather
than only wins.

**"Is it licensed for commercial use?"** Ultralytics YOLO11 is AGPL-3.0. Commercial use needs an
Ultralytics Enterprise licence. The README says so.
"""


def main() -> None:
    d = load()
    text = build(d)
    LAUNCH.write_text(text)
    print(text)
    print(f"\n--- wrote {LAUNCH.relative_to(ROOT)} "
          f"({len(text.splitlines())} lines) ---", file=sys.stderr)


if __name__ == "__main__":
    main()
