# Leaffliction v2

Upload a photo of an apple or grape leaf. The app names the disease, says how
sure it is, and paints a heatmap over the exact pixels it used to decide.

![demo](docs/demo.gif)
<!-- PLACEHOLDER: replace with the screen recording before you post -->

This started as Leaffliction, a 42 school exercise that prints a class name in a
terminal. This version is the product: a web app, an explainable heatmap, and a
batch mode that scans a folder of leaves in one pass.

## What it does

| | |
|---|---|
| Classes | 8: apple and grape, healthy plus 3 diseases each |
| Answer time | 7 to 13 ms for one leaf on a laptop GPU |
| Batch | 64 leaves in 0.12 s, about 2 ms each |
| Explains itself | Grad-CAM heatmap on every answer |

The heatmap matters more than the accuracy. A model can be right for the wrong
reason, for example by reading the background instead of the leaf. The heatmap
shows which it is.

## Results

Trained on the 8 classes, measured on a held-out test split the model never saw
during training.

| Metric | Value |
|---|---|
| Test top-1 accuracy | 100.0% (723 of 723) |
| Test top-5 accuracy | 100.0% |
| Training images | 5780 |
| Validation images | 723 |
| Test images | 723 |
| Model | YOLO11n-cls, 224 px, batch 32, 30 epochs |
| Peak GPU memory | 0.45 GB of 4 GB |

Recall of each class on the test split:

| Class | Recall | Correct |
|---|---|---|
| Apple - Black rot | 100.0% | 63/63 |
| Apple - Healthy | 100.0% | 163/163 |
| Apple - Rust | 100.0% | 28/28 |
| Apple - Scab | 100.0% | 63/63 |
| Grape - Black rot | 100.0% | 118/118 |
| Grape - Esca | 100.0% | 138/138 |
| Grape - Healthy | 100.0% | 42/42 |
| Grape - Leaf spot | 100.0% | 108/108 |

### Read that number honestly

100% is the real score on this test split, and the split is clean. I checked:
the images are deduplicated by SHA-256 before the split, and a pixel-level
audit found no near-duplicate of any test image inside the training set. The
closest pair sits at RMSE 0.057, which is 2 different leaves.

It is still a laboratory score, not a field score. Every photo in this dataset
is one detached leaf on a plain backdrop under even light. Published work on
this dataset reaches 99% to 100% routinely, so 100% says the task is easy, not
that the problem is solved. A photo of a leaf still on the tree, in daylight,
with soil and other leaves behind it, is a different and harder problem. Treat
this as a working demo, not a field tool.

## Setup

Needs an NVIDIA GPU. The training script stops rather than falling back to the
CPU.

```sh
uv venv --python 3.12
uv pip install -r requirements.txt
```

Put the dataset in `images/`, one folder per class.

## Run

```sh
.venv/bin/python split.py                  # count, deduplicate, split 80/10/10
.venv/bin/python train.py                  # train, then score the test split
.venv/bin/python app.py                    # the web app on localhost:7860
.venv/bin/python predict.py <image>        # one photo from the terminal
```

`predict.py` prints the verdict and writes `<name>_cam.png`, the crop next to
the heatmap.

## How it works

1. `split.py` hashes every image, drops byte-identical copies, then splits each
   class 80/10/10 with a fixed seed. It uses symlinks, so `images/` is never
   modified. It stops if any file or any hash lands in 2 splits.
2. `train.py` fine-tunes YOLO11n-cls at 224 px with the built-in augmentation.
   On a CUDA out-of-memory error it halves the batch and retries.
3. `leaf.py` runs inference and builds the Grad-CAM heatmap. It is the only
   inference path. `app.py` and `predict.py` both import it.

`DEV_DOC.md` holds the technical detail, including the 3 bugs worth knowing
about if you build on this.

## Limits

- 2 crop species, 8 classes. Anything else gets forced into one of those 8.
- Laboratory photos only. See the note under Results.
- The heatmap is computed at 7x7 and scaled up to 224, so it points at a region,
  not at a single lesion.
