# Developer notes

Technical detail for Leaffliction v2. `README.md` is the portfolio page. This
file is the one you read before you change the code.

## Environment

| Item | Value |
|---|---|
| OS | Ubuntu 26.04.1, kernel 7.0.0-42-generic |
| GPU | RTX 3050 Ti Mobile (GA107M), 4 GB |
| Driver | 595.99.02, CUDA 13.2 |
| Python | 3.12.13, created by `uv venv --python 3.12` |
| torch | 2.15.0+cu132, from the default PyPI wheel |
| ultralytics | 8.4.199 |
| gradio | 6.31.0 |

The system Python is 3.14, which has no torch wheel. The venv must be 3.12 or
3.13. The PyPI `torch` wheel already carries CUDA 13.2, so no custom
`--index-url` is needed.

## Dataset

Source layout: `images/<class>/*.JPG`, 7233 files, 129 MB.

| Class | Images |
|---|---|
| Apple_healthy | 1640 |
| Grape_Esca | 1382 |
| Grape_Black_rot | 1178 |
| Grape_spot | 1075 |
| Apple_scab | 629 |
| Apple_Black_rot | 626 |
| Grape_healthy | 422 |
| Apple_rust | 281 |

Imbalance is 5.8:1 between the largest and the smallest class. Nothing
rebalances it. `train.py` prints the recall of each class so a weak class cannot
hide behind the top-1 figure.

## Split

`split.py` does 3 things in one pass.

1. Prints the count of each class.
2. Hashes every file with SHA-256 and keeps one path per hash. This run dropped
   **7 byte-identical duplicates**, 7233 down to 7226, with **0 cross-class
   conflicts**, meaning no identical image carried 2 different labels.
3. Splits each class 80/10/10 with `random.Random(42)`, as 5780 / 723 / 723.

The split uses absolute symlinks into `images/`. Nothing in `images/` is moved
or written. Torchvision `ImageFolder`, which Ultralytics classification uses,
follows symlinks.

The script stops if a path or a hash appears in 2 splits. It refuses to delete
`data/`; it prints the `rm -rf data` command and exits.

Deduplication runs **before** the split on purpose. A duplicate that lands in
both train and test inflates the test score silently.

### Leakage audit

Byte-level hashing cannot catch 2 photos of the same leaf from slightly
different angles. 2 audits ran on top of it.

| Audit | Result |
|---|---|
| 8x8 average hash, Hamming <= 2 | 191 of 723 test images flagged (26.4%) |
| 32x32 RGB pixel RMSE | 0 test images below RMSE 0.05, minimum 0.0569, median 0.1095 |

The average-hash figure is a false positive. These photos share a plain grey
backdrop that dominates an 8x8 grayscale thumbnail, and one flagged pair even
crossed classes. The pixel-level audit is the reliable one, and it found no
near-duplicate. The split is clean, so the 100% test score is not leakage. It
reflects an easy dataset. See the honesty note in `README.md`.

## Training

```
yolo11n-cls.pt, imgsz 224, epochs 30, batch 32, amp on, device 0, seed 42
```

Built-in Ultralytics augmentation only, no custom transform. Batch 32 fitted on
the first attempt and peaked at 0.451 GB of the 4 GB, so the out-of-memory
retry never fired. The retry halves the batch down to a floor of 4 and prints
every size it tries.

`train.py` exits if `torch.cuda.is_available()` is false. There is no CPU path.

Artifacts land in `runs/classify/train/`: `weights/best.pt`, `results.csv`, and
the confusion matrix plots. All are gitignored.

## Grad-CAM

Target layer, confirmed by printing the module list rather than assuming it:

```
model.10.conv        Conv block of the Classify head, 1280 channels at 7x7
model.10.linear      gradient taken here, before the softmax
```

`leaf.py` prints both names on every load, so a heatmap is never traced to an
unnamed layer.

### 3 traps

1. **`model.predict()` runs inside `torch.no_grad()`.** Hooks fire but no
   gradient ever arrives, so the heatmap comes out empty. `leaf.py` calls the
   raw `nn.Module` instead.
2. **Ultralytics shares one `nn.SiLU` instance across the whole network.**
   `default_act` is a class attribute, so `named_modules()` reports that object
   once and `model.10.conv.act` has no full-network name. A hook placed on the
   shared activation fires for every layer in the model. `leaf.py` hooks the
   `Conv` block instead, whose output is already post-activation, and finds it
   as `net.model[-1].conv` by structure rather than by string.
3. **The `Classify` head returns softmax scores in eval mode.** At 100%
   confidence the gradient of that score is about zero and the heatmap becomes
   noise. `leaf.py` reads the pre-softmax output of the linear layer through a
   second hook.

A 4th detail: `classify_transforms(224)` is `Resize(224)` then
`CenterCrop(224)`, so the model never sees the full photo. The heatmap is drawn
on that centre crop. Drawn on the original it lands misaligned.

## Inference module

`leaf.py` is the only inference path. `app.py` and `predict.py` import it and
add no model code of their own.

| Function | Returns |
|---|---|
| `load()` | Loads weights, installs both hooks, warms the GPU |
| `answer(image)` | Class, confidence, all 8 scores, milliseconds, crop, overlay |
| `batch_answer(images)` | One forward per chunk of 64, verdicts plus total milliseconds |
| `side_by_side(result)` | Crop and overlay in one array |
| `pretty(name)` | `Apple_Black_rot` to `Apple - Black rot` |

`load()` runs one forward on random noise before returning. Without that
warm-up the first answer reports roughly 1800 ms instead of roughly 10 ms, which
ruins the latency readout in a recording.

`batch_answer` reads logits from the hook, not from the model return value,
because the module does not always return a plain tensor. Chunks are capped at
64 images so a large drop cannot exhaust 4 GB.

Weights resolve from `$LEAF_WEIGHTS`, else the newest
`runs/classify/train*/weights/best.pt`.

## Measured performance

| Path | Result |
|---|---|
| One leaf, in process | 7 to 13 ms |
| One leaf, through the Gradio HTTP API | 34 ms |
| 12 leaves, batch | 0.04 s, 3 ms each |
| 64 leaves, batch | 0.12 s, 2 ms each |
| Batch agreement with folder labels | 64 of 64 |

Batch timing includes JPEG decode on the CPU, which dominates it. That is
deliberate: it is the time the user actually waits.

## Web app

`app.py`, Gradio, 2 tabs.

- **One leaf.** Verdict and confidence as Markdown headings, which gives large
  text with no CSS file. The crop and the heatmap arrive as one side-by-side
  image, so the 2 always line up. `gr.Label(num_top_classes=8)` draws a bar for
  each class.
- **Many leaves.** `gr.File(file_count="multiple")` into `batch_answer`, then a
  gallery with a verdict caption per photo and a summary line.

Gradio 6 moved `theme` from the `Blocks` constructor to `launch()`. Passing it
to `Blocks` only raises a `UserWarning` and is ignored.

The model loads once at import, not per request.

## Things left undone

- No class rebalancing.
- No field photographs, so no measure of real-world accuracy.
- No test suite. The asserts inside `split.py` are the only automated check.
- The heatmap is 7x7 upscaled to 224. `Grape_Black_rot` is the weakest case,
  where the heat sits centre while the lesions are near the leaf edge.
