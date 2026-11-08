# Leaffliction v2

Drop in a photo of an apple or grape leaf. The model names the disease, says how sure it is, and
paints a heatmap over the exact pixels it used to decide.

![demo](./docs/demo.gif)
<!-- PLACEHOLDER: docs/demo.gif does not exist yet. Record it with the shot list in LAUNCH.md
     (run `python linkedin.py` to generate that file), then drop the GIF at docs/demo.gif. -->

8 classes: apple black rot, apple healthy, apple rust, apple scab, grape black rot, grape esca,
grape healthy, grape leaf spot.

| | |
|---|---|
| Test accuracy | **99.54%** on 1,087 held-out images |
| Speed | 2.05 ms per image on a laptop RTX 3050 Ti, 6.78 ms on CPU |
| Model | yolo11n-cls, 3.2 MB, 1,541,352 parameters, 224 px input |
| Explains itself | Grad-CAM on every prediction, 79% of the heat inside the leaf |
| Honest about doubt | under 40% confidence it answers "uncertain", not a guess |

## Run the demo

The trained weights are committed, so this works from a cold clone. No training step.

```bash
git clone https://github.com/myousaf64/Leaffliction-V2.git
cd Leaffliction-V2
python3.10 -m venv .venv && .venv/bin/pip install -r requirements.txt
./run_demo.sh
```

`run_demo.sh` starts the backend, waits for it to answer, and opens
<http://127.0.0.1:8000> in your browser.

In the page:

- Drag one leaf photo onto the drop zone, or drag a dozen. Click it to pick files instead.
  On a phone the picker offers the camera.
- **Choose folder** points it at a whole directory and classifies every image inside.
  Chrome and Edge support this; Firefox ignores it, so drag the files in there.
- With more than one photo you get a filmstrip and a summary line:
  `8 leaves · 5 diseased · 3 healthy · 446 ms total, 6.5 ms per leaf`. Each thumbnail carries a
  forest dot for healthy, terracotta for diseased, a hollow ring for uncertain.
- Click a thumbnail, or use `←` and `→`, to put any photo in the large view.
- The slider fades the Grad-CAM heatmap in and out.
- `H` toggles the heatmap. `Space` picks new photos.

Batch limits: 32 images per request, 64 MB in total. A file that is not a readable image is
reported on its own thumbnail and does not spoil the rest of the batch.

Use a different port with `PORT=8080 ./run_demo.sh`.

### What to expect on the first run

- `pip install` pulls roughly 3 GB of CUDA wheels. It takes a few minutes.
- No NVIDIA GPU is required. Without one, PyTorch falls back to CPU and inference runs at about
  6.8 ms per image, which is still instant in the browser.
- The server warms the model up at startup, so the first prediction is as fast as the rest.

### Requirements

| | |
|---|---|
| Python | 3.10 or newer |
| Disk | ~4 GB for the virtual environment, plus 129 MB if you also fetch the dataset |
| GPU | optional. NVIDIA with a recent driver for the 2 ms path |
| OS | developed and measured on Linux |

### Weights

`runs/cls/weights/best.pt` ships in the repository. The ONNX export is a release asset:

```bash
curl -L -o best.onnx \
  https://github.com/myousaf64/Leaffliction-V2/releases/download/v2.0/best.onnx
```

## Reproduce the model from scratch

You need the dataset, which is not in the repository.

### 1. Get the images

This uses the apple and grape subset of **PlantVillage** - the same `leaves.zip` the 42
Leaffliction subject distributes. It is also on Kaggle as *PlantVillage Dataset* (the colour
variant). Extract so the layout is one folder per class:

```
images/
  Apple_Black_rot/    image (1).JPG ...
  Apple_healthy/
  Apple_rust/
  Apple_scab/
  Grape_Black_rot/
  Grape_Esca/
  Grape_healthy/
  Grape_spot/
```

7,233 JPEG files, all 256x256. Class counts are uneven on purpose, from 281 to 1,640.

### 2. Run the chain

```bash
.venv/bin/python pipeline.py split     # 70/15/15 + offline balance  -> splits_report.txt
.venv/bin/python pipeline.py train     # fine-tune yolo11n-cls        -> runs/cls/weights/best.pt
.venv/bin/python pipeline.py eval      # test split only              -> metrics.json
.venv/bin/python pipeline.py cam       # Grad-CAM layer probe         -> assets/cam/
.venv/bin/python pipeline.py export    # ONNX                         -> runs/cls/weights/best.onnx
.venv/bin/python assets.py             # 6 PNGs + carousel.pdf        -> assets/
.venv/bin/python linkedin.py           # launch playbook              -> LAUNCH.md
```

Measured on an RTX 3050 Ti Laptop (4 GB): split about 90 seconds, training 22 epochs in roughly
5 minutes at batch 32, evaluation under a minute. Training stops early on patience 10, so the epoch
count varies with the seed.

`pipeline.py self-check` runs the assertions on the pure logic and needs no weights and no dataset.

## Results

Measured on the test split only, which is never seen in training or validation.
Every number here is read from `metrics.json`.

| | |
|---|---|
| Test accuracy | **99.54%** (1082 / 1087) |
| Top-5 accuracy | 100% |
| GPU | 2.05 ms per image (RTX 3050 Ti Laptop, batch 1) |
| CPU | 6.78 ms per image (batch 1) |
| Model | yolo11n-cls, 3.2 MB, 1,541,352 parameters, 224px input |

Per class:

| class | precision | recall | F1 | test images |
|---|---|---|---|---|
| Apple black rot | 1.0000 | 1.0000 | 1.0000 | 95 |
| Apple healthy | 0.9960 | 1.0000 | 0.9980 | 246 |
| Apple rust | 1.0000 | 1.0000 | 1.0000 | 39 |
| Apple scab | 1.0000 | 0.9895 | 0.9947 | 95 |
| Grape black rot | 0.9780 | 1.0000 | 0.9889 | 178 |
| Grape esca | 1.0000 | 0.9856 | 0.9927 | 208 |
| Grape healthy | 1.0000 | 0.9844 | 0.9921 | 64 |
| Grape leaf spot | 1.0000 | 1.0000 | 1.0000 | 162 |

5 errors in 1,087 images. Grad-CAM puts 79% of its heat inside the leaf mask on average,
so the model reads the leaf, not the background.

## How the split is kept honest

The split runs before any augmentation, so no augmented copy can reach val or test.

`images/` still carries 12 augmented copies left over from Leaffliction v1, named
`image (100)_Rotate.JPG` beside their parent `image (100).JPG`. A naive per-file split put 2 of
them in the test set while their parent sat in training, and the model scored 99.91% partly on an
answer it had already been shown. The split is therefore stratified over parent groups:
`group_key()` strips the six v1 operation suffixes, and a whole group always lands in one split.
`pipeline.py split` asserts this and prints the group count.

The honest number after the fix is lower. It is also real.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `torch.cuda.is_available()` is False | No NVIDIA driver, or a driver older than the CUDA wheels. The demo still works on CPU. For training, fix the driver first - CPU training takes hours. |
| `Address already in use` on 8000 | Something else holds the port. Run `PORT=8080 ./run_demo.sh`. |
| `no weights. run: .venv/bin/python pipeline.py train` | `runs/cls/weights/best.pt` is missing. It ships in the repository, so this means it was deleted or the clone was shallow. Re-clone, or pull it from the v2.0 release. |
| `CUDA out of memory` during training | `pipeline.py train` retries at batch 16 then 8 by itself. If all three fail, close other GPU processes or pass `--batch 8`. |
| `missing dataset directory: .../images` | The dataset is not in the repository. See "Reproduce the model from scratch". |
| `missing metrics.json` from `assets.py` or `linkedin.py` | Run `pipeline.py eval` first. Both scripts refuse to invent numbers. |
| `33 files. The limit is 32 images at a time.` | Batch cap. Send fewer, or run two batches. |
| `The batch is over 64 MB. Send fewer photos.` | Total upload cap. Individual files are capped at 10 MB. |
| **Choose folder** does nothing | Firefox does not support `webkitdirectory`. Drag the files onto the drop zone instead. |
| The page loads but every upload fails | The backend died. Check the terminal running `run_demo.sh`. |

## Files

| file | what it does |
|---|---|
| `pipeline.py` | split, train, eval, cam, export. Also the shared module: `load_model`, `preprocess`, `predict`, `gradcam`, `leaf_mask` |
| `assets.py` | the six 1080x1350 PNGs and `carousel.pdf`, every number read from `metrics.json` |
| `linkedin.py` | generates `LAUNCH.md`: post copy, carousel order, recording shot list, checklist |
| `server.py` | FastAPI. `POST /predict` for one image, `POST /predict-batch` for up to 32, both returning top-3, a Grad-CAM overlay and inference ms |
| `index.html` | the demo page. One file, no framework, no build step |
| `run_demo.sh` | starts the backend, waits for it, opens the page |
| `splits_report.txt`, `metrics.json` | generated, the source of every published number |

## Notes

- Training uses offline balancing to 1,148 images per class on top of Ultralytics' built-in
  augmentation. Offline copies exist in the train split only.
- Grad-CAM hooks the 1280-channel conv inside the `Classify` head. `pipeline.py cam` also tries the
  last backbone block and keeps whichever puts more heat inside the leaf mask.
- Predictions below 40% confidence are returned as `uncertain`, so a photo that is not an apple or
  grape leaf does not get a confident diagnosis.

## Licence

Ultralytics YOLO11 is **AGPL-3.0**. Commercial use needs an Ultralytics Enterprise licence:
<https://www.ultralytics.com/license>. This repository's own code is subject to the same terms
while it imports Ultralytics.
