# Architecture

4 scripts and one module. No package, no framework, no configuration system.

```
images/<class>/*.JPG          the dataset, never modified, never committed
    |
    v  split.py               SHA-256 dedupe, then stratified 80/10/10
data/{train,val,test}/<class>/   absolute symlinks back into images/
    |
    v  train.py               Ultralytics YOLO11n-cls, 224 px, AMP
runs/classify/train/weights/best.pt
    |
    v  leaf.py                the only inference path
    |                         load, answer, batch_answer, side_by_side, pretty
    +---> app.py              Gradio, 2 tabs: one leaf, many leaves
    +---> predict.py          the command line
```

## Why it is shaped this way

`leaf.py` holds every piece of model code: weight resolution, preprocessing,
both forward hooks, the Grad-CAM arithmetic, and the plain-words label map.
`app.py` and `predict.py` contain presentation only. A change to the model or
the heatmap touches exactly one file.

`split.py` writes symlinks rather than copies, so the 129 MB dataset is never
duplicated on disk and the originals cannot be modified by accident.

## Data flow at inference

1. The photo arrives as a path, a numpy array, or a PIL image.
2. `classify_transforms(224)` resizes the short side and crops the centre.
3. A forward pass through the raw `nn.Module` fills 2 hooks: the activation of
   the `Classify` head `Conv` block, and the pre-softmax linear output.
4. The winning logit is backpropagated. The channel gradients weight the
   activation, which becomes the heatmap.
5. The heatmap is drawn on the centre crop, not on the original photo, because
   the crop is what the model saw.

`DEV_DOC.md` records the 3 Ultralytics traps this flow works around.
