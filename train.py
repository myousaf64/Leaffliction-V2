#!/usr/bin/env python3
"""Train a YOLO classify model on data/ and report the test split score.

The script stops if CUDA is not available. It never falls back to the CPU.
If the GPU runs out of memory, the script halves the batch size and retries.
"""

import sys
from pathlib import Path

import torch
from ultralytics import YOLO

MODEL = "yolo11n-cls.pt"
DATA = Path("data").resolve()
IMGSZ = 224
EPOCHS = 30
START_BATCH = 32
MIN_BATCH = 4
SEED = 42


def require_gpu():
    """Stop if CUDA is not available."""
    if not torch.cuda.is_available():
        sys.exit(
            "STOP: torch.cuda.is_available() is False.\n"
            "Run nvidia-smi. Boot a generic kernel, not the gke kernel.\n"
            "This script does not run on the CPU."
        )
    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name}, {props.total_memory / 1e9:.2f} GB")


def is_out_of_memory(error):
    """Return True if the error is a CUDA out-of-memory error."""
    if isinstance(error, torch.cuda.OutOfMemoryError):
        return True
    return isinstance(error, RuntimeError) and "out of memory" in str(error).lower()


def train():
    """Train the model. Halve the batch size on an out-of-memory error."""
    batch = START_BATCH
    while batch >= MIN_BATCH:
        print(f"\n=== train: batch {batch}, imgsz {IMGSZ}, amp on ===")
        model = YOLO(MODEL)
        try:
            model.train(
                data=DATA,
                imgsz=IMGSZ,
                epochs=EPOCHS,
                batch=batch,
                device=0,
                amp=True,
                seed=SEED,
                plots=True,
            )
            print(f"\nbatch {batch} fits in this GPU")
            return model, batch
        except Exception as error:  # noqa: BLE001 - retry only on out of memory
            if not is_out_of_memory(error):
                raise
            print(f"out of memory at batch {batch}")
            del model
            torch.cuda.empty_cache()
            batch //= 2
    sys.exit(f"out of memory even at batch {MIN_BATCH}")


def per_class_recall(model):
    """Print the recall of each class on the test split."""
    name_to_index = {name: index for index, name in model.names.items()}
    print(f"\n{'class':<17}  recall   correct/total")
    worst = None
    for folder in sorted((DATA / "test").iterdir()):
        files = sorted(folder.iterdir())
        truth = name_to_index[folder.name]
        results = model.predict(files, imgsz=IMGSZ, device=0, verbose=False)
        correct = sum(1 for r in results if int(r.probs.top1) == truth)
        recall = correct / len(files)
        print(f"{folder.name:<17}  {recall:6.1%}   {correct}/{len(files)}")
        if worst is None or recall < worst[1]:
            worst = (folder.name, recall)
    print(f"weakest class: {worst[0]} at {worst[1]:.1%}")


def main():
    require_gpu()
    model, batch = train()

    print("\n=== test split ===")
    metrics = model.val(split="test", imgsz=IMGSZ, device=0)
    print(f"test top-1: {metrics.top1:.4f}")
    print(f"test top-5: {metrics.top5:.4f}")
    per_class_recall(model)
    print(f"\nbatch size used: {batch}")
    print(f"weights: {model.trainer.best}")


if __name__ == "__main__":
    main()
