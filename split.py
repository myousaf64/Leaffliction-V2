#!/usr/bin/env python3
"""Count the leaf dataset, remove duplicate images, and split it 80/10/10.

The split uses symlinks. This script never moves or writes a file in images/.
It removes duplicates before it splits. A duplicate image in two splits
inflates the test score.
"""

import hashlib
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

SRC = Path("images")
DST = Path("data")
SEED = 42
VAL_FRAC = 0.1
TEST_FRAC = 0.1
PARTS = ("train", "val", "test")


def sha256(path):
    """Return the SHA-256 hex digest of one file."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_classes():
    """Return {class name: [file path]} for each directory in images/."""
    out = {}
    for folder in sorted(SRC.iterdir()):
        if folder.is_dir():
            out[folder.name] = sorted(p for p in folder.iterdir() if p.is_file())
    return out


def print_counts(by_class):
    """Print the image count of each class. Return the total."""
    rows = sorted(by_class.items(), key=lambda kv: -len(kv[1]))
    total = sum(len(v) for v in by_class.values())
    width = max(len(name) for name in by_class)
    print(f"{'class':<{width}}  images")
    for name, paths in rows:
        print(f"{name:<{width}}  {len(paths):>6}")
    print(f"{'TOTAL':<{width}}  {total:>6}")
    most, least = len(rows[0][1]), len(rows[-1][1])
    print(f"imbalance {most / least:.1f}:1  ({rows[0][0]} vs {rows[-1][0]})")
    return total


def dedupe(by_class):
    """Keep one path for each hash. Return the kept files and the duplicates."""
    seen = {}
    kept = defaultdict(list)
    dropped = 0
    conflicts = []
    for name in sorted(by_class):
        for path in by_class[name]:
            digest = sha256(path)
            if digest in seen:
                dropped += 1
                first_class, first_path = seen[digest]
                if first_class != name:
                    conflicts.append((first_path, path))
                continue
            seen[digest] = (name, path)
            kept[name].append((digest, path))
    return kept, dropped, conflicts


def make_splits(kept):
    """Split each class 80/10/10 with a fixed seed."""
    rng = random.Random(SEED)
    out = {part: [] for part in PARTS}
    for name in sorted(kept):
        items = sorted(kept[name])
        rng.shuffle(items)
        count = len(items)
        n_val = round(count * VAL_FRAC)
        n_test = round(count * TEST_FRAC)
        n_train = count - n_val - n_test
        if n_train <= 0:
            sys.exit(f"class {name} has only {count} images. It is too small to split.")
        out["train"] += [(name, d, p) for d, p in items[:n_train]]
        out["val"] += [(name, d, p) for d, p in items[n_train:n_train + n_val]]
        out["test"] += [(name, d, p) for d, p in items[n_train + n_val:]]
    return out


def link(splits):
    """Create one symlink for each image under data/<part>/<class>/."""
    for part, rows in splits.items():
        for name, _digest, path in rows:
            folder = DST / part / name
            folder.mkdir(parents=True, exist_ok=True)
            os.symlink(path.resolve(), folder / path.name)


def check(splits):
    """Stop if one path or one hash appears in two splits."""
    paths = {p: {str(r[2].resolve()) for r in rows} for p, rows in splits.items()}
    hashes = {p: {r[1] for r in rows} for p, rows in splits.items()}
    for i, first in enumerate(PARTS):
        for second in PARTS[i + 1:]:
            shared_path = paths[first] & paths[second]
            shared_hash = hashes[first] & hashes[second]
            if shared_path:
                sys.exit(f"FAIL: {len(shared_path)} paths are in both {first} and {second}")
            if shared_hash:
                sys.exit(f"FAIL: {len(shared_hash)} hashes are in both {first} and {second}")
    print("checks passed: no path and no hash is in two splits")


def print_splits(kept, splits):
    """Print the image count of each class in each split."""
    per = {part: defaultdict(int) for part in PARTS}
    for part, rows in splits.items():
        for name, _digest, _path in rows:
            per[part][name] += 1
    width = max(len(name) for name in kept)
    print(f"{'class':<{width}}  {'train':>6} {'val':>5} {'test':>5}")
    for name in sorted(kept):
        print(f"{name:<{width}}  {per['train'][name]:>6} "
              f"{per['val'][name]:>5} {per['test'][name]:>5}")
    print(f"{'TOTAL':<{width}}  {len(splits['train']):>6} "
          f"{len(splits['val']):>5} {len(splits['test']):>5}")


def main():
    if not SRC.is_dir():
        sys.exit(f"no {SRC}/ directory in {Path.cwd()}")
    if DST.exists():
        sys.exit(f"{DST}/ already exists. Remove it first:\n\n    rm -rf {DST}\n")

    by_class = read_classes()
    if not by_class:
        sys.exit(f"{SRC}/ holds no class directory")

    print("== dataset ==")
    total = print_counts(by_class)

    print("\n== duplicates ==")
    kept, dropped, conflicts = dedupe(by_class)
    n_kept = sum(len(v) for v in kept.values())
    print(f"hashed {total} files, dropped {dropped} duplicates, kept {n_kept}")
    print(f"cross-class duplicates (label conflicts): {len(conflicts)}")
    for first, second in conflicts[:5]:
        print(f"  {first}  ==  {second}")
    if len(conflicts) > 5:
        print(f"  ... and {len(conflicts) - 5} more")

    print("\n== split 80/10/10, seed 42 ==")
    splits = make_splits(kept)
    link(splits)
    check(splits)
    print_splits(kept, splits)


if __name__ == "__main__":
    main()
