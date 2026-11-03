#!/usr/bin/env python3
"""Classify one leaf photo from the command line.

    python predict.py <image>

It prints the disease, the confidence, and every class score. It saves the
heatmap beside the crop as <name>_cam.png in the current directory.
"""

import sys
from pathlib import Path

import cv2

import leaf


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: python predict.py <image>")

    source = Path(sys.argv[1])
    if not source.is_file():
        sys.exit(f"no such file: {source}")

    result = leaf.answer(source)

    print(f"\n{result['pretty']}")
    print(f"{result['confidence']:.1%} confident")
    print(f"answered in {result['ms']:.0f} ms\n")
    for name, score in sorted(result["scores"].items(), key=lambda kv: -kv[1]):
        bar = "#" * round(score * 30)
        print(f"  {name:<22} {score:6.1%}  {bar}")

    # Saved in the current directory, never next to the input. A PNG dropped
    # inside images/<class>/ becomes training data on the next split.
    dest = Path(f"{source.stem}_cam.png")
    cv2.imwrite(str(dest), cv2.cvtColor(leaf.side_by_side(result), cv2.COLOR_RGB2BGR))
    print(f"\nsaved {dest}  (left: what the model saw, right: where it looked)")


if __name__ == "__main__":
    main()
