#!/usr/bin/env python3
"""Pre-process a real image (JPG/PNG/...) into a single-channel PGM for conv2d_run.

    python3 scripts/prepare_image.py photo.jpg data/photo.pgm [--resize 4096x4096]

The assignment allows converting inputs to grayscale before evaluation.
Requires Pillow (pip install pillow).
"""
import argparse

from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--resize", help="WxH, e.g. 4096x4096")
    a = ap.parse_args()
    img = Image.open(a.src).convert("L")
    if a.resize:
        w, h = (int(v) for v in a.resize.lower().split("x"))
        img = img.resize((w, h), Image.BICUBIC)
    img.save(a.dst, format="PPM")  # 'L' mode + PPM writer = binary P5 PGM
    print(f"wrote {a.dst} ({img.width}x{img.height}, grayscale)")


if __name__ == "__main__":
    main()
