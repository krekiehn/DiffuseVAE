"""Preprocessing utilities for CheXpert chest X-ray images.

This module implements a common pipeline described in literature for CheXpert:
1. convert to grayscale
2. apply CLAHE (contrast limited adaptive histogram equalisation)
3. resize images to a fixed square size.
"""

import argparse
import os
from pathlib import Path

import cv2
from PIL import Image
import numpy as np
from tqdm import tqdm


def preprocess_image(img: Image.Image, size: int = 224, apply_clahe: bool = True) -> Image.Image:
    """Preprocess a single PIL image.

    Steps are gathered from common CheXpert preprocessing routines in the
    literature: convert to grayscale, optionally apply CLAHE to improve
    contrast, resize and center crop to the desired size.
    """
    img = img.convert("L")
    np_img = np.array(img)
    if apply_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        np_img = clahe.apply(np_img)
    img = Image.fromarray(np_img)
    # Resize and center crop
    img = img.resize((size, size), Image.BILINEAR)
    return img


def process_directory(src_root: str, dst_root: str, size: int = 224, apply_clahe: bool = True):
    src_root = Path(src_root)
    dst_root = Path(dst_root)
    dst_root.mkdir(parents=True, exist_ok=True)

    images = []
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        images.extend(src_root.rglob(ext))

    for img_path in tqdm(images, desc="Preprocessing"):
        img = Image.open(img_path)
        processed = preprocess_image(img, size=size, apply_clahe=apply_clahe)
        rel = img_path.relative_to(src_root)
        out_path = dst_root / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        processed.save(out_path.with_suffix(".png"))


def main():
    parser = argparse.ArgumentParser(description="Preprocess CheXpert images")
    parser.add_argument("src", type=str, help="Directory with original images")
    parser.add_argument("dst", type=str, help="Output directory for processed images")
    parser.add_argument("--size", type=int, default=224, help="Output image size")
    parser.add_argument("--no-clahe", action="store_true", help="Disable CLAHE contrast enhancement")
    args = parser.parse_args()

    process_directory(args.src, args.dst, size=args.size, apply_clahe=not args.no_clahe)


if __name__ == "__main__":
    main()
