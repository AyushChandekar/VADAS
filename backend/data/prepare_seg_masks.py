"""
Pre-render binary drivable masks from IDD Segmentation annotations.

Usage:
    python -m data.prepare_seg_masks \
        --input  path/to/IDD_Segmentation \
        --output path/to/drivable_masks

Supports both:
  - JSON polygon annotations (*_gtFine_polygons.json)
  - Label ID images (*_gtFine_labellevel3Ids.png)
"""

import argparse
import os
import glob
import json
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw

from data.idd_classes import DRIVABLE_LEVEL3_IDS, DRIVABLE_POLYGON_NAMES


def render_from_json(json_path: str, img_w: int, img_h: int) -> np.ndarray:
    """Render binary mask from polygon JSON."""
    with open(json_path) as f:
        data = json.load(f)

    w = data.get("imgWidth", img_w)
    h = data.get("imgHeight", img_h)

    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)

    for obj in data.get("objects", []):
        label = obj.get("label", "").lower().strip()
        if label in DRIVABLE_POLYGON_NAMES:
            polygon = obj.get("polygon", [])
            if len(polygon) >= 3:
                pts = [(p[0], p[1]) for p in polygon]
                draw.polygon(pts, fill=255)

    return np.array(mask)


def render_from_label_img(label_path: str) -> np.ndarray:
    """Render binary mask from labellevel3Ids PNG."""
    arr = np.array(Image.open(label_path))
    mask = np.zeros_like(arr, dtype=np.uint8)
    for lid in DRIVABLE_LEVEL3_IDS:
        mask[arr == lid] = 255
    return mask


def main():
    parser = argparse.ArgumentParser(description="IDD Segmentation → binary drivable masks")
    parser.add_argument("--input", required=True, help="Path to IDD_Segmentation/ or idd20kII/")
    parser.add_argument("--output", required=True, help="Output directory for masks")
    args = parser.parse_args()

    # Find annotations
    polygons = glob.glob(os.path.join(args.input, "**/*_gtFine_polygons.json"), recursive=True)
    label_imgs = glob.glob(os.path.join(args.input, "**/*_gtFine_labellevel3Ids.png"), recursive=True)

    print(f"Found {len(polygons)} polygon JSONs, {len(label_imgs)} label images")

    use_json = len(polygons) > 0
    annotations = polygons if use_json else label_imgs
    print(f"Using: {'JSON polygons' if use_json else 'label ID images'}")

    # Find corresponding images
    left_imgs = glob.glob(os.path.join(args.input, "**/*_leftImg8bit.jpg"), recursive=True)
    left_imgs += glob.glob(os.path.join(args.input, "**/*_leftImg8bit.png"), recursive=True)
    img_by_stem = {}
    for p in left_imgs:
        stem = Path(p).stem.replace("_leftImg8bit", "")
        img_by_stem[stem] = p

    rendered = 0
    for annot_path in annotations:
        stem = Path(annot_path).stem
        for suffix in ["_gtFine_polygons", "_gtFine_labellevel3Ids"]:
            if stem.endswith(suffix):
                stem = stem[:-len(suffix)]
                break

        # Detect split from path
        path_lower = annot_path.lower()
        if "/train/" in path_lower:
            split = "train"
        elif "/val/" in path_lower:
            split = "val"
        elif "/test/" in path_lower:
            split = "test"
        else:
            split = "train"

        out_dir = os.path.join(args.output, split)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{stem}_drivable.png")

        if os.path.exists(out_path):
            rendered += 1
            continue

        if use_json:
            img_path = img_by_stem.get(stem)
            if img_path:
                img = Image.open(img_path)
                mask = render_from_json(annot_path, img.size[0], img.size[1])
            else:
                mask = render_from_json(annot_path, 1920, 1080)
        else:
            mask = render_from_label_img(annot_path)

        Image.fromarray(mask).save(out_path)
        rendered += 1

        if rendered % 500 == 0:
            print(f"  Rendered {rendered}/{len(annotations)} masks...")

    print(f"\nDone! Rendered {rendered} masks to {args.output}")


if __name__ == "__main__":
    main()
