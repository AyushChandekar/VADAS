"""
Verify dataset integrity for both detection and segmentation.

Usage:
    python -m data.verify_dataset --type detection --path path/to/yolo_dataset
    python -m data.verify_dataset --type segmentation --path path/to/IDD_Segmentation
"""

import argparse
import os
import glob
from pathlib import Path
from collections import Counter


def verify_yolo_detection(path: str):
    """Verify YOLO-format detection dataset."""
    print(f"Verifying YOLO detection dataset: {path}")

    for split in ["train", "val", "test"]:
        img_dir = os.path.join(path, "images", split)
        lbl_dir = os.path.join(path, "labels", split)

        if not os.path.exists(img_dir):
            print(f"  WARNING: {img_dir} does not exist")
            continue

        images = glob.glob(os.path.join(img_dir, "*"))
        labels = glob.glob(os.path.join(lbl_dir, "*.txt"))

        img_stems = {Path(p).stem for p in images}
        lbl_stems = {Path(p).stem for p in labels}

        missing_labels = img_stems - lbl_stems
        missing_images = lbl_stems - img_stems

        # Count classes
        class_counts = Counter()
        total_boxes = 0
        for lbl in labels:
            with open(lbl) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_counts[int(parts[0])] += 1
                        total_boxes += 1

        print(f"\n  {split}:")
        print(f"    Images: {len(images)}")
        print(f"    Labels: {len(labels)}")
        print(f"    Total boxes: {total_boxes}")
        print(f"    Missing labels: {len(missing_labels)}")
        print(f"    Missing images: {len(missing_images)}")
        print(f"    Class distribution:")
        for cls_id in sorted(class_counts):
            print(f"      {cls_id}: {class_counts[cls_id]}")

    # Check data.yaml
    yaml_path = os.path.join(path, "data.yaml")
    if os.path.exists(yaml_path):
        print(f"\n  data.yaml: EXISTS")
    else:
        print(f"\n  WARNING: data.yaml NOT FOUND")


def verify_segmentation(path: str):
    """Verify IDD segmentation dataset."""
    print(f"Verifying IDD segmentation dataset: {path}")

    left_imgs = glob.glob(os.path.join(path, "**/*_leftImg8bit.*"), recursive=True)
    polygons = glob.glob(os.path.join(path, "**/*_gtFine_polygons.json"), recursive=True)
    label_imgs = glob.glob(os.path.join(path, "**/*_gtFine_labellevel3Ids.png"), recursive=True)

    print(f"\n  Images (_leftImg8bit): {len(left_imgs)}")
    print(f"  Polygon JSONs: {len(polygons)}")
    print(f"  Label ID PNGs: {len(label_imgs)}")

    # Check splits
    for split in ["train", "val", "test"]:
        split_imgs = [p for p in left_imgs if f"/{split}/" in p.lower()]
        print(f"  {split} images: {len(split_imgs)}")

    if left_imgs:
        print(f"\n  Sample image: {left_imgs[0]}")
    if polygons:
        print(f"  Sample JSON:  {polygons[0]}")


def main():
    parser = argparse.ArgumentParser(description="Dataset verification")
    parser.add_argument("--type", choices=["detection", "segmentation"], required=True)
    parser.add_argument("--path", required=True, help="Dataset path")
    args = parser.parse_args()

    if args.type == "detection":
        verify_yolo_detection(args.path)
    else:
        verify_segmentation(args.path)

    print("\nVerification complete.")


if __name__ == "__main__":
    main()
