"""
Convert IDD Detection XML annotations to YOLO format.

Usage:
    python -m data.prepare_yolo_labels \
        --input  path/to/IDD_Detection \
        --output path/to/yolo_dataset

IDD Detection structure expected:
    IDD_Detection/
        JPEGImages/.../*.jpg
        Annotations/.../*.xml
        train.txt, val.txt, test.txt
"""

import argparse
import os
import glob
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict

import yaml

from data.idd_classes import DETECTION_NAME_MAP, DETECTION_CLASSES


def parse_xml(xml_path: str, class_to_id: dict) -> tuple:
    """Parse IDD VOC XML → (img_w, img_h, list of (cls_id, xc, yc, w, h))."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find("size")
    if size is None:
        return None, None, []

    img_w = int(size.find("width").text)
    img_h = int(size.find("height").text)
    if img_w == 0 or img_h == 0:
        return None, None, []

    boxes = []
    for obj in root.findall("object"):
        name_el = obj.find("name")
        if name_el is None or name_el.text is None:
            continue

        raw = name_el.text.strip().lower()
        canonical = DETECTION_NAME_MAP.get(raw)
        if canonical is None or canonical not in class_to_id:
            continue

        bndbox = obj.find("bndbox")
        if bndbox is None:
            continue

        try:
            xmin = float(bndbox.find("xmin").text)
            ymin = float(bndbox.find("ymin").text)
            xmax = float(bndbox.find("xmax").text)
            ymax = float(bndbox.find("ymax").text)
        except (TypeError, ValueError):
            continue

        xc = max(0, min(1, ((xmin + xmax) / 2) / img_w))
        yc = max(0, min(1, ((ymin + ymax) / 2) / img_h))
        w = max(0, min(1, (xmax - xmin) / img_w))
        h = max(0, min(1, (ymax - ymin) / img_h))

        if w > 0.001 and h > 0.001:
            boxes.append((class_to_id[canonical], xc, yc, w, h))

    return img_w, img_h, boxes


def main():
    parser = argparse.ArgumentParser(description="IDD Detection → YOLO converter")
    parser.add_argument("--input", required=True, help="Path to IDD_Detection/")
    parser.add_argument("--output", required=True, help="Output YOLO dataset dir")
    args = parser.parse_args()

    class_to_id = {name: i for i, name in enumerate(DETECTION_CLASSES)}
    print(f"Classes ({len(DETECTION_CLASSES)}):")
    for i, name in enumerate(DETECTION_CLASSES):
        print(f"  {i}: {name}")

    # Create output dirs
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(args.output, "images", split), exist_ok=True)
        os.makedirs(os.path.join(args.output, "labels", split), exist_ok=True)

    # Find images and XMLs
    images = glob.glob(os.path.join(args.input, "**/*.jpg"), recursive=True)
    images += glob.glob(os.path.join(args.input, "**/*.png"), recursive=True)

    xmls = glob.glob(os.path.join(args.input, "**/*.xml"), recursive=True)
    xml_by_stem = {Path(x).stem: x for x in xmls}

    # Load split files
    split_map = {}
    for split_name in ["train", "val", "test"]:
        txt = os.path.join(args.input, f"{split_name}.txt")
        if os.path.exists(txt):
            with open(txt) as f:
                for line in f:
                    stem = line.strip()
                    if stem:
                        split_map[stem] = split_name
            print(f"Loaded {split_name}.txt: {sum(1 for v in split_map.values() if v == split_name)} entries")

    counts = defaultdict(int)
    total_boxes = 0

    for img_path in images:
        stem = Path(img_path).stem
        xml_path = xml_by_stem.get(stem)
        if xml_path is None:
            continue

        _, _, boxes = parse_xml(xml_path, class_to_id)
        if not boxes:
            continue

        split = split_map.get(stem)
        if split is None:
            h = hash(stem) % 100
            split = "train" if h < 85 else ("val" if h < 95 else "test")

        # Copy image
        parent = Path(img_path).parent.name
        ext = Path(img_path).suffix
        unique = f"{parent}_{stem}{ext}"
        dest = os.path.join(args.output, "images", split, unique)
        if not os.path.exists(dest):
            shutil.copy2(img_path, dest)

        # Write label
        label_path = os.path.join(args.output, "labels", split, f"{parent}_{stem}.txt")
        with open(label_path, "w") as f:
            for cid, xc, yc, w, h in boxes:
                f.write(f"{cid} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

        total_boxes += len(boxes)
        counts[split] += 1

    # Write data.yaml
    data_yaml = {
        "path": os.path.abspath(args.output),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(DETECTION_CLASSES),
        "names": DETECTION_CLASSES,
    }
    yaml_path = os.path.join(args.output, "data.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(data_yaml, f, default_flow_style=False)

    print(f"\nDone! Train: {counts['train']}, Val: {counts['val']}, Test: {counts['test']}")
    print(f"Total boxes: {total_boxes}")
    print(f"Config: {yaml_path}")


if __name__ == "__main__":
    main()
