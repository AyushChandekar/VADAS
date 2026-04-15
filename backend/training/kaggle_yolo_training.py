"""
KAGGLE NOTEBOOK — YOLOv8s Fine-tuning on IDD Detection
=======================================================
Dataset: kaggle.com/datasets/vinayak21574/idd-detection
Mount:   /kaggle/input/idd-detection/

Instructions:
  1. Create new Kaggle notebook
  2. Settings → Accelerator → GPU T4 x2
  3. Settings → Internet → ON
  4. Add Input → search "idd-detection" by vinayak21574 → Add
  5. Paste this ENTIRE file as a single cell
  6. Click "Save & Run All" (dropdown, NOT the Run button)
  7. Close browser / shut down laptop — runs up to 12 hours
  8. Tomorrow: Output tab → download yolo_idd_best.pt
"""

# ═══════════════════════════════════════════════════════════════════════
# 0. INSTALL
# ═══════════════════════════════════════════════════════════════════════
import subprocess
subprocess.run(["pip", "install", "ultralytics", "-q"])

import os
import glob
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict
import shutil
import yaml

KAGGLE_INPUT = "/kaggle/input"
WORK_DIR = "/kaggle/working"

# Auto-detect dataset directory (Kaggle mount names can vary)
_candidates = [d for d in os.listdir(KAGGLE_INPUT)
               if os.path.isdir(os.path.join(KAGGLE_INPUT, d)) and "detect" in d.lower()]
if _candidates:
    DATASET_DIR = os.path.join(KAGGLE_INPUT, _candidates[0])
else:
    # Fallback: use first directory in /kaggle/input/
    _all_dirs = [d for d in os.listdir(KAGGLE_INPUT) if os.path.isdir(os.path.join(KAGGLE_INPUT, d))]
    DATASET_DIR = os.path.join(KAGGLE_INPUT, _all_dirs[0]) if _all_dirs else KAGGLE_INPUT

print(f"Dataset directory: {DATASET_DIR}")
print(f"Contents: {os.listdir(DATASET_DIR)}")

# ═══════════════════════════════════════════════════════════════════════
# 1. DEEP SCAN — Print full directory tree (first 4 levels)
# ═══════════════════════════════════════════════════════════════════════
print("=" * 70)
print("STEP 1: FULL DIRECTORY TREE SCAN")
print("=" * 70)

def print_tree(root, max_depth=4, max_files=15):
    for dirpath, dirnames, filenames in os.walk(root):
        depth = dirpath.replace(root, "").count(os.sep)
        if depth >= max_depth:
            dirnames.clear()
            continue
        indent = "│   " * depth
        print(f"{indent}├── {os.path.basename(dirpath)}/")
        sub_indent = "│   " * (depth + 1)
        shown = filenames[:max_files]
        for f in shown:
            print(f"{sub_indent}├── {f}")
        if len(filenames) > max_files:
            print(f"{sub_indent}└── ... +{len(filenames) - max_files} more files")

print_tree(DATASET_DIR)

# ═══════════════════════════════════════════════════════════════════════
# 2. AUTO-DETECT: Find all images and annotations
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 2: AUTO-DETECTING IMAGES & ANNOTATIONS")
print("=" * 70)

# Find ALL images recursively
all_images = []
for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]:
    all_images.extend(glob.glob(f"{DATASET_DIR}/**/{ext}", recursive=True))

# Find ALL XML annotations (IDD Detection uses Pascal VOC XML)
all_xmls = glob.glob(f"{DATASET_DIR}/**/*.xml", recursive=True)

# Also check for YOLO txt labels (dataset might already be converted)
all_txts = glob.glob(f"{DATASET_DIR}/**/*.txt", recursive=True)
# Filter out non-label txts
label_txts = [t for t in all_txts if not any(x in t.lower() for x in ["readme", "license", "classes", "names"])]

# Check for JSON (COCO format)
all_jsons = glob.glob(f"{DATASET_DIR}/**/*.json", recursive=True)

print(f"Images found:     {len(all_images)}")
print(f"XML annotations:  {len(all_xmls)}")
print(f"TXT files:        {len(label_txts)}")
print(f"JSON files:       {len(all_jsons)}")

# Show sample paths
if all_images:
    print(f"\nSample images:")
    for p in all_images[:5]: print(f"  {p}")
if all_xmls:
    print(f"\nSample XMLs:")
    for p in all_xmls[:5]: print(f"  {p}")
if all_jsons:
    print(f"\nSample JSONs:")
    for p in all_jsons[:3]: print(f"  {p}")

# ═══════════════════════════════════════════════════════════════════════
# 3. READ ONE XML — Discover actual class names in this dataset
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 3: DISCOVERING CLASS NAMES FROM ANNOTATIONS")
print("=" * 70)

discovered_classes = defaultdict(int)

# Sample up to 500 XMLs to find all class names
sample_xmls = all_xmls[:500] if len(all_xmls) > 500 else all_xmls

for xml_path in sample_xmls:
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for obj in root.findall("object"):
            name_el = obj.find("name")
            if name_el is not None and name_el.text:
                discovered_classes[name_el.text.strip()] += 1
    except:
        pass

print("Classes found in XML annotations:")
for cls_name, count in sorted(discovered_classes.items(), key=lambda x: -x[1]):
    print(f"  {cls_name:30s} : {count} instances")

# Also print first XML content for debugging
if all_xmls:
    print(f"\nFull content of first XML ({all_xmls[0]}):")
    with open(all_xmls[0]) as f:
        print(f.read()[:2000])

# ═══════════════════════════════════════════════════════════════════════
# 4. BUILD CLASS MAPPING — Dynamically map discovered classes
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 4: BUILDING CLASS MAPPING")
print("=" * 70)

# IDD class mapping — handles various naming conventions
IDD_CLASS_MAP = {}

KNOWN_MAPPINGS = {
    # Vehicles
    "car": "car",
    "truck": "truck",
    "bus": "bus",
    "motorcycle": "motorcycle",
    "motorbike": "motorcycle",
    "bicycle": "bicycle",
    "autorickshaw": "autorickshaw",
    "auto rickshaw": "autorickshaw",
    "auto": "autorickshaw",
    "rickshaw": "autorickshaw",
    "vehicle fallback": "vehicle_fallback",
    "vehicle_fallback": "vehicle_fallback",
    "caravan": "vehicle_fallback",
    "trailer": "vehicle_fallback",
    # Vulnerable road users
    "person": "person",
    "pedestrian": "person",
    "rider": "rider",
    "animal": "animal",
    # Infrastructure
    "traffic sign": "traffic_sign",
    "traffic_sign": "traffic_sign",
    "trafficsign": "traffic_sign",
    "ts": "traffic_sign",
    "traffic light": "traffic_light",
    "traffic_light": "traffic_light",
    "trafficlight": "traffic_light",
    "tl": "traffic_light",
    # Other
    "pole": "pole",
    "wall": "wall",
    "fence": "fence",
    "guard rail": "fence",
    "guardrail": "fence",
}

# Build the final class list from discovered classes
final_classes = []
class_id_map = {}  # original_name -> class_id

for orig_name in discovered_classes:
    normalized = orig_name.lower().strip()
    mapped = KNOWN_MAPPINGS.get(normalized, None)
    if mapped and mapped not in final_classes:
        final_classes.append(mapped)

# If no known mappings found, use discovered classes directly
if len(final_classes) == 0:
    print("WARNING: No known IDD classes matched. Using discovered classes directly.")
    final_classes = list(discovered_classes.keys())

# Sort for consistency
final_classes.sort()

# Build ID map: original_annotation_name -> integer class id
for orig_name in discovered_classes:
    normalized = orig_name.lower().strip()
    mapped = KNOWN_MAPPINGS.get(normalized, None)
    if mapped and mapped in final_classes:
        class_id_map[orig_name] = final_classes.index(mapped)
    elif orig_name in final_classes:
        class_id_map[orig_name] = final_classes.index(orig_name)

NUM_CLASSES = len(final_classes)
print(f"\nFinal {NUM_CLASSES} classes:")
for i, name in enumerate(final_classes):
    print(f"  {i:2d}: {name}")

print(f"\nMapping (annotation name -> class ID):")
for orig, cid in sorted(class_id_map.items(), key=lambda x: x[1]):
    print(f"  '{orig}' -> {cid} ({final_classes[cid]})")

# ═══════════════════════════════════════════════════════════════════════
# 5. CONVERT XML TO YOLO FORMAT
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 5: CONVERTING TO YOLO FORMAT")
print("=" * 70)

def parse_xml_to_yolo(xml_path):
    """Parse IDD XML and return YOLO-format boxes."""
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
        name = name_el.text.strip()

        cid = class_id_map.get(name, None)
        if cid is None:
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

        xc = ((xmin + xmax) / 2.0) / img_w
        yc = ((ymin + ymax) / 2.0) / img_h
        w = (xmax - xmin) / img_w
        h = (ymax - ymin) / img_h

        xc = max(0, min(1, xc))
        yc = max(0, min(1, yc))
        w = max(0, min(1, w))
        h = max(0, min(1, h))

        if w > 0.001 and h > 0.001:
            boxes.append((cid, xc, yc, w, h))

    return img_w, img_h, boxes


# Create YOLO directory
YOLO_DIR = f"{WORK_DIR}/idd_yolo"
for split in ["train", "val", "test"]:
    os.makedirs(f"{YOLO_DIR}/images/{split}", exist_ok=True)
    os.makedirs(f"{YOLO_DIR}/labels/{split}", exist_ok=True)

# ── Build image <-> XML pairs ──────────────────────────────────────────
# Strategy: match by filename stem
xml_by_stem = {}
for xp in all_xmls:
    stem = Path(xp).stem
    xml_by_stem[stem] = xp

# Check for ImageSets (train.txt, val.txt, test.txt)
imagesets = {}
for txt_file in glob.glob(f"{DATASET_DIR}/**/ImageSets/**/*.txt", recursive=True):
    name = Path(txt_file).stem.lower()
    if name in ("train", "val", "trainval", "test"):
        with open(txt_file) as f:
            ids = [line.strip() for line in f if line.strip()]
        imagesets[name] = set(ids)
        print(f"Found ImageSet: {name} ({len(ids)} entries) at {txt_file}")

# Also check for split info in folder names
def detect_split(filepath, stem):
    """Detect train/val/test split."""
    path_lower = filepath.lower()

    # Check ImageSets first
    if imagesets:
        if stem in imagesets.get("train", set()) or stem in imagesets.get("trainval", set()):
            return "train"
        if stem in imagesets.get("val", set()):
            return "val"
        if stem in imagesets.get("test", set()):
            return "test"

    # Check directory names
    if "/train/" in path_lower or "\\train\\" in path_lower:
        return "train"
    if "/val/" in path_lower or "\\val\\" in path_lower:
        return "val"
    if "/test/" in path_lower or "\\test\\" in path_lower:
        return "test"

    return None

# ── Match and convert ──────────────────────────────────────────────────
split_counts = {"train": 0, "val": 0, "test": 0}
total_boxes = 0
skipped_no_xml = 0
skipped_no_boxes = 0

for img_path in all_images:
    stem = Path(img_path).stem

    # Try matching XML
    xml_path = xml_by_stem.get(stem)

    # Try common suffix removal
    if xml_path is None:
        for suffix in ["_leftImg8bit", "_image", "_raw"]:
            if stem.endswith(suffix):
                xml_path = xml_by_stem.get(stem[:-len(suffix)])
                if xml_path:
                    break

    if xml_path is None:
        skipped_no_xml += 1
        continue

    # Parse XML
    img_w, img_h, boxes = parse_xml_to_yolo(xml_path)
    if img_w is None or len(boxes) == 0:
        skipped_no_boxes += 1
        continue

    # Detect split
    split = detect_split(img_path, stem) or detect_split(xml_path, stem)
    if split is None:
        h = hash(stem) % 100
        split = "train" if h < 85 else ("val" if h < 95 else "test")

    # Create unique filename (prepend parent folder to avoid collisions)
    parent = Path(img_path).parent.name
    ext = Path(img_path).suffix
    unique_name = f"{parent}_{stem}{ext}"

    # Symlink image (fast, no disk copy)
    dest_img = f"{YOLO_DIR}/images/{split}/{unique_name}"
    if not os.path.exists(dest_img):
        try:
            os.symlink(img_path, dest_img)
        except OSError:
            shutil.copy2(img_path, dest_img)

    # Write YOLO label
    label_name = f"{parent}_{stem}.txt"
    with open(f"{YOLO_DIR}/labels/{split}/{label_name}", "w") as f:
        for cid, xc, yc, w, h in boxes:
            f.write(f"{cid} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

    total_boxes += len(boxes)
    split_counts[split] += 1

print(f"\nConversion Results:")
print(f"  Train images:  {split_counts['train']}")
print(f"  Val images:    {split_counts['val']}")
print(f"  Test images:   {split_counts['test']}")
print(f"  Total boxes:   {total_boxes}")
print(f"  Skipped (no matching XML): {skipped_no_xml}")
print(f"  Skipped (no valid boxes):  {skipped_no_boxes}")

# ── SAFETY CHECK ───────────────────────────────────────────────────────
total_images = sum(split_counts.values())
if total_images == 0:
    print("\n" + "!" * 70)
    print("ERROR: No image-annotation pairs found!")
    print("The dataset structure may be different than expected.")
    print("Check the tree printout above and adjust matching logic.")
    print("Printing all unique parent folder names for images:")
    img_parents = set(str(Path(p).parent) for p in all_images[:100])
    for p in sorted(img_parents): print(f"  {p}")
    xml_parents = set(str(Path(p).parent) for p in all_xmls[:100])
    print("Printing all unique parent folder names for XMLs:")
    for p in sorted(xml_parents): print(f"  {p}")
    print("!" * 70)
    raise SystemExit("Cannot proceed without data. Check output above.")

# If val set is empty, split from train
if split_counts["val"] == 0:
    print("\nWARNING: No validation split detected. Splitting 15% from train...")
    train_imgs = sorted(glob.glob(f"{YOLO_DIR}/images/train/*"))
    n_val = int(len(train_imgs) * 0.15)
    for img_f in train_imgs[:n_val]:
        fname = os.path.basename(img_f)
        lbl_fname = Path(fname).stem + ".txt"
        shutil.move(img_f, f"{YOLO_DIR}/images/val/{fname}")
        lbl_src = f"{YOLO_DIR}/labels/train/{lbl_fname}"
        if os.path.exists(lbl_src):
            shutil.move(lbl_src, f"{YOLO_DIR}/labels/val/{lbl_fname}")
    split_counts["val"] = n_val
    split_counts["train"] -= n_val
    print(f"  Moved {n_val} images to val. Train: {split_counts['train']}, Val: {split_counts['val']}")

# ═══════════════════════════════════════════════════════════════════════
# 6. CREATE data.yaml
# ═══════════════════════════════════════════════════════════════════════
data_yaml = {
    "path": YOLO_DIR,
    "train": "images/train",
    "val": "images/val",
    "test": "images/test",
    "nc": NUM_CLASSES,
    "names": final_classes,
}

yaml_path = f"{YOLO_DIR}/data.yaml"
with open(yaml_path, "w") as f:
    yaml.dump(data_yaml, f, default_flow_style=False)

print(f"\ndata.yaml:")
with open(yaml_path) as f:
    print(f.read())

# ═══════════════════════════════════════════════════════════════════════
# 7. TRAIN YOLOv8s
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 7: TRAINING YOLOv8s")
print(f"  Classes:    {NUM_CLASSES}")
print(f"  Train imgs: {split_counts['train']}")
print(f"  Val imgs:   {split_counts['val']}")
print("=" * 70)

from ultralytics import YOLO

model = YOLO("yolov8s.pt")

results = model.train(
    data=yaml_path,
    epochs=100,
    batch=32,                   # T4 16GB handles batch 32 at 640x640
    imgsz=640,
    device=0,
    amp=True,                   # FP16 mixed precision
    patience=15,                # Early stopping

    # Optimizer
    optimizer="SGD",
    lr0=0.01,
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3,

    # Augmentation
    mosaic=1.0,
    mixup=0.1,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    flipud=0.0,
    fliplr=0.5,
    scale=0.5,

    # Output
    project=f"{WORK_DIR}/runs/detect",
    name="idd_yolov8s",
    save=True,
    save_period=10,             # Checkpoint every 10 epochs
    plots=True,
    verbose=True,
)

# ═══════════════════════════════════════════════════════════════════════
# 8. COPY RESULTS FOR EASY DOWNLOAD
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 8: SAVING OUTPUTS")
print("=" * 70)

run_dir = f"{WORK_DIR}/runs/detect/idd_yolov8s"
best_pt = f"{run_dir}/weights/best.pt"
last_pt = f"{run_dir}/weights/last.pt"

if os.path.exists(best_pt):
    shutil.copy2(best_pt, f"{WORK_DIR}/yolo_idd_best.pt")
    print(f"best.pt → {WORK_DIR}/yolo_idd_best.pt ({os.path.getsize(best_pt)/1e6:.1f} MB)")

if os.path.exists(last_pt):
    shutil.copy2(last_pt, f"{WORK_DIR}/yolo_idd_last.pt")
    print(f"last.pt → {WORK_DIR}/yolo_idd_last.pt")

for f in glob.glob(f"{run_dir}/*.png") + glob.glob(f"{run_dir}/*.csv"):
    shutil.copy2(f, f"{WORK_DIR}/{os.path.basename(f)}")
    print(f"Copied: {os.path.basename(f)}")

# Print final metrics
results_csv = f"{run_dir}/results.csv"
if os.path.exists(results_csv):
    import pandas as pd
    df = pd.read_csv(results_csv)
    df.columns = df.columns.str.strip()
    print(f"\nLast epoch metrics:")
    print(df.tail(1).to_string())
    if "metrics/mAP50(B)" in df.columns:
        best_map = df["metrics/mAP50(B)"].max()
        print(f"\n>>> Best mAP@0.5: {best_map:.4f}")

print("\n" + "=" * 70)
print("DONE! Download 'yolo_idd_best.pt' from the Output tab")
print("=" * 70)
