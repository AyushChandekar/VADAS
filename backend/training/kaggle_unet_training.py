"""
KAGGLE NOTEBOOK — U-Net Drivable Zone Segmentation on IDD
==========================================================
Datasets (add BOTH as Input):
  1. kaggle.com/datasets/sounakp/idd-segmentation
     Mount: /kaggle/input/idd-segmentation/
  2. kaggle.com/datasets/thegovindkrishna/idd-segmentation-dataset-part-2
     Mount: /kaggle/input/idd-segmentation-dataset-part-2/

Instructions:
  1. Create new Kaggle notebook
  2. Settings → Accelerator → GPU T4 x2
  3. Settings → Internet → ON
  4. Add Input → search "idd-segmentation" by sounakp → Add
  5. Add Input → search "idd-segmentation-dataset-part-2" by thegovindkrishna → Add
  6. Paste this ENTIRE file as a single cell
  7. Click "Save & Run All" (dropdown, NOT the Run button)
  8. Close browser / shut down laptop — runs up to 12 hours
  9. Tomorrow: Output tab → download unet_drivable_best.pth
"""

# ═══════════════════════════════════════════════════════════════════════
# 0. INSTALL
# ═══════════════════════════════════════════════════════════════════════
import subprocess
subprocess.run(["pip", "install", "segmentation-models-pytorch", "-q"])

import os
import glob
import json
import time
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    props = torch.cuda.get_device_properties(0)
    vram = getattr(props, 'total_memory', None) or getattr(props, 'total_mem', 0)
    print(f"VRAM: {vram / 1e9:.1f} GB")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WORK_DIR = "/kaggle/working"

# Auto-detect all dataset directories under /kaggle/input/
print("All directories in /kaggle/input/:")
for d in os.listdir("/kaggle/input"):
    full = os.path.join("/kaggle/input", d)
    if os.path.isdir(full):
        print(f"  {d}/")

# Find all input directories (Kaggle mount names can have spaces, underscores, etc.)
ALL_INPUT_DIRS = [os.path.join("/kaggle/input", d)
                  for d in os.listdir("/kaggle/input")
                  if os.path.isdir(os.path.join("/kaggle/input", d))]

print(f"\nWill search across {len(ALL_INPUT_DIRS)} input directories:"  )
for d in ALL_INPUT_DIRS:
    print(f"  {d}")

# ═══════════════════════════════════════════════════════════════════════
# 1. DEEP SCAN — Print full directory tree for BOTH datasets
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


for input_dir in ALL_INPUT_DIRS:
    print(f"\n--- {input_dir} ---")
    print_tree(input_dir)

# ═══════════════════════════════════════════════════════════════════════
# 2. AUTO-DETECT: Find ALL images and annotations from BOTH datasets
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 2: AUTO-DETECTING IMAGES & ANNOTATIONS")
print("=" * 70)

# Search across ALL detected input directories
SEARCH_ROOTS = ALL_INPUT_DIRS


def find_files(pattern):
    """Find files matching glob pattern across all dataset roots."""
    results = []
    for root in SEARCH_ROOTS:
        if os.path.exists(root):
            results.extend(glob.glob(f"{root}/**/{pattern}", recursive=True))
    return results


# --- Images ---
# IDD images are typically named *_leftImg8bit.{jpg,png} OR just *.jpg
all_left_imgs = []
for ext in ["*_leftImg8bit.jpg", "*_leftImg8bit.png",
            "*_leftImg8bit.jpeg", "*_leftImg8bit.JPG"]:
    all_left_imgs.extend(find_files(ext))

# If no leftImg8bit naming, grab all images
all_plain_imgs = []
if len(all_left_imgs) == 0:
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]:
        all_plain_imgs.extend(find_files(ext))
    # Filter out anything that looks like a label/mask
    all_plain_imgs = [p for p in all_plain_imgs
                      if not any(x in p.lower() for x in
                                 ["gtfine", "label", "mask", "color", "instance"])]

# --- Annotations ---
# IDD provides several annotation formats. Detect which ones exist:
# Format A: *_gtFine_polygons.json (polygon annotations — render masks from these)
all_polygons = find_files("*_gtFine_polygons.json")
# Format B: *_gtFine_labellevel3Ids.png (pre-rendered label maps — pixel = class ID)
all_label_l3 = find_files("*_gtFine_labellevel3Ids.png")
# Format C: *_gtFine_labelids.png (alternative naming)
all_label_ids = find_files("*_gtFine_labelids.png")
# Format D: Generic label PNGs in a separate masks/labels folder
all_generic_masks = find_files("*_mask.png") + find_files("*_label.png")

print(f"Images (_leftImg8bit):   {len(all_left_imgs)}")
print(f"Images (plain):          {len(all_plain_imgs)}")
print(f"Polygon JSONs:           {len(all_polygons)}")
print(f"Label level3 IDs PNGs:   {len(all_label_l3)}")
print(f"Label ID PNGs:           {len(all_label_ids)}")
print(f"Generic mask PNGs:       {len(all_generic_masks)}")

# Show samples
for name, lst in [("leftImg8bit images", all_left_imgs),
                  ("plain images", all_plain_imgs),
                  ("polygon JSONs", all_polygons),
                  ("labellevel3Ids", all_label_l3),
                  ("labelids", all_label_ids)]:
    if lst:
        print(f"\nSample {name}:")
        for p in lst[:3]:
            print(f"  {p}")

# Decide which images to use
all_images = all_left_imgs if all_left_imgs else all_plain_imgs
print(f"\n>>> Using {len(all_images)} images for training")

# Decide which annotation format to use (priority order)
ANNOT_FORMAT = None
all_annotations = []

if len(all_label_l3) > 0:
    ANNOT_FORMAT = "labellevel3Ids"
    all_annotations = all_label_l3
    print(f">>> Using annotation format: labellevel3Ids ({len(all_label_l3)} files)")
elif len(all_label_ids) > 0:
    ANNOT_FORMAT = "labelids"
    all_annotations = all_label_ids
    print(f">>> Using annotation format: labelids ({len(all_label_ids)} files)")
elif len(all_polygons) > 0:
    ANNOT_FORMAT = "polygons"
    all_annotations = all_polygons
    print(f">>> Using annotation format: polygons ({len(all_polygons)} files)")
elif len(all_generic_masks) > 0:
    ANNOT_FORMAT = "generic_masks"
    all_annotations = all_generic_masks
    print(f">>> Using annotation format: generic masks ({len(all_generic_masks)} files)")
else:
    print("WARNING: No annotations found! Will attempt alternative matching...")

# ═══════════════════════════════════════════════════════════════════════
# 3. DISCOVER IDD LABEL IDs (if using label images)
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 3: DISCOVERING LABEL VALUES")
print("=" * 70)

# IDD level3 label mapping (standard):
#   0 = road, 1 = drivable fallback, 2 = sidewalk, 3 = non-drivable fallback,
#   4 = living things, 5 = 2-wheeler, 6 = 4-wheeler, 7 = far objects,
#   8 = sky, 9 = misc, 255 = void/unlabeled
#
# We want: road (0) + drivable fallback (1) → drivable (label=1 in our binary mask)
# Everything else → non-drivable (label=0)

# IDD level3 IDs where 0=road, 1=drivable_fallback
DRIVABLE_IDS_LEVEL3 = {0, 1}

if ANNOT_FORMAT in ("labellevel3Ids", "labelids") and len(all_annotations) > 0:
    # Sample 20 label images to find unique pixel values
    unique_vals = set()
    for label_path in all_annotations[:20]:
        arr = np.array(Image.open(label_path))
        unique_vals.update(np.unique(arr).tolist())

    print(f"Unique pixel values in label images (sampled 20): {sorted(unique_vals)}")
    print("IDD level3 mapping: 0=road, 1=drivable_fallback, 2=sidewalk, ...")
    print("We will treat pixel values {0, 1} as DRIVABLE, everything else as NON-DRIVABLE")

    if 0 in unique_vals and len(unique_vals) > 2:
        print("Label values look correct for IDD level3Ids format.")
    else:
        print(f"NOTE: Unusual values detected. Will proceed but check results.")

elif ANNOT_FORMAT == "polygons" and len(all_polygons) > 0:
    with open(all_polygons[0]) as f:
        sample_json = json.load(f)

    label_names = set()
    for obj in sample_json.get("objects", []):
        label_names.add(obj.get("label", "unknown"))

    print(f"Labels found in first JSON: {sorted(label_names)}")
    if "imgHeight" in sample_json and "imgWidth" in sample_json:
        print(f"Image size from JSON: {sample_json['imgWidth']} x {sample_json['imgHeight']}")

# ═══════════════════════════════════════════════════════════════════════
# 4. BUILD IMAGE-ANNOTATION PAIRS
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 4: MATCHING IMAGES TO ANNOTATIONS")
print("=" * 70)

# Build annotation lookup by stem
annot_by_stem = {}
for ap in all_annotations:
    stem = Path(ap).stem
    for suffix in ["_gtFine_labellevel3Ids", "_gtFine_labelids",
                   "_gtFine_polygons", "_mask", "_label",
                   "_gtFine_labelTrainIds", "_gtFine_color",
                   "_gtFine_instanceIds"]:
        if stem.endswith(suffix):
            stem = stem[:-len(suffix)]
            break
    annot_by_stem[stem] = ap

print(f"Annotation lookup entries: {len(annot_by_stem)}")
if annot_by_stem:
    sample_stems = list(annot_by_stem.keys())[:5]
    print(f"Sample annotation stems: {sample_stems}")


def get_image_stem(img_path):
    stem = Path(img_path).stem
    for suffix in ["_leftImg8bit", "_image", "_raw"]:
        if stem.endswith(suffix):
            return stem[:-len(suffix)]
    return stem


def get_split(filepath):
    path_lower = filepath.lower()
    if "/train/" in path_lower:
        return "train"
    elif "/val/" in path_lower:
        return "val"
    elif "/test/" in path_lower:
        return "test"
    return None


pairs_by_split = {"train": [], "val": [], "test": []}
unmatched = 0

for img_path in all_images:
    stem = get_image_stem(img_path)
    annot_path = annot_by_stem.get(stem)

    if annot_path is None:
        unmatched += 1
        continue

    split = get_split(img_path) or get_split(annot_path)
    if split is None:
        h = hash(stem) % 100
        split = "train" if h < 85 else ("val" if h < 95 else "test")

    pairs_by_split[split].append((img_path, annot_path))

print(f"\nMatched pairs:")
print(f"  Train: {len(pairs_by_split['train'])}")
print(f"  Val:   {len(pairs_by_split['val'])}")
print(f"  Test:  {len(pairs_by_split['test'])}")
print(f"  Unmatched images: {unmatched}")

total_pairs = sum(len(v) for v in pairs_by_split.values())

# ── SAFETY CHECK ──────────────────────────────────────────────────────
if total_pairs == 0:
    print("\n" + "!" * 70)
    print("ERROR: No image-annotation pairs found!")
    print(f"\nSample image stems:")
    for p in all_images[:10]:
        print(f"  {get_image_stem(p)} ← {p}")
    print(f"\nSample annotation stems:")
    for stem, path in list(annot_by_stem.items())[:10]:
        print(f"  {stem} ← {path}")
    print("!" * 70)
    raise SystemExit("Cannot proceed without data.")

# If val is empty, split from train
if len(pairs_by_split["val"]) == 0 and len(pairs_by_split["train"]) > 0:
    print("\nWARNING: No val split detected. Splitting 15% from train...")
    import random
    random.seed(42)
    random.shuffle(pairs_by_split["train"])
    n_val = int(len(pairs_by_split["train"]) * 0.15)
    pairs_by_split["val"] = pairs_by_split["train"][:n_val]
    pairs_by_split["train"] = pairs_by_split["train"][n_val:]
    print(f"  Train: {len(pairs_by_split['train'])}, Val: {len(pairs_by_split['val'])}")

# ═══════════════════════════════════════════════════════════════════════
# 5. MASK RENDERING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════

DRIVABLE_NAMES = {"road", "drivable fallback", "drivable_fallback", "driveable fallback"}


def render_binary_mask(annot_path, img_w=None, img_h=None):
    """
    Render a binary drivable mask from whatever annotation format we have.
    Returns: numpy array (H, W) with values 0 or 255.
    """
    if ANNOT_FORMAT in ("labellevel3Ids", "labelids"):
        label_arr = np.array(Image.open(annot_path))
        mask = np.zeros_like(label_arr, dtype=np.uint8)
        for lid in DRIVABLE_IDS_LEVEL3:
            mask[label_arr == lid] = 255
        return mask

    elif ANNOT_FORMAT == "polygons":
        with open(annot_path) as f:
            data = json.load(f)

        w = data.get("imgWidth", img_w)
        h = data.get("imgHeight", img_h)
        if w is None or h is None:
            return None

        mask = Image.new("L", (w, h), 0)
        draw = ImageDraw.Draw(mask)

        for obj in data.get("objects", []):
            label = obj.get("label", "").lower().strip()
            if label in DRIVABLE_NAMES:
                polygon = obj.get("polygon", [])
                if len(polygon) >= 3:
                    pts = [(p[0], p[1]) for p in polygon]
                    draw.polygon(pts, fill=255)

        return np.array(mask)

    elif ANNOT_FORMAT == "generic_masks":
        mask_arr = np.array(Image.open(annot_path).convert("L"))
        return ((mask_arr > 0) * 255).astype(np.uint8)

    return None


# ═══════════════════════════════════════════════════════════════════════
# 6. VERIFY MASKS
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 5: VERIFYING MASK RENDERING")
print("=" * 70)

drivable_pcts = []
bad_masks = 0

for img_path, annot_path in pairs_by_split["train"][:20]:
    img = Image.open(img_path)
    mask = render_binary_mask(annot_path, img.size[0], img.size[1])
    if mask is None:
        bad_masks += 1
        continue
    pct = (mask > 0).sum() / mask.size * 100
    drivable_pcts.append(pct)

if drivable_pcts:
    avg_pct = np.mean(drivable_pcts)
    print(f"Drivable area coverage (first 20 samples):")
    print(f"  Average: {avg_pct:.1f}%")
    print(f"  Range:   {min(drivable_pcts):.1f}% — {max(drivable_pcts):.1f}%")
    print(f"  Bad masks: {bad_masks}")

    if avg_pct < 1.0:
        print("\nWARNING: Very low drivable coverage!")
        print("Trying with IDs {0, 1, 2} as drivable...")
        DRIVABLE_IDS_LEVEL3.add(2)
        for img_path, annot_path in pairs_by_split["train"][:5]:
            mask = render_binary_mask(annot_path)
            if mask is not None:
                pct = (mask > 0).sum() / mask.size * 100
                print(f"  With ID 2: {pct:.1f}%")

    if avg_pct > 1.0:
        print("Masks look good!")

# Save sample visualization
if len(pairs_by_split["train"]) >= 4:
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for i in range(4):
        img_path, annot_path = pairs_by_split["train"][i]
        img = Image.open(img_path).convert("RGB")
        mask = render_binary_mask(annot_path, img.size[0], img.size[1])
        axes[0, i].imshow(img)
        axes[0, i].set_title(f"Image {i}")
        axes[0, i].axis("off")
        if mask is not None:
            axes[1, i].imshow(mask, cmap="gray", vmin=0, vmax=255)
            pct = (mask > 0).sum() / mask.size * 100
            axes[1, i].set_title(f"Drivable ({pct:.1f}%)")
        else:
            axes[1, i].set_title("FAILED")
        axes[1, i].axis("off")
    plt.suptitle("Sample Training Pairs")
    plt.tight_layout()
    plt.savefig(f"{WORK_DIR}/sample_masks.png", dpi=100)
    print(f"Sample visualization saved.")


# ═══════════════════════════════════════════════════════════════════════
# 7. DATASET CLASS
# ═══════════════════════════════════════════════════════════════════════

class IDDDrivableDataset(Dataset):
    def __init__(self, pairs, img_w=512, img_h=256, is_train=True):
        self.pairs = pairs
        self.img_w = img_w
        self.img_h = img_h
        self.is_train = is_train

        self.img_transform_train = transforms.Compose([
            transforms.Resize((img_h, img_w), interpolation=Image.BILINEAR),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.img_transform_val = transforms.Compose([
            transforms.Resize((img_h, img_w), interpolation=Image.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.flip_prob = 0.5 if is_train else 0.0

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, annot_path = self.pairs[idx]

        image = Image.open(img_path).convert("RGB")
        mask_arr = render_binary_mask(annot_path, image.size[0], image.size[1])

        if mask_arr is None:
            mask_arr = np.zeros((image.size[1], image.size[0]), dtype=np.uint8)

        mask_pil = Image.fromarray(mask_arr)
        mask_resized = mask_pil.resize((self.img_w, self.img_h), Image.NEAREST)

        # Synchronized random horizontal flip
        if self.is_train and np.random.random() < self.flip_prob:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            mask_resized = mask_resized.transpose(Image.FLIP_LEFT_RIGHT)

        if self.is_train:
            image_tensor = self.img_transform_train(image)
        else:
            image_tensor = self.img_transform_val(image)

        binary_mask = (np.array(mask_resized) > 0).astype(np.int64)
        mask_tensor = torch.from_numpy(binary_mask).long()

        return image_tensor, mask_tensor


# ═══════════════════════════════════════════════════════════════════════
# 8. U-NET MODEL (base=64, ~5M params)
# ═══════════════════════════════════════════════════════════════════════

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        layers += [
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=2, base=64, dropout=0.1):
        super().__init__()
        b = base
        self.enc1 = DoubleConv(in_channels, b)
        self.enc2 = DoubleConv(b, b * 2, dropout)
        self.enc3 = DoubleConv(b * 2, b * 4, dropout)
        self.enc4 = DoubleConv(b * 4, b * 8, dropout)
        self.pool = nn.MaxPool2d(2, 2)
        self.bottleneck = DoubleConv(b * 8, b * 16, dropout)
        self.up4 = nn.ConvTranspose2d(b * 16, b * 8, 2, stride=2)
        self.dec4 = DoubleConv(b * 16, b * 8, dropout)
        self.up3 = nn.ConvTranspose2d(b * 8, b * 4, 2, stride=2)
        self.dec3 = DoubleConv(b * 8, b * 4, dropout)
        self.up2 = nn.ConvTranspose2d(b * 4, b * 2, 2, stride=2)
        self.dec2 = DoubleConv(b * 4, b * 2, dropout)
        self.up1 = nn.ConvTranspose2d(b * 2, b, 2, stride=2)
        self.dec1 = DoubleConv(b * 2, b)
        self.out_conv = nn.Conv2d(b, num_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        bn = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(bn), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out_conv(d1)


# ═══════════════════════════════════════════════════════════════════════
# 9. LOSS + METRICS
# ═══════════════════════════════════════════════════════════════════════

class DiceBCELoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
        self.ce = nn.CrossEntropyLoss()

    def dice_loss(self, pred_probs, target):
        target_f = target.float()
        intersection = (pred_probs * target_f).sum(dim=(1, 2))
        union = pred_probs.sum(dim=(1, 2)) + target_f.sum(dim=(1, 2))
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()

    def forward(self, logits, target):
        ce_loss = self.ce(logits, target)
        probs = F.softmax(logits, dim=1)[:, 1, :, :]
        dice = self.dice_loss(probs, target)
        return 0.5 * ce_loss + 0.5 * dice


def compute_iou(pred_mask, true_mask, smooth=1e-6):
    pred_flat = (pred_mask == 1).float().view(-1)
    true_flat = (true_mask == 1).float().view(-1)
    inter = (pred_flat * true_flat).sum()
    union = pred_flat.sum() + true_flat.sum() - inter
    return ((inter + smooth) / (union + smooth)).item()


def compute_pixel_accuracy(pred_mask, true_mask):
    return (pred_mask == true_mask).float().mean().item()


def compute_dice(pred_mask, true_mask, smooth=1e-6):
    pred_flat = (pred_mask == 1).float().view(-1)
    true_flat = (true_mask == 1).float().view(-1)
    inter = (pred_flat * true_flat).sum()
    return ((2 * inter + smooth) / (pred_flat.sum() + true_flat.sum() + smooth)).item()


# ═══════════════════════════════════════════════════════════════════════
# 10. CONFIG
# ═══════════════════════════════════════════════════════════════════════
IMG_W = 512
IMG_H = 256
BASE_CHANNELS = 64
BATCH_SIZE = 16
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
NUM_WORKERS = 4
PATIENCE = 10

# ═══════════════════════════════════════════════════════════════════════
# 11. DATA LOADERS
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 6: CREATING DATA LOADERS")
print("=" * 70)

train_ds = IDDDrivableDataset(pairs_by_split["train"], IMG_W, IMG_H, is_train=True)
val_ds = IDDDrivableDataset(pairs_by_split["val"], IMG_W, IMG_H, is_train=False)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=True)

print(f"Train: {len(train_ds)} samples, {len(train_loader)} batches")
print(f"Val:   {len(val_ds)} samples, {len(val_loader)} batches")

# Sanity check
print("Loading first batch...")
imgs, masks = next(iter(train_loader))
print(f"  Image batch: {imgs.shape}, dtype={imgs.dtype}")
print(f"  Mask batch:  {masks.shape}, dtype={masks.dtype}")
print(f"  Mask values:  {masks.unique().tolist()}")
print(f"  Drivable %:   {(masks == 1).float().mean().item() * 100:.1f}%")

# ═══════════════════════════════════════════════════════════════════════
# 12. TRAIN
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 7: TRAINING U-NET")
print(f"  Model:  U-Net base={BASE_CHANNELS}")
print(f"  Input:  {IMG_W}x{IMG_H}")
print(f"  Batch:  {BATCH_SIZE}")
print(f"  Epochs: {NUM_EPOCHS}")
print(f"  Device: {DEVICE}")
print("=" * 70)

model = UNet(in_channels=3, num_classes=2, base=BASE_CHANNELS, dropout=0.1).to(DEVICE)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model parameters: {total_params / 1e6:.2f}M")

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)
criterion = DiceBCELoss()
scaler = torch.amp.GradScaler("cuda", enabled=True)

best_iou = 0.0
patience_counter = 0
training_log = []

for epoch in range(1, NUM_EPOCHS + 1):
    # ── Train ─────────────────────────────────────────────────────────
    model.train()
    train_loss = 0.0
    t0 = time.time()

    for i, (images, masks) in enumerate(train_loader):
        images = images.to(DEVICE, non_blocking=True)
        masks = masks.to(DEVICE, non_blocking=True)

        optimizer.zero_grad()
        with torch.amp.autocast(device_type="cuda"):
            logits = model(images)
            loss = criterion(logits, masks)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()

        if (i + 1) % 50 == 0:
            print(f"  Epoch {epoch:02d} | Step {i+1:04d}/{len(train_loader)} | Loss: {loss.item():.4f}")

    train_loss /= len(train_loader)

    # ── Validate ──────────────────────────────────────────────────────
    model.eval()
    val_loss = 0.0
    val_iou = 0.0
    val_acc = 0.0
    val_dice = 0.0

    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(DEVICE, non_blocking=True)
            masks = masks.to(DEVICE, non_blocking=True)

            with torch.amp.autocast(device_type="cuda"):
                logits = model(images)
                loss = criterion(logits, masks)

            pred_mask = logits.argmax(dim=1)
            val_loss += loss.item()
            val_iou += compute_iou(pred_mask, masks)
            val_acc += compute_pixel_accuracy(pred_mask, masks)
            val_dice += compute_dice(pred_mask, masks)

    n = max(len(val_loader), 1)
    val_loss /= n
    val_iou /= n
    val_acc /= n
    val_dice /= n

    scheduler.step()
    elapsed = time.time() - t0

    print(f"\nEpoch {epoch:02d}/{NUM_EPOCHS} ({elapsed:.0f}s)")
    print(f"  Train Loss : {train_loss:.4f}")
    print(f"  Val Loss   : {val_loss:.4f}")
    print(f"  Val IoU    : {val_iou:.4f}  (target >= 0.80)")
    print(f"  Val Dice   : {val_dice:.4f}")
    print(f"  Pixel Acc  : {val_acc:.4f}")
    print(f"  LR         : {scheduler.get_last_lr()[0]:.2e}")

    training_log.append({
        "epoch": epoch, "train_loss": train_loss,
        "val_loss": val_loss, "val_iou": val_iou,
        "val_dice": val_dice, "val_acc": val_acc,
    })

    if val_iou > best_iou:
        best_iou = val_iou
        patience_counter = 0
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "val_iou": val_iou,
            "val_dice": val_dice,
            "val_loss": val_loss,
            "config": {
                "base_channels": BASE_CHANNELS,
                "img_w": IMG_W,
                "img_h": IMG_H,
                "num_classes": 2,
                "annot_format": ANNOT_FORMAT,
            }
        }, f"{WORK_DIR}/unet_drivable_best.pth")
        print(f"  >>> Best model saved (IoU: {best_iou:.4f})")
    else:
        patience_counter += 1
        print(f"  No improvement ({patience_counter}/{PATIENCE})")

    if epoch % 10 == 0:
        torch.save(model.state_dict(), f"{WORK_DIR}/unet_epoch_{epoch:02d}.pth")
        print(f"  Checkpoint: unet_epoch_{epoch:02d}.pth")

    if patience_counter >= PATIENCE:
        print(f"\nEarly stopping at epoch {epoch}")
        break

# ═══════════════════════════════════════════════════════════════════════
# 13. SAVE RESULTS
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 8: SAVING RESULTS")
print("=" * 70)

with open(f"{WORK_DIR}/training_log.json", "w") as f:
    json.dump(training_log, f, indent=2)

# Training curves
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
ep_list = [e["epoch"] for e in training_log]

axes[0].plot(ep_list, [e["train_loss"] for e in training_log], label="Train")
axes[0].plot(ep_list, [e["val_loss"] for e in training_log], label="Val")
axes[0].set_title("Loss"); axes[0].legend(); axes[0].set_xlabel("Epoch")

axes[1].plot(ep_list, [e["val_iou"] for e in training_log], label="IoU", color="green")
axes[1].axhline(y=0.80, color="red", linestyle="--", label="Target")
axes[1].set_title("IoU"); axes[1].legend(); axes[1].set_xlabel("Epoch")

axes[2].plot(ep_list, [e["val_dice"] for e in training_log], label="Dice", color="purple")
axes[2].plot(ep_list, [e["val_acc"] for e in training_log], label="Pixel Acc", color="orange")
axes[2].set_title("Dice & Accuracy"); axes[2].legend(); axes[2].set_xlabel("Epoch")

plt.suptitle(f"U-Net (base={BASE_CHANNELS}) — Best IoU: {best_iou:.4f}")
plt.tight_layout()
plt.savefig(f"{WORK_DIR}/training_curves.png", dpi=150)

# ═══════════════════════════════════════════════════════════════════════
# 14. PREDICTION SAMPLES
# ═══════════════════════════════════════════════════════════════════════
print("Generating prediction samples...")

ckpt = torch.load(f"{WORK_DIR}/unet_drivable_best.pth", map_location=DEVICE)
model.load_state_dict(ckpt["model_state"])
model.eval()

val_tf = transforms.Compose([
    transforms.Resize((IMG_H, IMG_W), interpolation=Image.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

n_samples = min(4, len(pairs_by_split["val"]))
if n_samples > 0:
    fig, axes = plt.subplots(3, n_samples, figsize=(4 * n_samples, 12))
    if n_samples == 1:
        axes = axes.reshape(-1, 1)

    for i in range(n_samples):
        img_path, annot_path = pairs_by_split["val"][i]
        orig = Image.open(img_path).convert("RGB")
        gt = render_binary_mask(annot_path, orig.size[0], orig.size[1])
        if gt is None:
            gt = np.zeros((orig.size[1], orig.size[0]), dtype=np.uint8)
        gt_r = np.array(Image.fromarray(gt).resize((IMG_W, IMG_H), Image.NEAREST))

        tensor = val_tf(orig).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            with torch.amp.autocast(device_type="cuda"):
                pred = model(tensor).argmax(dim=1).squeeze(0).cpu().numpy()

        axes[0, i].imshow(orig.resize((IMG_W, IMG_H)))
        axes[0, i].set_title("Image"); axes[0, i].axis("off")
        axes[1, i].imshow(gt_r > 0, cmap="gray")
        axes[1, i].set_title("Ground Truth"); axes[1, i].axis("off")
        iou = compute_iou(torch.tensor(pred), torch.tensor((gt_r > 0).astype(int)))
        axes[2, i].imshow(pred, cmap="gray")
        axes[2, i].set_title(f"Predicted (IoU: {iou:.3f})"); axes[2, i].axis("off")

    plt.suptitle(f"Best Model Predictions (IoU: {best_iou:.4f})")
    plt.tight_layout()
    plt.savefig(f"{WORK_DIR}/prediction_samples.png", dpi=150)

# ═══════════════════════════════════════════════════════════════════════
# 15. SUMMARY
# ═══════════════════════════════════════════════════════════════════════
file_size = os.path.getsize(f"{WORK_DIR}/unet_drivable_best.pth") / 1e6

print("\n" + "=" * 70)
print("FINAL RESULTS")
print("=" * 70)
print(f"  Best IoU      : {best_iou:.4f}")
print(f"  Best Dice     : {ckpt['val_dice']:.4f}")
print(f"  Best Epoch    : {ckpt['epoch']}")
print(f"  Model Size    : {file_size:.1f} MB")
print(f"  Annot Format  : {ANNOT_FORMAT}")
print(f"  Train Samples : {len(pairs_by_split['train'])}")
print(f"  Val Samples   : {len(pairs_by_split['val'])}")
print(f"\nOutput files:")
for f in sorted(glob.glob(f"{WORK_DIR}/*")):
    if os.path.isfile(f):
        sz = os.path.getsize(f) / 1e6
        print(f"  {os.path.basename(f):40s} {sz:8.1f} MB")

print("\n" + "=" * 70)
print("DONE! Download 'unet_drivable_best.pth' from the Output tab")
print("=" * 70)
