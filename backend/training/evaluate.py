"""
Evaluate trained models on validation data.

Usage:
    python -m training.evaluate --model detection --checkpoint checkpoints/yolo_idd_best.pt
    python -m training.evaluate --model segmentation --checkpoint checkpoints/unet_drivable_best.pth --images path/to/images --masks path/to/masks
"""

import argparse
import os
import glob

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from pathlib import Path


def evaluate_yolo(checkpoint: str):
    """Evaluate YOLO on IDD Detection val set."""
    from ultralytics import YOLO
    model = YOLO(checkpoint)
    metrics = model.val(data="configs/yolo_idd.yaml", split="val", verbose=True)
    print(f"\nResults:")
    print(f"  mAP@0.5:      {metrics.box.map50:.4f}")
    print(f"  mAP@0.5:0.95: {metrics.box.map:.4f}")
    print(f"  Precision:     {metrics.box.mp:.4f}")
    print(f"  Recall:        {metrics.box.mr:.4f}")


def evaluate_unet(checkpoint: str, images_dir: str, masks_dir: str):
    """Evaluate U-Net on drivable segmentation val set."""
    from models.segmentor import UNet

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    base = cfg.get("base_channels", 64)
    img_w = cfg.get("img_w", 512)
    img_h = cfg.get("img_h", 256)

    model = UNet(base=base).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((img_h, img_w)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    val_images = glob.glob(os.path.join(images_dir, "**/*.*"), recursive=True)
    mask_by_stem = {}
    for p in glob.glob(os.path.join(masks_dir, "**/*_drivable.png"), recursive=True):
        mask_by_stem[Path(p).stem.replace("_drivable", "")] = p

    ious, accs, dices = [], [], []

    for img_path in val_images:
        stem = Path(img_path).stem.replace("_leftImg8bit", "")
        mask_path = mask_by_stem.get(stem)
        if mask_path is None:
            continue

        img = Image.open(img_path).convert("RGB")
        gt = np.array(Image.open(mask_path).resize((img_w, img_h), Image.NEAREST)) > 0

        tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(tensor).argmax(dim=1).squeeze(0).cpu().numpy()

        inter = (pred & gt).sum()
        union = (pred | gt).sum()
        iou = inter / (union + 1e-6)
        acc = (pred == gt).mean()
        dice = 2 * inter / (pred.sum() + gt.sum() + 1e-6)

        ious.append(iou)
        accs.append(acc)
        dices.append(dice)

    print(f"\nResults ({len(ious)} samples):")
    print(f"  IoU:           {np.mean(ious):.4f}")
    print(f"  Pixel Accuracy:{np.mean(accs):.4f}")
    print(f"  Dice:          {np.mean(dices):.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["detection", "segmentation"], required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--images", help="Validation images dir (segmentation only)")
    parser.add_argument("--masks", help="Validation masks dir (segmentation only)")
    args = parser.parse_args()

    if args.model == "detection":
        evaluate_yolo(args.checkpoint)
    else:
        if not args.images or not args.masks:
            print("Error: --images and --masks required for segmentation")
            return
        evaluate_unet(args.checkpoint, args.images, args.masks)


if __name__ == "__main__":
    main()
