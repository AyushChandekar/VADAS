"""
Local U-Net training on IDD Segmentation (fallback — prefer Kaggle).

Usage:
    python -m training.train_unet \
        --images path/to/leftImg8bit \
        --masks  path/to/drivable_masks \
        --epochs 25
"""

import argparse
import os
import glob
import time
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path

import segmentation_models_pytorch as smp


class DrivableDataset(Dataset):
    def __init__(self, image_paths, mask_paths, img_w=512, img_h=256, is_train=True):
        self.pairs = list(zip(image_paths, mask_paths))
        self.img_w = img_w
        self.img_h = img_h
        self.is_train = is_train

        self.transform_train = transforms.Compose([
            transforms.Resize((img_h, img_w)),
            transforms.ColorJitter(0.4, 0.4, 0.3, 0.08),
            transforms.RandomGrayscale(p=0.1),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            transforms.RandomAffine(degrees=5, translate=(0.05, 0.05)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.transform_val = transforms.Compose([
            transforms.Resize((img_h, img_w)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).resize((self.img_w, self.img_h), Image.NEAREST)

        tf = self.transform_train if self.is_train else self.transform_val
        image_tensor = tf(image)
        mask_tensor = torch.from_numpy((np.array(mask) > 0).astype(np.int64)).long()
        return image_tensor, mask_tensor


class DiceBCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits, target):
        ce = self.ce(logits, target)
        probs = F.softmax(logits, dim=1)[:, 1]
        target_f = target.float()
        inter = (probs * target_f).sum(dim=(1, 2))
        union = probs.sum(dim=(1, 2)) + target_f.sum(dim=(1, 2))
        dice = 1 - (2 * inter + 1) / (union + 1)
        return 0.5 * ce + 0.5 * dice.mean()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True, help="Image directory")
    parser.add_argument("--masks", required=True, help="Pre-rendered mask directory")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    # Collect image-mask pairs
    all_masks = glob.glob(os.path.join(args.masks, "**/*_drivable.png"), recursive=True)
    mask_by_stem = {Path(p).stem.replace("_drivable", ""): p for p in all_masks}

    all_imgs = glob.glob(os.path.join(args.images, "**/*_leftImg8bit.*"), recursive=True)

    train_imgs, train_masks = [], []
    val_imgs, val_masks = [], []

    for img_path in all_imgs:
        stem = Path(img_path).stem.replace("_leftImg8bit", "")
        mask_path = mask_by_stem.get(stem)
        if mask_path is None:
            continue

        if "/val/" in img_path.lower() or "/validation/" in img_path.lower():
            val_imgs.append(img_path)
            val_masks.append(mask_path)
        else:
            train_imgs.append(img_path)
            train_masks.append(mask_path)

    print(f"Train: {len(train_imgs)}, Val: {len(val_imgs)}")

    train_ds = DrivableDataset(train_imgs, train_masks, is_train=True)
    val_ds = DrivableDataset(val_imgs, val_masks, is_train=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                            num_workers=4, pin_memory=True)

    model = smp.Unet(
        encoder_name="resnet18",
        encoder_weights="imagenet",
        in_channels=3,
        classes=2,
        activation=None,
    ).to(args.device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params / 1e6:.2f}M (ResNet18 encoder, pretrained)")

    # Differential LR: lower for pretrained encoder, full LR for decoder
    optimizer = torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": args.lr * 0.1},
        {"params": model.decoder.parameters(), "lr": args.lr},
        {"params": model.segmentation_head.parameters(), "lr": args.lr},
    ], weight_decay=1e-5)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = DiceBCELoss()
    scaler = torch.amp.GradScaler("cuda")

    best_iou = 0
    patience_cnt = 0
    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(args.device), masks.to(args.device)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda"):
                loss = criterion(model(imgs), masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_iou = 0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(args.device), masks.to(args.device)
                pred = model(imgs).argmax(dim=1)
                inter = ((pred == 1) & (masks == 1)).float().sum()
                union = ((pred == 1) | (masks == 1)).float().sum()
                val_iou += (inter / (union + 1e-6)).item()
        val_iou /= max(len(val_loader), 1)

        scheduler.step()
        print(f"Epoch {epoch:02d} | Loss: {train_loss:.4f} | IoU: {val_iou:.4f}")

        if val_iou > best_iou:
            best_iou = val_iou
            patience_cnt = 0
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_iou": val_iou,
                "config": {
                    "model_type": "smp_resnet18",
                    "encoder": "resnet18",
                    "img_w": 512,
                    "img_h": 256,
                },
            }, "checkpoints/unet_drivable_best.pth")
            print(f"  >>> Saved (IoU: {best_iou:.4f})")
        else:
            patience_cnt += 1
            if patience_cnt >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    print(f"Best IoU: {best_iou:.4f}")


if __name__ == "__main__":
    main()
