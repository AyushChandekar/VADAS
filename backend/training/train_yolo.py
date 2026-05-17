"""
Local YOLOv8 training on IDD Detection (fallback — prefer Kaggle).

Usage:
    python -m training.train_yolo --data configs/yolo_idd.yaml --epochs 100
"""

import argparse
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 on IDD Detection")
    parser.add_argument("--data", default="configs/yolo_idd.yaml", help="Dataset config")
    parser.add_argument("--model", default="yolov8s.pt", help="Base model")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16, help="Batch size (16 for 6GB VRAM)")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--patience", type=int, default=15)
    args = parser.parse_args()

    model = YOLO(args.model)

    model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        amp=True,
        patience=args.patience,
        optimizer="SGD",
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        mosaic=1.0,
        mixup=0.1,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        fliplr=0.5,
        scale=0.5,
        project="runs/detect",
        name="idd_yolov8s",
        save=True,
        save_period=10,
        plots=True,
        verbose=True,
    )


if __name__ == "__main__":
    main()
