import numpy as np
from dataclasses import dataclass
from ultralytics import YOLO


@dataclass
class Detection:
    class_name: str
    class_id: int
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self):
        return self.x2 - self.x1

    @property
    def height(self):
        return self.y2 - self.y1

    @property
    def center_x(self):
        return (self.x1 + self.x2) // 2

    @property
    def center_y(self):
        return (self.y1 + self.y2) // 2

    @property
    def bottom_y(self):
        return self.y2

    @property
    def area(self):
        return self.width * self.height


class ObjectDetector:
    """Wrapper for YOLOv8 inference."""

    def __init__(self, checkpoint_path: str, device: str = "cuda",
                 conf: float = 0.35, iou: float = 0.45, imgsz: int = 640):
        self.model = YOLO(checkpoint_path)
        self.device = device
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz

    def predict(self, frame_bgr: np.ndarray) -> list[Detection]:
        """
        Args:
            frame_bgr: OpenCV BGR image (H, W, 3)
        Returns:
            List of Detection objects.
        """
        results = self.model.predict(
            frame_bgr,
            device=self.device,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            verbose=False,
        )

        detections = []
        for r in results:
            boxes = r.boxes
            if boxes is None:
                continue
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                cls_name = r.names[cls_id]
                conf = float(boxes.conf[i].item())
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)
                detections.append(Detection(
                    class_name=cls_name,
                    class_id=cls_id,
                    confidence=conf,
                    x1=int(x1), y1=int(y1),
                    x2=int(x2), y2=int(y2),
                ))

        return detections
