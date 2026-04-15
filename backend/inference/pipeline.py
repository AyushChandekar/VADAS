import time
import threading
import numpy as np
import cv2

from models.detector import ObjectDetector
from models.segmentor import DrivableSegmentor
from models.trajectory import TrajectoryPlanner
from models.decision_engine import DecisionEngine
from inference.annotator import annotate_frame


class InferencePipeline:
    """
    Combined AI pipeline: runs YOLO + U-Net on every frame,
    computes trajectory and decision, produces annotated output.

    Thread-safe — the backend reads latest results while the
    pipeline processes frames continuously.
    """

    def __init__(self, yolo_path: str, unet_path: str, device: str = "cuda"):
        self.device = device

        # Load models
        print(f"Loading YOLOv8 from {yolo_path}...")
        self.detector = ObjectDetector(yolo_path, device=device)
        print(f"Loading U-Net from {unet_path}...")
        self.segmentor = DrivableSegmentor(unet_path, device=device)
        print("Models loaded.")

        self.trajectory_planner = TrajectoryPlanner()
        self.decision_engine = DecisionEngine()

        # Shared state (thread-safe)
        self.lock = threading.Lock()
        self._latest_annotated = None
        self._latest_detections = []
        self._latest_trajectory = {}
        self._latest_decision = {"action": "INITIALIZING", "reason": "", "confidence": 0}
        self._fps = 0.0
        self._frame_count = 0
        self._fps_time = time.time()

    def process_frame(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Run full pipeline on one frame. Returns annotated frame.
        Also updates internal latest state for API access.
        """
        h, w = frame_bgr.shape[:2]
        t0 = time.time()

        # Run models
        detections = self.detector.predict(frame_bgr)
        drivable_mask = self.segmentor.predict(frame_bgr)

        # Trajectory
        trajectory = self.trajectory_planner.plan(drivable_mask)

        # Decision
        decision = self.decision_engine.decide(detections, trajectory, h, w)

        # Annotate
        annotated = annotate_frame(frame_bgr, detections, drivable_mask,
                                   trajectory, decision)

        # Update FPS
        self._frame_count += 1
        elapsed = time.time() - self._fps_time
        if elapsed >= 1.0:
            fps = self._frame_count / elapsed
            self._frame_count = 0
            self._fps_time = time.time()
        else:
            fps = self._fps

        # Store results (thread-safe)
        with self.lock:
            self._latest_annotated = annotated
            self._latest_detections = detections
            self._latest_trajectory = trajectory
            self._latest_decision = decision
            self._fps = fps

        return annotated

    # ── Thread-safe getters for the API ───────────────────────────────

    def get_latest_frame_jpeg(self, quality: int = 85) -> bytes | None:
        with self.lock:
            frame = self._latest_annotated
        if frame is None:
            return None
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return buf.tobytes()

    def get_latest_status(self) -> dict:
        with self.lock:
            return {
                "action": self._latest_decision.get("action", "UNKNOWN"),
                "reason": self._latest_decision.get("reason", ""),
                "confidence": self._latest_decision.get("confidence", 0),
                "fps": round(self._fps, 1),
            }

    def get_latest_detections(self) -> list[dict]:
        with self.lock:
            dets = self._latest_detections
        return [
            {
                "class_name": d.class_name,
                "confidence": round(d.confidence, 3),
                "bbox": [d.x1, d.y1, d.x2, d.y2],
                "proximity": _proximity_label(d.bottom_y, d.area, 720, 1280),
            }
            for d in dets
        ]

    def get_latest_trajectory(self) -> dict:
        with self.lock:
            return dict(self._latest_trajectory)


def _proximity_label(bottom_y: int, area: int, frame_h: int, frame_w: int) -> str:
    rel_bottom = bottom_y / frame_h
    rel_area = area / (frame_h * frame_w)
    if rel_bottom > 0.85 or rel_area > 0.08:
        return "NEAR"
    if rel_bottom > 0.65 or rel_area > 0.03:
        return "MEDIUM"
    return "FAR"
