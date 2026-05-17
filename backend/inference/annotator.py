import cv2
import numpy as np
from models.detector import Detection

# Color palette
COLOR_DRIVABLE = (0, 200, 0)       # green overlay
COLOR_TRAJECTORY = (0, 0, 255)     # red polyline
COLOR_TARGET = (0, 255, 255)       # yellow circle
COLOR_BOX_VEHICLE = (255, 180, 0)  # orange
COLOR_BOX_PERSON = (0, 0, 255)     # red
COLOR_BOX_OTHER = (200, 200, 0)    # cyan

ACTION_COLORS = {
    "MOVE FORWARD": (0, 200, 0),
    "SLOW DOWN": (0, 200, 255),
    "STOP": (0, 0, 255),
    "MOVE LEFT": (255, 200, 0),
    "MOVE RIGHT": (255, 200, 0),
    "HORN": (0, 255, 255),
    "WIPER": (200, 200, 200),
}

PERSON_CLASSES = {"person", "rider", "animal"}
VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle", "bicycle", "autorickshaw", "vehicle_fallback"}


def annotate_frame(
    frame_bgr: np.ndarray,
    detections: list[Detection],
    drivable_mask: np.ndarray | None,
    trajectory: dict | None,
    decision: dict | None,
) -> np.ndarray:
    """Draw all annotations on a copy of the frame."""
    out = frame_bgr.copy()
    h, w = out.shape[:2]

    # 1. Drivable zone overlay (semi-transparent green border only)
    if drivable_mask is not None:
        mask_resized = cv2.resize(drivable_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        overlay = np.zeros_like(out)
        overlay[mask_resized > 127] = COLOR_DRIVABLE
        out = cv2.addWeighted(out, 0.85, overlay, 0.15, 0)

    # 2. Trajectory polyline
    if trajectory and trajectory.get("polyline"):
        polyline = trajectory["polyline"]
        # Scale polyline from mask coords to frame coords
        mask_h, mask_w = (drivable_mask.shape[:2] if drivable_mask is not None
                          else (256, 512))
        scale_x = w / mask_w
        scale_y = h / mask_h
        pts = [(int(x * scale_x), int(y * scale_y)) for x, y in polyline]

        for i in range(len(pts) - 1):
            cv2.line(out, pts[i], pts[i + 1], COLOR_TRAJECTORY, 4, cv2.LINE_AA)

        # Steering target (more subtle)
        st = trajectory["steering_target"]
        target_pt = (int(st[0] * scale_x), int(st[1] * scale_y))
        cv2.circle(out, target_pt, 6, COLOR_TARGET, -1, cv2.LINE_AA)

    # 3. Detection boxes
    for det in detections:
        if det.class_name in PERSON_CLASSES:
            color = COLOR_BOX_PERSON
        elif det.class_name in VEHICLE_CLASSES:
            color = COLOR_BOX_VEHICLE
        else:
            color = COLOR_BOX_OTHER

        cv2.rectangle(out, (det.x1, det.y1), (det.x2, det.y2), color, 2)
        label = f"{det.class_name} {det.confidence:.0%}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (det.x1, det.y1 - th - 6),
                      (det.x1 + tw + 4, det.y1), color, -1)
        cv2.putText(out, label, (det.x1 + 2, det.y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                    cv2.LINE_AA)

    # 4. Decision banner (small, top-left, semi-transparent)
    if decision:
        action = decision["action"]
        reason = decision.get("reason", "")
        color = ACTION_COLORS.get(action, (200, 200, 200))

        # Semi-transparent background
        banner = out[0:35, 0:w].copy()
        cv2.rectangle(banner, (0, 0), (w, 35), (0, 0, 0), -1)
        out[0:35, 0:w] = cv2.addWeighted(out[0:35, 0:w], 0.5, banner, 0.5, 0)
        cv2.putText(out, action, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

        # Reason at bottom
        cv2.putText(out, reason, (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                    cv2.LINE_AA)

    return out
