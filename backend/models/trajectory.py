import numpy as np
import cv2
from models.detector import Detection
from scipy.interpolate import interp1d

class PathGuidancePlanner:
    """
    Compute safe forward guidance paths using Fusion-Sequence logic:
    1. Fusion: Drivable Mask - YOLO Obstacles (Obstacle Repulsion).
    2. Medial Axis: Distance Transform to find the safest center path.
    3. Spline-like Smoothing: Temporal averaging and interpolation for stable red lines.
    """

    def __init__(self, history_len: int = 5, alpha: float = 0.4):
        self.history_len = history_len
        self.alpha = alpha  # Smoothing factor (0-1), lower is more stable
        self._prev_path = None
        
    def plan(self, drivable_mask: np.ndarray, detections: list[Detection]) -> dict:
        """
        Args:
            drivable_mask: Binary mask (H, W) where 255=drivable.
            detections: List of Detection objects from YOLO.
        Returns:
            dict with path and steering info.
        """
        h, w = drivable_mask.shape[:2]
        
        # 1. Fusion & Obstacle Repulsion
        # Create a "Safe Zone" mask by removing obstacles from the drivable area
        safe_zone = (drivable_mask > 127).astype(np.uint8) * 255
        
        for det in detections:
            # Mask out the bounding box with a small safety buffer (10%)
            padding_w = int(det.width * 0.1)
            padding_h = int(det.height * 0.1)
            x1 = max(0, det.x1 - padding_w)
            y1 = max(0, det.y1 - padding_h)
            x2 = min(w, det.x2 + padding_w)
            y2 = min(h, det.y2 + padding_h)
            
            # Map detections to mask resolution if they differ
            # (Assuming detections are in frame resolution, mask is 512x256)
            # We'll handle scaling here if needed, but usually InferencePipeline handles this.
            # For now, we assume coordinates are aligned or scaled by the caller.
            cv2.rectangle(safe_zone, (x1, y1), (x2, y2), 0, -1)

        # 2. Medial Axis Transformation (via Distance Transform)
        # Distance transform finds the distance of each pixel to the nearest "wall" (non-safe pixel)
        dist_transform = cv2.distanceTransform(safe_zone, cv2.DIST_L2, 5)
        
        # We find the center path by looking for local maxima in horizontal slices
        num_points = 10
        raw_path = []
        slice_h = h // num_points
        
        for i in range(num_points):
            y = h - (i * slice_h) - (slice_h // 2)
            if y < 0: break
            
            # Get the horizontal slice at height y
            row_dist = dist_transform[y, :]
            
            # The "Medial Axis" point is where the distance is maximum (furthest from obstacles/edges)
            if np.max(row_dist) > 2:  # Only if there's actual drivable space
                best_x = np.argmax(row_dist)
                raw_path.append((int(best_x), int(y)))

        if not raw_path or len(raw_path) < 3:
            return self._empty_response(h, w)

        # 3. Temporal Smoothing (Sequence Logic)
        smoothed_path = self._apply_temporal_smoothing(raw_path)
        self._prev_path = smoothed_path

        # 4. Spline Interpolation for "Clean Stroke Lines"
        final_path = self._interpolate_path(smoothed_path)

        # 5. Steering and Status
        steering_target = final_path[min(len(final_path)-1, 10)] # Look-ahead point
        direction = self._get_direction(steering_target, w)
        
        # Calculate width ratio at bottom for "narrow" detection
        bottom_row = safe_zone[int(h*0.8), :]
        drivable_width = np.sum(bottom_row > 0) / w

        return {
            "polyline": final_path,
            "steering_target": steering_target,
            "steering_direction": direction,
            "drivable_width_ratio": float(drivable_width),
        }

    def _interpolate_path(self, points):
        """Increase point density and smooth using cubic interpolation."""
        points = np.array(points)
        x = points[:, 0]
        y = points[:, 1]
        
        # We want to interpolate along the y-axis (bottom to top)
        # But y might not be strictly monotonic due to smoothing
        # So we use an index-based interpolation
        t = np.linspace(0, 1, len(points))
        t_new = np.linspace(0, 1, 30) # Increase to 30 points for smooth stroke
        
        fx = interp1d(t, x, kind='cubic')
        fy = interp1d(t, y, kind='cubic')
        
        interp_points = []
        for i in range(len(t_new)):
            interp_points.append((int(fx(t_new[i])), int(fy(t_new[i]))))
            
        return interp_points

    def _apply_temporal_smoothing(self, current_path):
        if self._prev_path is None or len(current_path) != len(self._prev_path):
            return current_path
        
        smoothed = []
        for curr, prev in zip(current_path, self._prev_path):
            sx = int(curr[0] * self.alpha + prev[0] * (1 - self.alpha))
            sy = int(curr[1] * self.alpha + prev[1] * (1 - self.alpha))
            smoothed.append((sx, sy))
        return smoothed

    def _get_direction(self, target, w):
        center_x = w / 2
        deviation = (target[0] - center_x) / center_x
        if deviation < -0.15: return "LEFT"
        if deviation > 0.15: return "RIGHT"
        return "CENTER"

    def _empty_response(self, h, w):
        return {
            "polyline": [],
            "steering_target": (w // 2, h // 2),
            "steering_direction": "CENTER",
            "drivable_width_ratio": 0.0,
        }
