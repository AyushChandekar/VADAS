import numpy as np


class TrajectoryPlanner:
    """Compute a steering trajectory from a binary drivable mask."""

    def __init__(self, num_slices: int = 8, smoothing: int = 3):
        self.num_slices = num_slices
        self.smoothing = smoothing

    def plan(self, drivable_mask: np.ndarray) -> dict:
        """
        Args:
            drivable_mask: Binary mask (H, W) where 255=drivable.
        Returns:
            dict with keys:
                - polyline: list of (x, y) points from bottom to top
                - steering_target: (x, y) of the look-ahead point
                - steering_direction: "LEFT" | "CENTER" | "RIGHT"
                - drivable_width_ratio: fraction of frame width that is drivable
                                         in the bottom third
        """
        h, w = drivable_mask.shape[:2]
        binary = (drivable_mask > 127).astype(np.uint8)

        # Divide into horizontal slices (bottom to top)
        slice_height = h // self.num_slices
        centers = []

        for i in range(self.num_slices):
            y_bottom = h - i * slice_height
            y_top = max(0, y_bottom - slice_height)
            strip = binary[y_top:y_bottom, :]
            cols = np.where(strip.any(axis=0))[0]

            if len(cols) > 0:
                cx = int((cols[0] + cols[-1]) / 2)
                cy = (y_top + y_bottom) // 2
                centers.append((cx, cy))

        if len(centers) < 2:
            return {
                "polyline": [],
                "steering_target": (w // 2, h // 2),
                "steering_direction": "CENTER",
                "drivable_width_ratio": 0.0,
            }

        # Smooth the polyline
        if self.smoothing > 1 and len(centers) >= self.smoothing:
            xs = [c[0] for c in centers]
            ys = [c[1] for c in centers]
            kernel = np.ones(self.smoothing) / self.smoothing
            xs_smooth = np.convolve(xs, kernel, mode="same").astype(int)
            centers = list(zip(xs_smooth.tolist(), ys))

        # Steering target: point at ~1/3 from bottom
        target_idx = min(len(centers) - 1, len(centers) // 3)
        steering_target = centers[target_idx]

        # Compute direction
        frame_center_x = w / 2
        deviation = (steering_target[0] - frame_center_x) / frame_center_x
        if deviation < -0.15:
            direction = "LEFT"
        elif deviation > 0.15:
            direction = "RIGHT"
        else:
            direction = "CENTER"

        # Drivable width in bottom third
        bottom_third = binary[2 * h // 3:, :]
        drivable_cols = np.where(bottom_third.any(axis=0))[0]
        if len(drivable_cols) > 0:
            width_ratio = (drivable_cols[-1] - drivable_cols[0]) / w
        else:
            width_ratio = 0.0

        return {
            "polyline": centers,
            "steering_target": steering_target,
            "steering_direction": direction,
            "drivable_width_ratio": width_ratio,
        }
