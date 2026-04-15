from models.detector import Detection
from configs.classes import CRITICAL_CLASSES, VEHICLE_CLASSES


class DecisionEngine:
    """Rule-based driving decision from detections + trajectory."""

    # Proximity zones based on bounding box bottom-y relative to frame height
    ZONE_CLOSE = 0.85    # bottom 15% of frame -> very close
    ZONE_MEDIUM = 0.65   # 65-85% -> medium distance
    # above 65% -> far

    # Minimum drivable width to consider road "narrow"
    NARROW_THRESHOLD = 0.30

    def decide(self, detections: list[Detection], trajectory: dict,
               frame_h: int, frame_w: int) -> dict:
        """
        Args:
            detections: list of Detection from YOLO
            trajectory: dict from TrajectoryPlanner.plan()
            frame_h, frame_w: original frame dimensions
        Returns:
            dict with keys:
                - action: str (STOP, SLOW DOWN, TURN LEFT, TURN RIGHT,
                          DRIVE FORWARD, HORN, WIPER)
                - reason: str (human-readable explanation)
                - confidence: float (0-1)
        """
        direction = trajectory.get("steering_direction", "CENTER")
        drivable_width = trajectory.get("drivable_width_ratio", 0.0)
        has_path = len(trajectory.get("polyline", [])) >= 2

        # Classify detections by proximity
        close_critical = []
        close_vehicles = []
        medium_critical = []
        medium_vehicles = []
        horn_candidates = []

        for det in detections:
            rel_bottom = det.bottom_y / frame_h
            rel_area = det.area / (frame_h * frame_w)

            if rel_bottom > self.ZONE_CLOSE or rel_area > 0.08:
                # Very close
                if det.class_name in CRITICAL_CLASSES:
                    close_critical.append(det)
                elif det.class_name in VEHICLE_CLASSES:
                    close_vehicles.append(det)
            elif rel_bottom > self.ZONE_MEDIUM or rel_area > 0.03:
                # Medium distance
                if det.class_name in CRITICAL_CLASSES:
                    medium_critical.append(det)
                elif det.class_name in VEHICLE_CLASSES:
                    medium_vehicles.append(det)

                # Horn candidates: person/animal in drivable zone at warning distance
                if det.class_name in ("person", "animal", "rider"):
                    horn_candidates.append(det)

        # Decision priority (highest to lowest)

        # 1. STOP — critical object very close OR no drivable path
        if close_critical:
            names = ", ".join(d.class_name for d in close_critical)
            return {
                "action": "STOP",
                "reason": f"Critical object very close: {names}",
                "confidence": 0.95,
            }

        if not has_path or drivable_width < 0.05:
            return {
                "action": "STOP",
                "reason": "No drivable path detected",
                "confidence": 0.85,
            }

        # 2. SLOW DOWN — vehicle close OR critical at medium OR narrow road
        if close_vehicles:
            names = ", ".join(d.class_name for d in close_vehicles[:3])
            return {
                "action": "SLOW DOWN",
                "reason": f"Vehicle close ahead: {names}",
                "confidence": 0.85,
            }

        if medium_critical:
            names = ", ".join(d.class_name for d in medium_critical)
            return {
                "action": "SLOW DOWN",
                "reason": f"Caution — {names} ahead",
                "confidence": 0.80,
            }

        if drivable_width < self.NARROW_THRESHOLD:
            return {
                "action": "SLOW DOWN",
                "reason": f"Narrow road (width: {drivable_width:.0%})",
                "confidence": 0.70,
            }

        # 3. HORN — person/animal at warning distance
        if horn_candidates:
            return {
                "action": "HORN",
                "reason": f"Warning: {horn_candidates[0].class_name} on road ahead",
                "confidence": 0.65,
            }

        # 4. TURN — based on trajectory steering
        if direction == "LEFT" and medium_vehicles:
            return {
                "action": "TURN LEFT",
                "reason": "Path curves left, vehicles ahead",
                "confidence": 0.75,
            }
        if direction == "RIGHT" and medium_vehicles:
            return {
                "action": "TURN RIGHT",
                "reason": "Path curves right, vehicles ahead",
                "confidence": 0.75,
            }
        if direction == "LEFT":
            return {
                "action": "TURN LEFT",
                "reason": "Road curves left",
                "confidence": 0.70,
            }
        if direction == "RIGHT":
            return {
                "action": "TURN RIGHT",
                "reason": "Road curves right",
                "confidence": 0.70,
            }

        # 5. DRIVE FORWARD — all clear
        return {
            "action": "DRIVE FORWARD",
            "reason": "Clear road ahead",
            "confidence": 0.90,
        }
