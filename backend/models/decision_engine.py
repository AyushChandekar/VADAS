from models.detector import Detection
from configs.classes import CRITICAL_CLASSES, VEHICLE_CLASSES
from collections import deque
import numpy as np

class DecisionEngine:
    """
    Robust, stable, and explainable driving decision engine.
    Uses temporal smoothing and priority handling to prevent flickering.
    """

    # Proximity zones (normalized y-coordinates)
    ZONE_STOP = 0.88      # Bottom 12% - immediate danger
    ZONE_SLOW = 0.65      # Middle-bottom - caution area
    
    # Path/Road thresholds
    MIN_DRIVABLE_WIDTH = 0.15
    STEERING_THRESHOLD = 0.20 # Normalized deviation for turning

    def __init__(self, history_len: int = 10):
        # Memory for temporal stability
        self.decision_history = deque(maxlen=history_len)
        self.reason_history = deque(maxlen=history_len)
        
        # Priority mapping (Higher value = higher priority)
        self.priority = {
            "STOP": 100,
            "SLOW DOWN": 80,
            "MOVE LEFT": 60,
            "MOVE RIGHT": 60,
            "MOVE FORWARD": 20,
            "INITIALIZING": 0
        }

    def decide(self, detections: list[Detection], trajectory: dict,
               frame_h: int, frame_w: int) -> dict:
        """
        Processes frame data and returns a stable decision with XAI reasoning.
        """
        # 1. Gather raw context
        steering_dir = trajectory.get("steering_direction", "CENTER")
        drivable_width = trajectory.get("drivable_width_ratio", 0.0)
        has_path = len(trajectory.get("polyline", [])) >= 3
        
        # 2. Evaluate environmental conditions (Raw Candidates)
        candidates = []

        # -- Condition: Obstacles --
        for det in detections:
            rel_bottom = det.bottom_y / frame_h
            
            # STOP Conditions
            if rel_bottom > self.ZONE_STOP:
                if det.class_name in CRITICAL_CLASSES:
                    candidates.append(("STOP", f"Critical obstacle ({det.class_name}) very close"))
                elif det.class_name in VEHICLE_CLASSES:
                    candidates.append(("STOP", f"Vehicle too close ahead"))

            # SLOW DOWN Conditions
            elif rel_bottom > self.ZONE_SLOW:
                if det.class_name in CRITICAL_CLASSES:
                    candidates.append(("SLOW DOWN", f"Caution: {det.class_name} ahead"))
                elif det.class_name in VEHICLE_CLASSES:
                    candidates.append(("SLOW DOWN", f"Following vehicle ahead"))

        # -- Condition: Road/Path --
        if not has_path or drivable_width < 0.05:
            candidates.append(("STOP", "No drivable path detected"))
        elif drivable_width < self.MIN_DRIVABLE_WIDTH:
            candidates.append(("SLOW DOWN", f"Narrow road (width: {drivable_width:.0%})"))

        # -- Condition: Turning (Guidance) --
        if steering_dir == "LEFT":
            candidates.append(("MOVE LEFT", "Path curves left"))
        elif steering_dir == "RIGHT":
            candidates.append(("MOVE RIGHT", "Path curves right"))

        # -- Default: Move Forward --
        if not candidates:
            candidates.append(("MOVE FORWARD", "Path ahead is clear"))

        # 3. Priority Handling (Pick the most critical candidate)
        best_action, best_reason = self._pick_highest_priority(candidates)

        # 4. Temporal Stability (Hysteresis/Voting)
        # We don't change decisions unless the new one persists or has much higher priority
        stable_action, stable_reason = self._smooth_decision(best_action, best_reason)

        return {
            "action": stable_action,
            "reason": stable_reason,
            "confidence": 0.9, # Rule-based confidence
        }

    def _pick_highest_priority(self, candidates):
        """Returns the candidate with the highest priority score."""
        top_action, top_reason = candidates[0]
        max_p = self.priority.get(top_action, 0)
        
        for action, reason in candidates[1:]:
            p = self.priority.get(action, 0)
            if p > max_p:
                max_p = p
                top_action, top_reason = action, reason
                
        return top_action, top_reason

    def _smooth_decision(self, action, reason):
        """Applies a voting mechanism to prevent flickering."""
        self.decision_history.append(action)
        self.reason_history.append(reason)
        
        # If the top priority action (like STOP) is requested, we act faster
        if action == "STOP":
            # Just 2 consecutive frames of STOP are enough to trigger it
            if list(self.decision_history)[-2:].count("STOP") >= 2:
                return action, reason
        
        # For other actions, use a majority vote over the last N frames
        unique_actions = set(self.decision_history)
        counts = {a: list(self.decision_history).count(a) for a in unique_actions}
        
        # Get action with max counts
        winner = max(counts, key=counts.get)
        
        # If winner is different from current action, but current has high priority, stick to winner 
        # only if it has a solid majority (e.g. 60% of history)
        if counts[winner] >= (self.decision_history.maxlen * 0.6):
            # Find the most recent reason for the winning action
            win_reason = reason
            for i in range(len(self.decision_history)-1, -1, -1):
                if self.decision_history[i] == winner:
                    win_reason = self.reason_history[i]
                    break
            return winner, win_reason
            
        # Fallback to the most recent decision if no clear majority
        return self.decision_history[-1], self.reason_history[-1]
