"""IDD class definitions and mappings for both detection and segmentation."""

# ── Detection ─────────────────────────────────────────────────────────
# Maps IDD annotation names (various spellings) → canonical class name
DETECTION_NAME_MAP = {
    "car": "car",
    "truck": "truck",
    "bus": "bus",
    "motorcycle": "motorcycle",
    "motorbike": "motorcycle",
    "bicycle": "bicycle",
    "autorickshaw": "autorickshaw",
    "auto rickshaw": "autorickshaw",
    "auto": "autorickshaw",
    "rickshaw": "autorickshaw",
    "vehicle fallback": "vehicle_fallback",
    "vehicle_fallback": "vehicle_fallback",
    "caravan": "vehicle_fallback",
    "trailer": "vehicle_fallback",
    "person": "person",
    "pedestrian": "person",
    "rider": "rider",
    "animal": "animal",
    "traffic sign": "traffic_sign",
    "traffic_sign": "traffic_sign",
    "trafficsign": "traffic_sign",
    "traffic light": "traffic_light",
    "traffic_light": "traffic_light",
    "trafficlight": "traffic_light",
    "pole": "pole",
    "wall": "wall",
    "fence": "fence",
    "guard rail": "fence",
    "guardrail": "fence",
}

# Canonical detection class list (sorted, matches training order)
DETECTION_CLASSES = sorted(set(DETECTION_NAME_MAP.values()))

# ── Segmentation ──────────────────────────────────────────────────────
# IDD level3 label IDs
# 0=road, 1=drivable fallback, 2=sidewalk, 3=non-drivable fallback,
# 4=living things, 5=2-wheeler, 6=4-wheeler, 7=far objects,
# 8=sky, 9=misc, 255=void
DRIVABLE_LEVEL3_IDS = {0, 1}  # road + drivable fallback

# Polygon label names considered drivable
DRIVABLE_POLYGON_NAMES = {"road", "drivable fallback", "drivable_fallback"}

# All IDD segmentation labels (level 4 / fine)
SEG_LABELS = [
    "road", "drivable fallback", "sidewalk", "non-drivable fallback",
    "person", "rider", "motorcycle", "bicycle", "autorickshaw",
    "car", "truck", "bus", "vehicle fallback",
    "curb", "wall", "fence", "guard rail",
    "billboard", "traffic sign", "traffic light", "pole",
    "obs-str-bar-fallback", "building", "bridge", "vegetation",
    "sky", "misc", "tunnel", "fallback background",
    "unlabeled", "ego vehicle", "rectification border", "out of roi",
    "rail track", "terrain",
]
