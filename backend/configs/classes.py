# IDD Detection classes (must match training order)
DETECTION_CLASSES = [
    "animal", "autorickshaw", "bicycle", "bus", "car",
    "motorcycle", "person", "rider", "traffic_light",
    "traffic_sign", "truck", "vehicle_fallback",
]

# Classes that are "close danger" — trigger STOP if very near
CRITICAL_CLASSES = {"person", "rider", "animal"}

# Classes that are vehicles — trigger SLOW DOWN
VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle", "bicycle", "autorickshaw", "vehicle_fallback"}

# Segmentation
SEG_IMG_W = 512
SEG_IMG_H = 256
SEG_BASE_CHANNELS = 64

# Detection
DET_IMGSZ = 640
DET_CONF_THRESHOLD = 0.35
DET_IOU_THRESHOLD = 0.45
