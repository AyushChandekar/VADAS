import os
import sys
import threading
import time

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, JSONResponse, StreamingResponse
import io

# Add backend/ to path so internal imports (models, inference, configs) resolve
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from inference.pipeline import InferencePipeline
from inference.camera import CameraCapture, VideoFileCapture

app = FastAPI(title="VADAS-India API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Globals (initialized on startup) ─────────────────────────────────
pipeline: InferencePipeline | None = None
camera: CameraCapture | None = None
inference_thread: threading.Thread | None = None
running = False


def _inference_loop():
    """Continuously grab frames and run the AI pipeline."""
    global running
    print("Inference loop: waiting for first frame...")
    frame_count = 0
    while running:
        try:
            frame = camera.get_latest_frame()
            if frame is not None:
                if frame_count == 0:
                    print(f"Inference loop: first frame received! Shape: {frame.shape}")
                pipeline.process_frame(frame)
                frame_count += 1
                if frame_count % 100 == 0:
                    print(f"Inference loop: processed {frame_count} frames")
            else:
                time.sleep(0.05)
        except Exception as e:
            print(f"Inference loop error: {e}")
            time.sleep(0.5)


@app.on_event("startup")
def startup():
    global pipeline, camera, inference_thread, running

    backend_dir = os.path.dirname(os.path.abspath(__file__))
    yolo_path = os.path.join(backend_dir, "checkpoints", "yolo_idd_best.pt")
    unet_path = os.path.join(backend_dir, "checkpoints", "unet_drivable_best.pth")

    # Check if model files exist
    if not os.path.exists(yolo_path):
        print(f"WARNING: YOLO checkpoint not found at {yolo_path}")
        print("Place yolo_idd_best.pt in backend/checkpoints/ folder")
    if not os.path.exists(unet_path):
        print(f"WARNING: U-Net checkpoint not found at {unet_path}")
        print("Place unet_drivable_best.pth in backend/checkpoints/ folder")

    # Determine device
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Initialize pipeline (only if checkpoints exist)
    if os.path.exists(yolo_path) and os.path.exists(unet_path):
        pipeline = InferencePipeline(yolo_path, unet_path, device=device)

    # Initialize camera
    # Check for video file first (for testing), then live camera
    project_root = os.path.dirname(backend_dir)
    video_test = os.path.join(project_root, "test_video.mp4")
    cam_source = os.environ.get("CAMERA_SOURCE", "0")

    try:
        if os.path.exists(video_test):
            print(f"Using test video: {video_test}")
            camera = VideoFileCapture(video_test)
        else:
            source = int(cam_source) if cam_source.replace('-','').isdigit() else cam_source
            print(f"Opening camera source: {source}")
            camera = CameraCapture(source=source)
    except RuntimeError as e:
        print(f"Camera error: {e}")
        print("Start the server anyway — camera can be connected later.")
        camera = None

    # Start inference loop
    if pipeline and camera:
        running = True
        inference_thread = threading.Thread(target=_inference_loop, daemon=True)
        inference_thread.start()
        print("Inference loop started.")
    else:
        print("Pipeline not started — waiting for models/camera.")


@app.on_event("shutdown")
def shutdown():
    global running
    running = False
    if camera:
        camera.release()


# ── API Endpoints ─────────────────────────────────────────────────────

@app.get("/api/frame")
def get_frame():
    """Returns the latest annotated frame as JPEG."""
    if pipeline is None:
        return Response(status_code=503, content="Pipeline not initialized")

    jpeg = pipeline.get_latest_frame_jpeg()
    if jpeg is None:
        return Response(status_code=204)

    return StreamingResponse(io.BytesIO(jpeg), media_type="image/jpeg")


@app.get("/api/status")
def get_status():
    """Returns current driving decision + FPS."""
    if pipeline is None:
        return JSONResponse({
            "action": "OFFLINE",
            "reason": "Pipeline not initialized",
            "confidence": 0,
            "fps": 0,
            "camera_connected": bool(camera and camera.is_opened),
        })

    status = pipeline.get_latest_status()
    status["camera_connected"] = bool(camera and camera.is_opened)
    return JSONResponse(status)


@app.get("/api/detections")
def get_detections():
    """Returns list of detected objects."""
    if pipeline is None:
        return JSONResponse([])
    return JSONResponse(pipeline.get_latest_detections())


@app.get("/api/trajectory")
def get_trajectory():
    """Returns trajectory polyline and steering info."""
    if pipeline is None:
        return JSONResponse({})
    return JSONResponse(pipeline.get_latest_trajectory())


@app.get("/api/health")
def health():
    """System health check."""
    import torch
    return JSONResponse({
        "status": "ok",
        "pipeline_loaded": pipeline is not None,
        "camera_connected": bool(camera and camera.is_opened),
        "gpu_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    })
