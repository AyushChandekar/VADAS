import os
import sys
import threading
import time
import io
import cv2
import numpy as np

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, JSONResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles

# Add backend/ to path so internal imports (models, inference, configs) resolve
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from inference.pipeline import InferencePipeline
from inference.camera import CameraCapture, VideoFileCapture, BufferCapture

# ── Globals (initialized on startup) ─────────────────────────────────
pipeline: InferencePipeline | None = None
camera: CameraCapture | VideoFileCapture | BufferCapture | None = None
inference_thread: threading.Thread | None = None
running = False


def _inference_loop():
    """Continuously grab frames and run the AI pipeline."""
    global running
    print("Inference loop: waiting for first frame...")
    frame_count = 0
    while running:
        try:
            if camera is None:
                time.sleep(0.1)
                continue

            frame = camera.get_latest_frame()
            if frame is not None:
                if frame_count == 0:
                    print(f"Inference loop: first frame received! Shape: {frame.shape}")
                pipeline.process_frame(frame)
                frame_count += 1
                if frame_count % 100 == 0:
                    print(f"Inference loop: processed {frame_count} frames")
            else:
                time.sleep(0.01)
        except Exception as e:
            print(f"Inference loop error: {e}")
            time.sleep(0.5)


app = FastAPI(title="VADAS-India API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup():
    global pipeline, camera, inference_thread, running

    backend_dir = os.path.dirname(os.path.abspath(__file__))
    yolo_path = os.path.join(backend_dir, "checkpoints", "yolo_idd_best.pt")
    unet_path = os.path.join(backend_dir, "checkpoints", "unet_drivable_best.pth")

    # Check if model files exist
    if not os.path.exists(yolo_path):
        print(f"WARNING: YOLO checkpoint not found at {yolo_path}")
    if not os.path.exists(unet_path):
        print(f"WARNING: U-Net checkpoint not found at {unet_path}")

    # Determine device
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Initialize pipeline (only if checkpoints exist)
    if os.path.exists(yolo_path) and os.path.exists(unet_path):
        pipeline = InferencePipeline(yolo_path, unet_path, device=device)

    # Initialize camera
    project_root = os.path.dirname(backend_dir)
    video_test = os.path.join(project_root, "test_video.mp4")
    cam_source = os.environ.get("CAMERA_SOURCE", "0")

    try:
        if os.path.exists(video_test):
            print(f"Using test video: {video_test}")
            camera = VideoFileCapture(video_test)
        else:
            source = int(cam_source) if cam_source.replace('-', '').isdigit() else cam_source
            print(f"Opening camera source: {source}")
            camera = CameraCapture(source=source)
    except Exception as e:
        print(f"Camera error: {e}")
        print("Falling back to BufferCapture for client-side streaming.")
        camera = BufferCapture()

    # Start inference loop
    if pipeline:
        running = True
        inference_thread = threading.Thread(target=_inference_loop, daemon=True)
        inference_thread.start()
        print("Inference loop started.")
    else:
        print("Pipeline not started — models not found.")


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
        # Log periodically to avoid spamming
        if time.time() % 5 < 0.1:
            print("API: get_frame returning 204 (no annotated frame yet)")
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
            "camera_connected": False,
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


@app.websocket("/api/ws/stream")
async def websocket_endpoint(websocket: WebSocket):
    global camera
    try:
        await websocket.accept()
        print(f"WebSocket connection accepted from {websocket.client}")

        # If we were using a real camera, switch to buffer for client streaming
        if not isinstance(camera, BufferCapture):
            print("Switching to BufferCapture for WebSocket stream")
            if camera:
                camera.release()
            camera = BufferCapture()

        frame_count = 0
        while True:
            data = await websocket.receive_bytes()
            if not data:
                continue
                
            nparr = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is not None:
                camera.set_frame(frame)
                frame_count += 1
                if frame_count % 100 == 0:
                    print(f"WebSocket: received {frame_count} frames")
            else:
                print("WebSocket: failed to decode frame")
                
    except WebSocketDisconnect:
        print("WebSocket disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")


# ── Static File Serving ──────────────────────────────────────────────

# Path to the frontend dist folder
backend_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(backend_dir)
frontend_dist = os.path.join(project_root, "frontend", "dist")

if os.path.exists(frontend_dist):
    # Mount assets folder if it exists
    assets_path = os.path.join(frontend_dist, "assets")
    if os.path.exists(assets_path):
        app.mount("/assets", StaticFiles(directory=assets_path), name="assets")

    @app.get("/{rest_of_path:path}")
    async def serve_frontend(request: Request, rest_of_path: str):
        # Don't intercept API or WebSocket calls
        if rest_of_path.startswith("api/"):
            return JSONResponse({"detail": "Not Found"}, status_code=404)

        # Handle root path
        if not rest_of_path or rest_of_path == "/":
            return FileResponse(os.path.join(frontend_dist, "index.html"))

        # Check if the requested file exists in dist
        file_path = os.path.join(frontend_dist, rest_of_path)
        if rest_of_path and os.path.isfile(file_path):
            return FileResponse(file_path)

        # Fallback to index.html for client-side routing
        return FileResponse(os.path.join(frontend_dist, "index.html"))
else:
    print(f"WARNING: Frontend dist not found at {frontend_dist}")
    @app.get("/")
    def root_warning():
        return {"error": "Frontend build not found. Please run 'npm run build' in frontend folder."}
