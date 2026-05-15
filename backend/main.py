import io
import os
import sys
import threading
import time
import urllib.request

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

# Add backend/ to path so internal imports (models, inference, configs) resolve
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from inference.camera import CameraCapture, VideoFileCapture
from inference.pipeline import InferencePipeline

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(BACKEND_DIR, "checkpoints")
UPLOADED_VIDEO_PATH = os.path.join(ROOT_DIR, "uploaded_video.mp4")
STATIC_DIR = os.path.join(ROOT_DIR, "frontend", "dist")

pipeline: InferencePipeline | None = None
camera: CameraCapture | None = None
inference_thread: threading.Thread | None = None
running = False


def download_checkpoint(url: str, dest_path: str) -> None:
    print(f"Downloading checkpoint from {url} to {dest_path}")
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    try:
        urllib.request.urlretrieve(url, dest_path)
        print(f"Downloaded checkpoint to {dest_path}")
    except Exception as exc:
        print(f"Failed to download checkpoint: {exc}")
        raise


def resolve_checkpoint(path: str, env_var: str, friendly_name: str) -> str:
    if os.path.exists(path):
        return path
    url = os.environ.get(env_var)
    if url:
        download_checkpoint(url, path)
        return path
    print(f"WARNING: {friendly_name} not found at {path}")
    print(f"Set {env_var} to a downloadable checkpoint URL or upload the file to backend/checkpoints/")
    return path


def create_capture(source: str | int) -> CameraCapture:
    if isinstance(source, str):
        if os.path.exists(source):
            return VideoFileCapture(source)
        if source.isdigit() or (source.startswith("-") and source[1:].isdigit()):
            return CameraCapture(int(source))
        return CameraCapture(source)
    return CameraCapture(source)


def stop_camera() -> None:
    global camera
    if camera is not None:
        try:
            camera.release()
        except Exception:
            pass
    camera = None


def stop_inference() -> None:
    global running, inference_thread
    running = False
    if inference_thread is not None and inference_thread.is_alive():
        inference_thread.join(timeout=2.0)


def start_inference() -> None:
    global inference_thread, running
    if pipeline is None or camera is None:
        return
    if inference_thread is not None and inference_thread.is_alive():
        return
    running = True
    inference_thread = threading.Thread(target=_inference_loop, daemon=True)
    inference_thread.start()
    print("Inference loop started.")


def _inference_loop() -> None:
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


app = FastAPI(title="VADAS-India API", version="1.0")

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Routes are defined below ---


@app.on_event("startup")
def startup() -> None:
    global pipeline, camera

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    yolo_path = resolve_checkpoint(
        os.path.join(CHECKPOINT_DIR, "yolo_idd_best.pt"),
        "YOLO_CHECKPOINT_URL",
        "YOLO checkpoint",
    )
    unet_path = resolve_checkpoint(
        os.path.join(CHECKPOINT_DIR, "unet_drivable_best.pth"),
        "UNET_CHECKPOINT_URL",
        "U-Net checkpoint",
    )

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if os.path.exists(yolo_path) and os.path.exists(unet_path):
        pipeline = InferencePipeline(yolo_path, unet_path, device=device)
    else:
        print("Pipeline will not load until checkpoint files are present.")

    video_source = os.environ.get("VIDEO_SOURCE") or os.environ.get("CAMERA_SOURCE")
    project_root = ROOT_DIR
    fallback_video = os.path.join(project_root, "test_video.mp4")
    if video_source:
        try:
            camera = create_capture(video_source)
            print(f"Opening video source from environment: {video_source}")
        except RuntimeError as e:
            print(f"Video source error: {e}")
            camera = None
    elif os.path.exists(fallback_video):
        try:
            camera = VideoFileCapture(fallback_video)
            print(f"Using fallback video: {fallback_video}")
        except RuntimeError as e:
            print(f"Fallback video error: {e}")
            camera = None
    else:
        print("No initial video source configured. Upload a video through /api/upload_video.")
        camera = None

    if pipeline and camera:
        start_inference()
    else:
        print("Pipeline not started — waiting for models or video source.")


@app.on_event("shutdown")
def shutdown() -> None:
    global running
    running = False
    if camera is not None:
        camera.release()


@app.post("/api/upload_video")
async def upload_video(file: UploadFile = File(...)) -> JSONResponse:
    if not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="Only video files are accepted.")

    # 40MB limit
    MAX_SIZE = 40 * 1024 * 1024
    contents = await file.read()
    if len(contents) > MAX_SIZE:
        raise HTTPException(status_code=413, detail="File too large. Maximum size is 40MB.")

    try:
        with open(UPLOADED_VIDEO_PATH, "wb") as dest_file:
            dest_file.write(contents)
        # Force flush to disk
        dest_file.flush()
        os.fsync(dest_file.fileno())
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Failed to save upload: {exc}")

    stop_inference()
    stop_camera()
    
    # Wait a bit for file to be ready
    time.sleep(1.0)

    try:
        # Re-initialize pipeline if it's not loaded yet
        if pipeline is None:
            print("Pipeline is None, triggering manual startup...")
            try:
                startup()
            except Exception as e:
                print(f"Manual startup failed: {e}")
                # Don't raise here, we might still be able to open the video
        
        camera = VideoFileCapture(UPLOADED_VIDEO_PATH)
        print(f"Uploaded video capture created. Pipeline state: {pipeline is not None}")
    except Exception as exc:
        print(f"Error opening uploaded video: {exc}")
        return JSONResponse(
            status_code=500,
            content={"detail": f"Cannot open uploaded video: {str(exc)}"}
        )

    if pipeline is not None and camera is not None:
        try:
            start_inference()
            print("Inference started for uploaded video.")
        except Exception as e:
            print(f"Error starting inference: {e}")

    return JSONResponse({"detail": "Video uploaded successfully. Inference will begin shortly."})


@app.get("/api/frame")
def get_frame() -> Response:
    if pipeline is None:
        return Response(status_code=503, content="Pipeline not initialized")

    jpeg = pipeline.get_latest_frame_jpeg()
    if jpeg is None:
        # Return a 1x1 transparent pixel instead of 204 to avoid frontend errors
        transparent_pixel = b'\xff\xd8\xff\xdb\x00\x43\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a\x1f\x1e\x1d\x1a\x1c\x1c\x20\x24\x2e\x27\x20\x22\x2c\x23\x1c\x1c\x28\x37\x29\x2c\x30\x31\x34\x34\x34\x1f\x27\x39\x3d\x38\x32\x3c\x2e\x33\x34\x32\xff\xcb\x00\x11\x08\x00\x01\x00\x01\x03\x01\x11\x00\x02\x11\x01\x03\x11\x01\xff\xda\x00\x08\x03\x01\x00\x02\x11\x03\x11\x00\x3f\x00\xa2\x8a\x0f\xff\xd9'
        return Response(content=transparent_pixel, media_type="image/jpeg")

    return StreamingResponse(io.BytesIO(jpeg), media_type="image/jpeg")


@app.get("/api/status")
def get_status() -> JSONResponse:
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
def get_detections() -> JSONResponse:
    if pipeline is None:
        return JSONResponse([])
    return JSONResponse(pipeline.get_latest_detections())


@app.get("/api/trajectory")
def get_trajectory() -> JSONResponse:
    if pipeline is None:
        return JSONResponse({})
    return JSONResponse(pipeline.get_latest_trajectory())


@app.get("/api/health")
def health() -> JSONResponse:
    import torch
    return JSONResponse({
        "status": "ok",
        "pipeline_loaded": pipeline is not None,
        "camera_connected": bool(camera and camera.is_opened),
        "gpu_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    })

# --- Static File Serving ---

@app.get("/")
async def serve_index():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return Response(content=open(index_path, "rb").read(), media_type="text/html")
    return Response(status_code=404, content="Frontend build not found.")

# Mount other static files (assets, etc.)
if os.path.isdir(STATIC_DIR):
    app.mount("/", StaticFiles(directory=STATIC_DIR), name="static")
