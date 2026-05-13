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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

if os.path.isdir(STATIC_DIR):
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
else:
    print(f"WARNING: Static assets directory not found: {STATIC_DIR}")


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

    contents = await file.read()
    try:
        with open(UPLOADED_VIDEO_PATH, "wb") as dest_file:
            dest_file.write(contents)
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Failed to save upload: {exc}")

    stop_inference()
    stop_camera()

    try:
        camera = VideoFileCapture(UPLOADED_VIDEO_PATH)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=f"Cannot open uploaded video: {exc}")

    if pipeline is not None:
        start_inference()

    return JSONResponse({"detail": "Video uploaded successfully. Inference will begin shortly."})


@app.get("/api/frame")
def get_frame() -> Response:
    if pipeline is None:
        return Response(status_code=503, content="Pipeline not initialized")

    jpeg = pipeline.get_latest_frame_jpeg()
    if jpeg is None:
        return Response(status_code=204)

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
