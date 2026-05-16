import io
import os
import sys
import threading
import time
import urllib.request
import cv2
import numpy as np

from fastapi import FastAPI, File, HTTPException, UploadFile, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles

# Add backend/ to path so internal imports resolve
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from inference.camera import CameraCapture, VideoFileCapture
from inference.pipeline import InferencePipeline

# --- Global Paths & Config ---
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(BACKEND_DIR, "checkpoints")
UPLOADED_VIDEO_PATH = os.path.join(ROOT_DIR, "uploaded_video.mp4")
PROCESSED_VIDEO_PATH = os.path.join(ROOT_DIR, "processed_video.mp4")
STATIC_DIR = os.path.join(ROOT_DIR, "frontend", "dist")

# --- Global State ---
pipeline: InferencePipeline | None = None
camera: CameraCapture | None = None
inference_thread: threading.Thread | None = None
running = False

processing_state = {
    "status": "IDLE", # IDLE, PROCESSING, COMPLETED, ERROR
    "progress": 0,
    "current_chunk": 0,
    "total_chunks": 0,
    "current_frame": 0,
    "total_frames": 0,
    "error": None
}

# --- Helper Functions ---

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

def _inference_loop() -> None:
    global running
    while running:
        try:
            frame = camera.get_latest_frame()
            if frame is not None:
                pipeline.process_frame(frame)
            else:
                time.sleep(0.05)
        except Exception as e:
            time.sleep(0.5)

# --- Video Processing Worker ---

def process_video_worker():
    global processing_state, pipeline, PROCESSED_VIDEO_PATH, UPLOADED_VIDEO_PATH
    print(f"DEBUG: Worker started. Source: {UPLOADED_VIDEO_PATH}")
    try:
        source_path = UPLOADED_VIDEO_PATH
        if not os.path.exists(source_path):
            source_path = "/tmp/uploaded_video.mp4"
            print(f"DEBUG: Falling back to {source_path}")
            
        if not os.path.exists(source_path):
            print(f"ERROR: Video not found at {source_path}")
            processing_state["status"] = "ERROR"
            processing_state["error"] = "Original video not found"
            return

        cap = cv2.VideoCapture(source_path)
        if not cap.isOpened():
            print(f"ERROR: Could not open video file {source_path}")
            processing_state["status"] = "ERROR"
            processing_state["error"] = "Failed to open video file"
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"DEBUG: Video info: {width}x{height}, {fps} FPS, {total_frames} frames")

        # 3 second chunks
        frames_per_chunk = int(fps * 3)
        total_chunks = (total_frames + frames_per_chunk - 1) // frames_per_chunk

        processing_state["status"] = "PROCESSING"
        processing_state["total_frames"] = total_frames
        processing_state["total_chunks"] = total_chunks
        processing_state["current_frame"] = 0
        processing_state["current_chunk"] = 0

        # Output path check
        output_path = PROCESSED_VIDEO_PATH
        try:
            with open(output_path, "wb") as f: pass
            print(f"DEBUG: Writing to {output_path}")
        except:
            output_path = "/tmp/processed_video.mp4"
            PROCESSED_VIDEO_PATH = output_path
            print(f"DEBUG: Writing to fallback {output_path}")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if not out.isOpened():
            print("ERROR: Could not open VideoWriter")
            processing_state["status"] = "ERROR"
            processing_state["error"] = "Failed to initialize video encoder"
            return

        if pipeline is None:
            print("DEBUG: Re-initializing pipeline...")
            startup()

        print("DEBUG: Starting frame loop...")
        while True:
            ret, frame = cap.read()
            if not ret:
                print("DEBUG: End of video stream")
                break
            
            # Process frame
            annotated = pipeline.process_frame(frame)
            out.write(annotated)

            processing_state["current_frame"] += 1
            processing_state["progress"] = int((processing_state["current_frame"] / total_frames) * 100)
            
            # Update chunk status
            new_chunk = (processing_state["current_frame"] // frames_per_chunk)
            if new_chunk != processing_state["current_chunk"]:
                processing_state["current_chunk"] = new_chunk
                print(f"Processed chunk {new_chunk}/{total_chunks}")

        cap.release()
        out.release()
        processing_state["status"] = "COMPLETED"
        processing_state["progress"] = 100
        print(f"Processing completed successfully: {output_path}")

    except Exception as e:
        print(f"CRITICAL: Processing worker crashed: {e}")
        import traceback
        traceback.print_exc()
        processing_state["status"] = "ERROR"
        processing_state["error"] = str(e)

# --- FastAPI App ---

app = FastAPI(title="VADAS-India API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    if os.path.exists(yolo_path) and os.path.exists(unet_path):
        pipeline = InferencePipeline(yolo_path, unet_path, device=device)

@app.post("/api/upload_video")
async def upload_video(file: UploadFile = File(...)) -> JSONResponse:
    global UPLOADED_VIDEO_PATH
    try:
        if not file.content_type.startswith("video/"):
            return JSONResponse(status_code=400, content={"detail": "Only video files are accepted."})
        MAX_SIZE = 40 * 1024 * 1024
        contents = await file.read()
        if len(contents) > MAX_SIZE:
            return JSONResponse(status_code=413, content={"detail": "File too large. Maximum size is 40MB."})

        save_path = UPLOADED_VIDEO_PATH
        try:
            with open(save_path, "wb") as f:
                f.write(contents)
                f.flush()
                os.fsync(f.fileno())
        except:
            save_path = "/tmp/uploaded_video.mp4"
            with open(save_path, "wb") as f:
                f.write(contents)
        
        UPLOADED_VIDEO_PATH = save_path
        return JSONResponse({"detail": "Video uploaded successfully."})
    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": str(e)})

@app.post("/api/process")
async def start_processing(background_tasks: BackgroundTasks):
    global processing_state
    if processing_state["status"] == "PROCESSING":
        return JSONResponse(status_code=400, content={"detail": "Already processing"})
    
    processing_state = {
        "status": "IDLE",
        "progress": 0,
        "current_chunk": 0,
        "total_chunks": 0,
        "current_frame": 0,
        "total_frames": 0,
        "error": None
    }
    background_tasks.add_task(process_video_worker)
    return JSONResponse({"detail": "Processing started"})

@app.get("/api/process/status")
def get_processing_status():
    return JSONResponse(processing_state)

@app.get("/api/video/original")
def get_original_video():
    if os.path.exists(UPLOADED_VIDEO_PATH):
        return FileResponse(UPLOADED_VIDEO_PATH, media_type="video/mp4")
    return Response(status_code=404)

@app.get("/api/video/processed")
def get_processed_video():
    if os.path.exists(PROCESSED_VIDEO_PATH):
        return FileResponse(PROCESSED_VIDEO_PATH, media_type="video/mp4")
    return Response(status_code=404)

@app.get("/api/stream")
async def stream_video():
    path = UPLOADED_VIDEO_PATH
    if not os.path.exists(path):
        path = "/tmp/uploaded_video.mp4"
    
    if not os.path.exists(path):
        return Response(status_code=404)
    
    def generate():
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        delay = 1.0 / fps
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process the frame in real-time (uses GPU if available)
            if pipeline is not None:
                processed = pipeline.process_frame(frame)
            else:
                processed = frame
                
            _, buffer = cv2.imencode('.jpg', processed)
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            
            # Control speed to match original video
            time.sleep(delay * 0.1) 
            
        cap.release()

    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/api/health")
def health() -> JSONResponse:
    import torch
    return JSONResponse({
        "status": "ok",
        "pipeline_loaded": pipeline is not None,
        "gpu_available": torch.cuda.is_available(),
    })

@app.get("/api/stream")
async def stream_video():
    if not os.path.exists(UPLOADED_VIDEO_PATH):
        return Response(status_code=404)
    
    def generate():
        cap = cv2.VideoCapture(UPLOADED_VIDEO_PATH)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        delay = 1.0 / fps
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process the frame in real-time (uses GPU if available)
            if pipeline is not None:
                processed = pipeline.process_frame(frame)
            else:
                processed = frame
                
            _, buffer = cv2.imencode('.jpg', processed)
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            
            # Control speed
            time.sleep(delay * 0.1) # Aggressive speed for real-time feel
            
        cap.release()

    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")

# --- Static File Serving ---
@app.get("/")
async def serve_index():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return Response(content=open(index_path, "rb").read(), media_type="text/html")
    return Response(status_code=404, content="Frontend build not found.")

if os.path.isdir(STATIC_DIR):
    app.mount("/", StaticFiles(directory=STATIC_DIR), name="static")
