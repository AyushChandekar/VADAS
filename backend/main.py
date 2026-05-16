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

# Fix Ultralytics config directory for Hugging Face
os.environ["YOLO_CONFIG_DIR"] = "/tmp"

from inference.camera import CameraCapture, VideoFileCapture
from inference.pipeline import InferencePipeline

# --- Global Paths & Config ---
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(BACKEND_DIR, "checkpoints")

# Hugging Face specific path handling
def get_safe_path(filename):
    # Prefer /tmp if on HF (indicated by lack of write permissions in ROOT_DIR or specific env vars)
    if os.environ.get("SPACE_ID"):
        return os.path.join("/tmp", filename)
    try:
        test_file = os.path.join(ROOT_DIR, ".test_write")
        with open(test_file, "w") as f: f.write("test")
        os.remove(test_file)
        return os.path.join(ROOT_DIR, filename)
    except:
        return os.path.join("/tmp", filename)

UPLOADED_VIDEO_PATH = get_safe_path("uploaded_video.mp4")
PROCESSED_VIDEO_PATH = get_safe_path("processed_video.mp4")
STATIC_DIR = os.path.join(ROOT_DIR, "frontend", "dist")

# --- Global State ---
pipeline: InferencePipeline | None = None
processing_state = {
    "status": "IDLE", # IDLE, PROCESSING, COMPLETED, ERROR
    "progress": 0,
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
        
        # Limit resolution for processing efficiency and web compatibility
        TARGET_HEIGHT = 480
        scale = TARGET_HEIGHT / height if height > TARGET_HEIGHT else 1.0
        out_width = int(width * scale)
        out_height = int(height * scale)
        
        print(f"DEBUG: Video info: {width}x{height} -> {out_width}x{out_height}, {fps} FPS, {total_frames} frames")

        # 3 second chunks (removed for simpler status tracking)
        processing_state["status"] = "PROCESSING"
        processing_state["total_frames"] = total_frames
        processing_state["current_frame"] = 0

        # Output path check
        output_path = PROCESSED_VIDEO_PATH
        print(f"DEBUG: Writing to {output_path}")

        # Robust VideoWriter initialization for different environments (HF/Local)
        # We try multiple codecs in order of preference
        codecs = [
            ('mp4v', '.mp4'), # Standard MP4 (most compatible, avoids noisy H.264 hardware errors)
            ('avc1', '.mp4'), # H.264 (often requires OpenH264 or hardware)
            ('XVID', '.avi'), # Very compatible but usually AVI
            ('MJPG', '.mp4')  # Fallback
        ]
        
        out = None
        current_fps = fps
        
        for fourcc_str, ext in codecs:
            try:
                fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
                test_path = output_path if ext == '.mp4' else output_path.replace('.mp4', ext)
                # Side-by-side output: width is doubled
                out = cv2.VideoWriter(test_path, fourcc, current_fps, (out_width * 2, out_height))
                if out.isOpened():
                    print(f"DEBUG: Successfully initialized VideoWriter with codec {fourcc_str}")
                    if ext != '.mp4':
                        PROCESSED_VIDEO_PATH = test_path
                    break
            except Exception as e:
                print(f"DEBUG: Failed to initialize codec {fourcc_str}: {e}")
            
        if out is None or not out.isOpened():
            print("ERROR: Could not open VideoWriter with any codec")
            processing_state["status"] = "ERROR"
            processing_state["error"] = "Failed to initialize video encoder"
            return

        if pipeline is None:
            print("DEBUG: Re-initializing pipeline...")
            startup()

        print("DEBUG: Starting frame loop...")
        is_gpu = pipeline is not None and pipeline.device == "cuda"
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("DEBUG: End of video stream")
                break
            
            # Resize frame for processing if it's large
            h, w = frame.shape[:2]
            if scale < 1.0:
                frame = cv2.resize(frame, (out_width, out_height))
            
            # Process frame
            if pipeline is not None:
                # If on CPU, further optimize if needed
                if not is_gpu and out_width > 640:
                    proc_scale = 640 / out_width
                    proc_frame = cv2.resize(frame, (640, int(out_height * proc_scale)))
                    annotated = pipeline.process_frame(proc_frame)
                    annotated = cv2.resize(annotated, (out_width, out_height))
                else:
                    annotated = pipeline.process_frame(frame)
            else:
                annotated = frame.copy()
            
            # Create side-by-side combined frame for the final video
            # Add labels for clarity in the saved video
            f_label = frame.copy()
            a_label = annotated.copy()
            
            label_scale = out_height / 720 # Scale text based on resolution
            thickness = max(1, int(2 * label_scale))
            font_scale = max(0.5, 1.2 * label_scale)

            cv2.putText(f_label, "ORIGINAL", (20, int(40 * label_scale)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness)
            cv2.putText(a_label, f"AI ANALYSIS {'(GPU)' if is_gpu else '(CPU)'}", (20, int(40 * label_scale)), 
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness)
            
            combined = np.hstack((f_label, a_label))
            out.write(combined)

            processing_state["current_frame"] += 1
            if processing_state["current_frame"] % 5 == 0: # Update progress every 5 frames to reduce overhead
                processing_state["progress"] = int((processing_state["current_frame"] / total_frames) * 100)

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
    global pipeline
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
    global UPLOADED_VIDEO_PATH, PROCESSED_VIDEO_PATH
    try:
        if not file.content_type.startswith("video/"):
            return JSONResponse(status_code=400, content={"detail": "Only video files are accepted."})
        MAX_SIZE = 40 * 1024 * 1024
        contents = await file.read()
        if len(contents) > MAX_SIZE:
            return JSONResponse(status_code=413, content={"detail": "File too large. Maximum size is 40MB."})

        # Ensure we have fresh paths
        UPLOADED_VIDEO_PATH = get_safe_path("uploaded_video.mp4")
        PROCESSED_VIDEO_PATH = get_safe_path("processed_video.mp4")
        
        # Cleanup old processed video if it exists
        if os.path.exists(PROCESSED_VIDEO_PATH):
            os.remove(PROCESSED_VIDEO_PATH)

        with open(UPLOADED_VIDEO_PATH, "wb") as f:
            f.write(contents)
            f.flush()
            os.fsync(f.fileno())
        
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
        return FileResponse(UPLOADED_VIDEO_PATH, media_type="video/mp4", headers={"Accept-Ranges": "bytes"})
    return Response(status_code=404)

@app.get("/api/video/processed")
def get_processed_video():
    if os.path.exists(PROCESSED_VIDEO_PATH):
        return FileResponse(PROCESSED_VIDEO_PATH, media_type="video/mp4", headers={"Accept-Ranges": "bytes"})
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
        if not cap.isOpened():
            print(f"ERROR: Could not open video for streaming: {path}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        frame_delay = 1.0 / fps
        
        # Limit processing resolution for HF CPU efficiency if needed
        # Most HF Spaces are CPU-only unless paid
        is_gpu = pipeline is not None and pipeline.device == "cuda"
        
        # Target height for streaming to save bandwidth and CPU
        STREAM_TARGET_HEIGHT = 360
        
        while True:
            t_start = time.time()
            ret, frame = cap.read()
            if not ret:
                break
            
            # Resize for efficiency before processing
            h, w = frame.shape[:2]
            stream_scale = STREAM_TARGET_HEIGHT / h if h > STREAM_TARGET_HEIGHT else 1.0
            if stream_scale < 1.0:
                frame = cv2.resize(frame, (int(w * stream_scale), STREAM_TARGET_HEIGHT))
                h, w = frame.shape[:2]
            
            # Process the frame
            if pipeline is not None:
                # If on CPU, we might want to resize further before processing to save time
                if not is_gpu and w > 640:
                    proc_scale = 640 / w
                    proc_frame = cv2.resize(frame, (640, int(h * proc_scale)))
                    processed = pipeline.process_frame(proc_frame)
                    processed = cv2.resize(processed, (w, h))
                else:
                    processed = pipeline.process_frame(frame)
            else:
                processed = frame.copy()
            
            # Create a side-by-side comparison
            combined = np.hstack((frame, processed))
            
            # Add labels to the combined frame
            label_scale = h / 720
            font_scale = max(0.5, 1.2 * label_scale)
            thickness = max(1, int(2 * label_scale))
            
            cv2.putText(combined, "ORIGINAL", (20, int(40 * label_scale)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness)
            cv2.putText(combined, f"AI ANALYSIS {'(GPU)' if is_gpu else '(CPU)'}", (w + 20, int(40 * label_scale)), 
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness)
            
            _, buffer = cv2.imencode('.jpg', combined, [cv2.IMWRITE_JPEG_QUALITY, 75])
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            
            # Control speed
            elapsed = time.time() - t_start
            wait_time = max(0.001, frame_delay - elapsed)
            time.sleep(wait_time)
            
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

# --- Static File Serving ---
@app.get("/")
async def serve_index():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return Response(content=open(index_path, "rb").read(), media_type="text/html")
    return Response(status_code=404, content="Frontend build not found.")

if os.path.isdir(STATIC_DIR):
    app.mount("/", StaticFiles(directory=STATIC_DIR), name="static")
