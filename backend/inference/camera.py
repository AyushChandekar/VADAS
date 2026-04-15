import cv2
import threading
import time
import sys


def open_camera_by_name(name: str, width=1280, height=720):
    """Open a camera by device name using DirectShow (Windows)."""
    if sys.platform == "win32":
        cap = cv2.VideoCapture(f"video={name}", cv2.CAP_DSHOW)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            return cap
    return None


class CameraCapture:
    """Thread-safe camera capture — always holds the latest frame."""

    def __init__(self, source=0, width=1280, height=720):
        # Try opening by name first (e.g. "DroidCam Video")
        if isinstance(source, str) and not source.isdigit():
            self.cap = open_camera_by_name(source, width, height)
            if self.cap is None or not self.cap.isOpened():
                raise RuntimeError(f"Cannot open camera by name: {source}")
        else:
            idx = int(source) if isinstance(source, str) else source
            self.cap = cv2.VideoCapture(idx)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera source: {source}")

        self.frame = None
        self.lock = threading.Lock()
        self.running = True
        self.fps = 0.0
        self._frame_count = 0
        self._fps_time = time.time()

        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

    def _capture_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame
                self._frame_count += 1
                elapsed = time.time() - self._fps_time
                if elapsed >= 1.0:
                    self.fps = self._frame_count / elapsed
                    self._frame_count = 0
                    self._fps_time = time.time()
            else:
                time.sleep(0.01)

    def get_latest_frame(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def release(self):
        self.running = False
        self.thread.join(timeout=2.0)
        self.cap.release()

    @property
    def is_opened(self):
        return self.cap.isOpened() and self.running


class VideoFileCapture(CameraCapture):
    """Same interface but reads from a video file (for testing without a camera)."""

    def __init__(self, video_path: str):
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        self.frame = None
        self.lock = threading.Lock()
        self.running = True
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._frame_count = 0
        self._fps_time = time.time()
        self._delay = 1.0 / self.fps

        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

    def _capture_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame
                time.sleep(self._delay)
            else:
                # Loop video
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
