---
title: VADAS-India
emoji: 🚗
colorFrom: blue
colorTo: red
sdk: docker
app_port: 7860
pinned: false
---

# VADAS-India — Vehicle Autonomous Driving Assistance System

VADAS-India is an AI-powered driving assistance system optimized for Indian road conditions. It performs real-time object detection (YOLOv8) and drivable area segmentation (U-Net) to provide path guidance and safety alerts.

## 🚀 Deployment on Hugging Face Spaces

This project is ready for deployment on Hugging Face Spaces using Docker.

### Model Checkpoints

The model weights (checkpoints) must be uploaded to your Space:
1. **YOLOv8 Checkpoint**: `yolo_idd_best.pt` → upload to `backend/checkpoints/`
2. **U-Net Checkpoint**: `unet_drivable_best.pth` → upload to `backend/checkpoints/`

**Important**: These files are excluded from git via .dockerignore to avoid large transfers. Upload them directly to your Space using the Hugging Face web interface or git LFS.

### Deployment Steps

1. **Create a new Space** on Hugging Face:
   - Go to https://huggingface.co/spaces
   - Click "Create new Space"
   - Set SDK to "Docker"
   - Choose a name (e.g., "VADAS")

2. **Clone the Space locally**:
   ```bash
   git clone https://huggingface.co/spaces/Aystar/VADAS
   cd VADAS
   ```

3. **Copy your project files** to the Space directory (or push from your existing repo)

4. **Upload model checkpoints** (required):
   - Using web interface: Go to your Space → Files → Upload `backend/checkpoints/yolo_idd_best.pt` and `backend/checkpoints/unet_drivable_best.pth`
   - Or using git LFS:
     ```bash
     git lfs install
     git lfs track "backend/checkpoints/*.pt"
     git lfs track "backend/checkpoints/*.pth"
     git add backend/checkpoints/
     git commit -m "Add model checkpoints"
     git push
     ```

5. **Push to Hugging Face**:
   ```bash
   git add .
   git commit -m "Deploy VADAS to HF Spaces"
   git push
   ```

6. **Monitor the build**: The Space will automatically build and deploy. Check the "Logs" tab for progress.

### Local Development

## 🛠 Features

- **Real-time Inference**: Processes video uploads and shows results instantly.
- **Object Detection**: Identifies vehicles, pedestrians, and obstacles common on Indian roads.
- **Drivable Area**: High-precision segmentation of the road ahead.
- **Path Guidance**: Visualizes the safest trajectory.
- **Safety Alerts**: Dynamic feedback (DRIVE, SLOW DOWN, STOP) based on road conditions.

## 📦 Tech Stack

- **Backend**: FastAPI, PyTorch, Ultralytics (YOLOv8), OpenCV.
- **Frontend**: React, TailwindCSS, Vite.
- **Deployment**: Docker, Hugging Face Spaces.
