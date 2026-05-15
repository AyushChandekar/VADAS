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

### Prerequisites

You will need the following model weights (checkpoints):
1. **YOLOv8 Checkpoint**: `yolo_idd_best.pt`
2. **U-Net Checkpoint**: `unet_drivable_best.pth`

### Environment Variables

To make the models work on Hugging Face, you can either:
- Upload them to the `backend/checkpoints/` directory in your Space.
- **(Recommended)** Set the following Environment Variables in your Space Settings:
  - `YOLO_CHECKPOINT_URL`: Direct download URL for the YOLO weights.
  - `UNET_CHECKPOINT_URL`: Direct download URL for the U-Net weights.

### Local Development

1. **Clone the repository**:
   ```bash
   git clone <repo-url>
   cd VADAS
   ```

2. **Run with Docker**:
   ```bash
   docker build -t vadas .
   docker run -p 7860:7860 vadas
   ```

3. **Manual Setup**:
   - **Backend**:
     ```bash
     pip install .
     python run.py
     ```
   - **Frontend**:
     ```bash
     cd frontend
     npm install
     npm run dev
     ```

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
