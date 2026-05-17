# Build the frontend
FROM node:20-alpine AS frontend-build

WORKDIR /app/frontend

COPY frontend/package.json frontend/package-lock.json* ./
RUN npm install

COPY frontend/ .
RUN npm run build


# Build the Python backend image
FROM python:3.12-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    YOLO_CONFIG_DIR=/tmp/Ultralytics

WORKDIR /app

# Install system dependencies required by OpenCV and Hugging Face
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libxcb1 \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user for Hugging Face Spaces (UID 1000)
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:${PATH}"

WORKDIR /home/user/app

# Copy dependency files first for caching
COPY --chown=user pyproject.toml ./

# Install project dependencies using uv (faster than pip)
RUN pip install --user uv
RUN uv pip install --user . torch torchvision

# Copy backend and other files
COPY --chown=user backend ./backend
COPY --chown=user run.py ./
COPY --chown=user .python-version ./

# Copy built frontend
COPY --chown=user --from=frontend-build /app/frontend/dist ./frontend/dist

# Expose Hugging Face Spaces default port
EXPOSE 7860

# Start backend using uvicorn
CMD ["python3", "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "7860"]
