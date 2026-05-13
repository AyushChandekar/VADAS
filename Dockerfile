# Build the frontend
FROM node:20-alpine AS frontend-build

WORKDIR /app/frontend

COPY frontend/package.json frontend/package-lock.json* ./

RUN npm install

COPY frontend/ .

RUN npm run build


# Build the Python backend image
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app


# Install system dependencies required by OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libxcb1 \
    && rm -rf /var/lib/apt/lists/*


# Copy dependency files
COPY pyproject.toml ./
COPY uv.lock ./


# Upgrade pip
RUN python -m pip install --upgrade pip


# Install torch CPU version
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu torch torchvision


# Install project dependencies
RUN pip install --no-cache-dir .


# Copy backend
COPY backend ./backend

# Copy run file
COPY run.py ./


# Copy built frontend
COPY --from=frontend-build /app/frontend/dist ./frontend/dist


# Expose Hugging Face Spaces port
EXPOSE 7860


# Start backend
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "7860"]