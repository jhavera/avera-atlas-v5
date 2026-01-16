# YOLOv8 Spacecraft Detection Service (Ultralytics Backend)
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    fonts-dejavu-core \
    libglib2.0-0 \
    libgl1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
# Note: Using CPU-only torch to reduce image size
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY main.py .
COPY detector_yolo.py .

# Copy YOLOv8 model (.pt or .onnx supported)
# IMPORTANT: Place your model file in the build context
COPY spark_detector.onnx .

# Expose the service port
EXPOSE 8000

# Environment variables (can be overridden at runtime)
ENV MODEL_PATH=spark_detector.onnx
ENV CONF_THRESHOLD=0.25
ENV IOU_THRESHOLD=0.45

# Run the service
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]