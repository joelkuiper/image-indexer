FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

WORKDIR /app

# Install minimal deps
RUN pip install --no-cache-dir runpod pillow

# Copy handler
COPY handler.py .

# RunPod expects the handler to run on startup
CMD ["python", "-u", "handler.py"]
