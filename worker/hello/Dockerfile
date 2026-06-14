# Minimal RunPod worker — just proves the plumbing works
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

WORKDIR /app

# Install runpod SDK + Pillow (already have torch from base image)
RUN pip install --no-cache-dir runpod pillow

# Copy handler
COPY worker/hello/handler.py .

CMD ["python", "-u", "handler.py"]
