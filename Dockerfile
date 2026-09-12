# Container for the CBCT-IOS registration web app (Hugging Face Spaces / any host).
# Serves the Flask app on port 7860. Live registration (Open3D FPFH + ICP) runs on
# CPU; the TIPs segmentation button stays disabled unless GPU + weights are present.
FROM python:3.12-slim

# System libraries Open3D needs at runtime (headless).
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libgomp1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# OMP_NUM_THREADS=1 avoids the Open3D/OpenMP double-load crash we hit before.
# OUTPUT_DIR points at a writable path (the image itself may be read-only).
ENV OMP_NUM_THREADS=1 \
    OUTPUT_DIR=/tmp/outputs \
    PORT=7860 \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gunicorn

COPY . .

EXPOSE 7860
# One worker (Open3D is memory-heavy), a few threads, long timeout for big scans.
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--workers", "1", "--threads", "4", \
     "--timeout", "600", "app:app"]
