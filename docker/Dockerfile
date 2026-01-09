# STAGE 1: Builder
FROM python:3.9-slim AS builder

RUN apt-get update && apt-get install -y \
    cmake build-essential libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# STAGE 2: Final Runner
FROM python:3.9-slim

# IMPORTANT: Added ffmpeg for video decoding
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /root/.local /root/.local
COPY . .

ENV PATH=/root/.local/bin:$PATH
# Ensure python can find your 'app' and 'core' folders
ENV PYTHONPATH=/app

# Cloud Run uses the PORT env var; we use a shell form to respect it
CMD ["sh", "-c", "python -m app.main_api --host 0.0.0.0 --port ${PORT:-8000}"]