# STAGE 1: Builder
FROM python:3.9-slim AS builder

RUN apt-get update && apt-get install -y \
    cmake build-essential libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# Install to a local folder so we can copy it later
RUN pip install --user --no-cache-dir -r requirements.txt

# STAGE 2: Final Runner
FROM python:3.9-slim

# Only install the bare minimum runtime libraries (No compilers!)
RUN apt-get update && apt-get install -y \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the installed python packages from the builder stage
COPY --from=builder /root/.local /root/.local
COPY . .

# Ensure the local bin is in the path
ENV PATH=/root/.local/bin:$PATH

EXPOSE 8000
CMD ["python", "-m", "app.main_api"]