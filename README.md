# Fatigue-detection-mlops
An End-to-End MLOps Pipeline: From Local Development to Cloud-Native Deployment

🚀 Project Overview
This project features a robust computer vision pipeline designed to monitor driver safety in real-time. By leveraging Dlib's pre-trained HOG + Linear SVM face detector and 68-point facial landmark predictor, the system analyzes facial geometry to detect signs of fatigue, specifically blinking patterns and yawning frequency.

Key Features:
Facial Landmark Analysis: Uses Eye Aspect Ratio (EAR) and Mouth Aspect Ratio (MAR) for precise detection.

Real-Time Analytics: Annotates frames with live counters for blinks and yawns.

Cloud-Native Architecture: Fully containerized and deployed on Google Cloud Run for global accessibility.

Asynchronous Processing: Built with FastAPI to handle high-concurrency video streaming.

🛠️ Tech Stack
ML/CV: Python, OpenCV, Dlib, NumPy.

API Framework: FastAPI, Uvicorn.

DevOps: Docker (Multi-stage builds), Google Artifact Registry.

Cloud: Google Cloud Run (Serverless).

🏗️ Development & Deployment Lifecycle
1. Local Development & Testing
The core logic was first developed and validated locally. We used FastAPI to create a StreamingResponse that allows the browser to render ML-processed frames as a live MJPEG stream.

Bash

# Running the app locally
uvicorn app.main_api:app --reload
2. Optimized Containerization (Multi-Stage Build)
To ensure high performance and low latency on the cloud, a Multi-Stage Dockerfile was implemented. This reduced the final image size by separating the build environment (compilers/headers) from the runtime environment.

3. Google Cloud Deployment (CI/CD Flow)
The deployment followed a professional production workflow:

A. Environment Setup
Install the Google Cloud SDK and authenticate:

Bash

gcloud auth login
gcloud auth configure-docker us-central1-docker.pkg.dev
B. Artifact Management
Create a private repository to host the production-ready Docker images:

Bash

gcloud artifacts repositories create drowsiness-repo \
    --repository-format=docker \
    --location=us-central1
C. Build, Tag, & Push
Bash

# Build the image
docker build -t fatigue-app-slim .

# Tag for Google Artifact Registry
docker tag fatigue-app-slim us-central1-docker.pkg.dev/[PROJECT-ID]/drowsiness-repo/fatigue-app-slim

# Push to the cloud
docker push us-central1-docker.pkg.dev/[PROJECT-ID]/drowsiness-repo/fatigue-app-slim
D. Serverless Deployment
Deploying to Cloud Run with allocated resources for ML inference:

Bash

gcloud run deploy drowsiness-service \
  --image us-central1-docker.pkg.dev/[PROJECT-ID]/drowsiness-repo/fatigue-app-slim \
  --memory 2Gi --cpu 2 --timeout 300 --platform managed --region us-central1 --allow-unauthenticated
📈 Future Roadmap
While this project serves as a functional MVP (Minimum Viable Product), future iterations will focus on:

Model Optimization: Migrating from Dlib to MediaPipe or YOLOv8-Face for higher FPS on mobile devices.

Temporal Analysis: Implementing LSTMs or Transformers to analyze the duration of eye closure rather than just frequency.

Road Safety Integration: Extending the use case to IoT devices for long-haul truck drivers to trigger real-time auditory alerts.

👨‍💻 Author
Aman Vasisht
