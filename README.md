💤 Fatigue Detection – End-to-End MLOps Pipeline

An end-to-end computer vision + MLOps project that detects driver fatigue in real time using facial landmark analysis.
The system tracks blinks and yawns from live video streams and is deployed as a cloud-native, serverless service.

🔍 What This Project Does

This application processes live camera frames, detects a human face, extracts facial landmarks, and analyzes eye and mouth geometry to identify signs of fatigue.

Fatigue indicators used

Eye Aspect Ratio (EAR) → Blink detection

Mouth Aspect Ratio (MAR) → Yawn detection

Processed frames are streamed back to the client with live annotations and counters.

✨ Key Highlights

📌 Facial landmark–based fatigue detection

🎥 Live MJPEG video streaming

⚡ Asynchronous FastAPI backend

🐳 Optimized multi-stage Docker builds

☁️ Serverless deployment on Google Cloud Run

🔁 Production-style image registry workflow

🧠 Tech Stack

Machine Learning / Computer Vision

Python

OpenCV

Dlib (HOG + Linear SVM face detector)

NumPy

Backend

FastAPI

Uvicorn

DevOps & Cloud

Docker (multi-stage builds)

Google Artifact Registry

Google Cloud Run

🏗️ Architecture Overview
Camera Stream
     ↓
Face Detection (Dlib)
     ↓
Facial Landmarks (68-point)
     ↓
EAR / MAR Computation
     ↓
Fatigue Events (Blink / Yawn)
     ↓
Annotated MJPEG Stream (FastAPI)

🚀 Local Development

Run the application locally for testing and development:

uvicorn app.main_api:app --reload


The browser receives a live video stream with fatigue metrics rendered in real time.

🐳 Dockerization Strategy

A multi-stage Dockerfile is used to:

Compile heavy dependencies (Dlib, OpenCV) in a build stage

Keep the runtime image lightweight and fast

Reduce cold-start latency on Cloud Run

☁️ Cloud Deployment (Google Cloud Run)
1️⃣ Authenticate with Google Cloud
gcloud auth login
gcloud auth configure-docker us-central1-docker.pkg.dev

2️⃣ Create Artifact Registry
gcloud artifacts repositories create drowsiness-repo \
  --repository-format=docker \
  --location=us-central1

3️⃣ Build and Push Docker Image
docker build -t fatigue-app-slim .

docker tag fatigue-app-slim \
  us-central1-docker.pkg.dev/[PROJECT-ID]/drowsiness-repo/fatigue-app-slim

docker push \
  us-central1-docker.pkg.dev/[PROJECT-ID]/drowsiness-repo/fatigue-app-slim

4️⃣ Deploy to Cloud Run
gcloud run deploy drowsiness-service \
  --image us-central1-docker.pkg.dev/[PROJECT-ID]/drowsiness-repo/fatigue-app-slim \
  --memory 2Gi \
  --cpu 2 \
  --timeout 300 \
  --region us-central1 \
  --platform managed \
  --allow-unauthenticated

📈 Future Improvements

This project is currently an MVP, with several planned upgrades:

🚀 Model optimization

Replace Dlib with MediaPipe or YOLOv8-Face for higher FPS

🧠 Temporal modeling

Use LSTMs / Transformers to detect prolonged eye closure

🔔 Real-world integration

Connect to IoT devices for real-time alerts in long-haul vehicles

👨‍💻 Author

Aman Vasisht