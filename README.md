# 🎨 Real-Time Driver Drowsiness Detection: Cloud-Native MLOps

![Python](https://img.shields.io/badge/Python-3.9-blue) ![GCP](https://img.shields.io/badge/GCP-Cloud_Run%20|%20Artifact_Registry-orange) ![FastAPI](https://img.shields.io/badge/FastAPI-v0.100+-green) ![Docker](https://img.shields.io/badge/Docker-Multi--Stage-blue) ![OpenCV](https://img.shields.io/badge/OpenCV-Enabled-lightgrey)

An end-to-end MLOps solution for driver safety. This project implements a real-time computer vision pipeline using Dlib for facial landmark detection, containerized via Docker, and deployed as a serverless microservice on Google Cloud Platform.

<p align="center">
  <video src="https://github.com/user-attachments/assets/95ff147b-6f91-4a2c-bb1e-ec09559749aa" width="100%" autoplay loop muted playsinline></video>
</p>

## 📋 Table of Contents
- [Project Architecture](#-project-architecture)
- [Key Features](#-key-features)
- [Prerequisites](#-prerequisites)
- [Local Environment Setup](#-local-environment-setup)
- [ML Inference Logic](#-ml-inference-logic)
- [Cloud Deployment (GCP)](#-cloud-deployment-gcp)
- [Future Roadmap](#-future-roadmap)

---

## 🏗 Project Architecture

This project follows a professional modular structure for production-ready deployment:
1.  **Inference Layer:** Dlib HOG + Linear SVM Face Detector.
2.  **API Layer:** FastAPI with StreamingResponse (MJPEG over HTTP).
3.  **Containerization:** Multi-stage Docker build for minimal image footprint.
4.  **Registry:** Google Artifact Registry (GAR).
5.  **Compute:** Google Cloud Run (Serverless CPU/RAM allocation).

---

## 🌟 Key Features

* **Advanced Facial Analysis:** Utilizes Eye Aspect Ratio (EAR) and Mouth Aspect Ratio (MAR) for precise drowsiness and yawning detection.
* **Asynchronous Streaming:** Implements FastAPI asynchronous generators to stream processed frames with low latency.
* **Environment Agnostic:** Configured via environment variables to switch seamlessly between local webcam and cloud video processing.
* **Resource Optimized:** Multi-stage Docker builds reduce image size by over 60%, ensuring faster cold starts in the cloud.

---

## 🛠 Prerequisites

* Python 3.9+
* Google Cloud SDK (gcloud CLI)
* Docker Desktop
* Dlib (Pre-trained `shape_predictor_68_face_landmarks.dat`)

---

## 💻 Local Environment Setup

1.  **Clone the Repository:**
    ```bash
    git clone <your-repo-url>
    cd fatigue-detection-mlops
    ```

2.  **Create Virtual Environment:**
    ```bash
    conda create -n drowsiness python=3.9 -y
    conda activate drowsiness
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    pip install python-multipart
    ```

4.  **Run Locally:**
    ```bash
    uvicorn app.main_api:app --reload
    ```

---

## ☁️ Cloud Deployment (GCP)



1.  **GCP Authentication:**
    ```bash
    gcloud auth login
    gcloud auth configure-docker us-central1-docker.pkg.dev
    ```

2.  **Build and Push to Artifact Registry:**
    ```bash
    # Build
    docker build -t fatigue-app-slim .

    # Tag (Replace [PROJECT-ID] with your GCP Project ID)
    docker tag fatigue-app-slim us-central1-docker.pkg.dev/[PROJECT-ID]/drowsiness-repo/fatigue-app-slim

    # Push
    docker push us-central1-docker.pkg.dev/[PROJECT-ID]/drowsiness-repo/fatigue-app-slim
    ```

3.  **Deploy to Cloud Run:**
    ```bash
    gcloud run deploy drowsiness-service \
      --image us-central1-docker.pkg.dev/[PROJECT-ID]/drowsiness-repo/fatigue-app-slim \
      --memory 2Gi --cpu 2 --timeout 300 --platform managed --region us-central1 --allow-unauthenticated
    ```

---

## 📈 Future Roadmap

* **Precision Upgrades:** Transitioning to MediaPipe for enhanced iris tracking.
* **Temporal Logic:** Integrating LSTMs to distinguish between natural blinking and fatigue.
* **Edge Deployment:** Optimization for Raspberry Pi and NVIDIA Jetson Nano.

---

## 👨‍💻 Author
**Aman Vasisht**
