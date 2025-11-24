# ExecuTorch MLOps Haptic Mat — Quantized Person Detection Pipeline

An end-to-end **MLOps framework** for deploying **quantized person-detection models** on an **Android-based haptic mat system** using **STM32 pressure sensing** and **ExecuTorch**.

This repo extends the original ExecuTorch MobileNet demo with a production-ready pipeline:
**Python PTQ + MLflow + MinIO + Jenkins CI/CD + Android ExecuTorch inference**.

Developed in collaboration with **Seroton GmbH**.

---

## Project Title
**MLOps Framework for Quantized Person Detection Models in Android-Based Haptic Mat Systems with STM32 Pressure Sensing**

## Abstract

This project builds an automated MLOps system for delivering and evaluating **on-device AI models** used in a haptic mat capable of detecting if a user is present, sitting, fully lying, or partially lying.

Pressure data from STM32 bladders is processed by a **quantized person-detection model**, enabling:

* private, offline inference
* reduced operational cost
* automatic app shutdown when no user is detected
* real-time responsiveness on mobile devices

Using **post-training quantization (PTQ)**, the system compresses models, evaluates them via MLflow, stores versions in MinIO, deploys them through Jenkins, and loads them inside an Android ExecuTorch app.

**References:**
* https://advanced.onlinelibrary.wiley.com/doi/10.1002/advs.202402461
* https://arxiv.org/abs/1712.05877
* https://github.com/google/XNNPACK
* https://www.vulkan.org/

---

## System Architecture

This diagram illustrates the complete end-to-end MLOps pipeline, from the initial Python script to the final on-device inference on Android.

```mermaid
flowchart LR

    %% ========== COMPONENTS ==========
    Script["Python Script<br/>log_model_to_mlflow.py"]
    MLflow["MLflow Tracking Server"]
    MySQL["MySQL Backend Store"]
    Minio["MinIO Object Storage"]
    Jenkins["Jenkins CI/CD Pipeline"]
    Android["Android App<br/>ExecuTorch Runtime"]

    %% ========== PIPELINES ==========

    %% Script → MLflow
    Script -->|log_param / log_metric / log_artifact| MLflow

    %% MLflow → Backend Stores
    MLflow -->|runs, params, metrics| MySQL
    MLflow -->|artifacts<br/>(Staging Bucket)| Minio

    %% Jenkins → MLflow → MinIO
    Jenkins -->|Search latest successful run| MLflow
    Jenkins -->|Download artifacts<br/>(from Staging)| Minio
    Jenkins -->|SHA-256 Verification| Jenkins
    Jenkins -->|Upload production model<br/>(to Production Bucket)| Minio

    %% Android App → MinIO
    Android -->|GET latest.json| Minio
    Android -->|Download model.pte<br/>SHA256 verify| Minio
    Android -->|On-device inference<br/>ExecuTorch XNNPACK| Android

    %% STYLES
    classDef server fill:#e7f0ff,stroke:#4a90e2,stroke-width:1px;
    classDef storage fill:#fff7e6,stroke:#e6a500,stroke-width:1px;
    classDef mobile fill:#e6fff2,stroke:#00a86b,stroke-width:1px;
    classDef script fill:#f6f8fa,stroke:#999,stroke-width:1px;
    classDef jenkins fill:#ffecec,stroke:#ff5757,stroke-width:1px;

    class Script script;
    class MLflow server;
    class Jenkins jenkins;
    class MySQL,Minio storage;
    class Android mobile;

---

## Tech Stack

* **MLOps Core:** MLflow, MinIO, Jenkins, MySQL, Docker
* **AI/ML Frameworks:** PyTorch, ExecuTorch, XNNPACK (CPU backend), Python
* **Mobile Development:** Kotlin, Jetpack Compose, Android SDK
* **Hardware:** STM32 Pressure Sensors

---

## 🚀 MLOps Workflow & Proof of Concept

This section demonstrates the actual execution of the pipeline with screenshots from our live environment.

### Step 1: Training & Experiment Tracking (MLflow)

The process begins with the `log_model_to_mlflow.py` script. It loads a pre-trained quantized model (`.pte`), calculates its SHA-256 hash for security, and logs all metadata, metrics (e.g., latency, size), and artifacts to the MLflow Tracking Server.

<div align="center">
  <img src="(GÖRSEL EKLE: Buraya MLflow görselinin yolunu yaz)" alt="MLflow Experiment Tracking" width="800">
  <p><i>Figure 1: MLflow dashboard showing a successful experiment run with logged metrics and artifacts.</i></p>
</div>

### Step 2: Artifact Storage (MinIO - Staging)

MLflow automatically stores the actual model files and manifests in our S3-compatible MinIO storage. This first bucket (`mlflow-artifacts`) acts as our **Staging Area**.

<div align="center">
  <img src="(GÖRSEL EKLE: Buraya MinIO Staging görselinin yolunu yaz)" alt="MinIO Staging Bucket" width="800">
  <p><i>Figure 2: MinIO Staging Bucket containing the raw model artifacts pushed by MLflow.</i></p>
</div>

### Step 3: CI/CD Deployment & Verification (Jenkins)

Jenkins is triggered to deploy the latest successful model. It fetches the artifacts from the Staging bucket, performs a critical **SHA-256 verification** to ensure integrity, and if successful, promotes the model to production.

<div align="center">
  <img src="(GÖRSEL EKLE: Buraya Jenkins görselinin yolunu yaz)" alt="Jenkins CI/CD Pipeline" width="800">
  <p><i>Figure 3: Jenkins CI/CD pipeline executing the model promotion and verification process.</i></p>
</div>

### Step 4: Production Storage (MinIO - Production)

The verified model and a newly generated `latest.json` manifest are uploaded by Jenkins to the **Production Bucket** (`mlops-test`). This bucket is publicly accessible (read-only) for the Android app.

<div align="center">
  <img src="(GÖRSEL EKLE: Buraya MinIO Production görselinin yolunu yaz)" alt="MinIO Production Bucket" width="800">
  <p><i>Figure 4: MinIO Production Bucket containing the deployment-ready model and manifest.</i></p>
</div>

---

## Android App Integration & Demo

The Android application, built with Kotlin and Jetpack Compose, uses the **ExecuTorch runtime** with the **XNNPACK backend** for accelerated CPU inference. It periodically checks the MinIO Production Bucket for updates via `latest.json` and downloads the new model automatically.

### Building the ExecuTorch Model
The `.pte` model file is generated using the following script, which exports a PyTorch MobileNetV2 to ExecuTorch format with XNNPACK optimization:

```bash
python mv2_xnnpack_build.py
### App Demo & Results
The app demonstrates successful loading and inference of the quantized model.

<div align="center">
  <table>
    <tr>
      <td align="center">
        <b>📱 Screenshot</b><br/>
        <img src="(GÖRSEL EKLE: screen_test.jpeg görselinin yolunu yaz)" alt="App Test Screenshot" width="300">
      </td>
      <td align="center">
        <b>🎥 Demo Video</b><br/>
        <img src="(GÖRSEL EKLE: demo.gif görselinin yolunu yaz)" alt="App Demo Video" width="300">
      </td>
    </tr>
  </table>
</div>

**Testing Results:**
* **Model Loading:** ✅ Successful from assets/MinIO
* **Inference Speed:** ⚡ **25ms** average execution time on device
* **Output Validation:** 📊 Correct [1, 1000] shape for ImageNet classification
* **Performance:** 🚀 XNNPACK acceleration working optimally

---

## Performance Metrics & Key Features

### Key Features
* **100% On-Device Inference:** Ensures complete privacy and offline capability.
* **Real-time Quantized AI:** Optimized for mobile CPUs using PTQ and XNNPACK.
* **Automated CI/CD Pipeline:** Streamlined deployment with Jenkins and SHA-256 security.
* **Reproducible Setup:** Entire backend infrastructure is Dockerized.

### Performance Metrics
* **Model Size:** ~9MB (Quantized MobileNetV2)
* **Inference Time:** 25ms (tested on Samsung device)
* **CPU Usage:** Optimized with XNNPACK backend

---

## How to Run the MLOps Pipeline

Follow these steps to stand up the entire infrastructure and deploy a model.

**1️⃣ Start the MLOps Infrastructure**
```bash
docker compose up -d
This commands starts MinIO, MLflow, MySQL, and Jenkins containers.

**2️⃣ Log a Model to MLflow & MinIO (Staging)**
```bash
python scripts/log_model_to_mlflow.py
This script processes the model and logs it to the MLflow tracking server.

**3️⃣ Run Jenkins Pipeline (Deployment)
Access Jenkins at http://localhost:8080.

Trigger the defined pipeline job.

Jenkins will automatically:

Download artifacts from Staging.

Validate SHA-256.

Upload the model and latest.json to the Production MinIO bucket.

**4️⃣ Run the Android App
Build and run the app in Android Studio.

Tap "Load Model".

The app will check MinIO, download the model if needed, and be ready for inference.
