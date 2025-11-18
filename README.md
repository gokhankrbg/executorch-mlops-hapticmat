# ExecuTorch MLOps Haptic Mat — Quantized Person Detection Pipeline

An end-to-end **MLOps framework** for deploying **quantized person-detection models** on an **Android-based haptic mat system** using **STM32 pressure sensing** and **ExecuTorch**.

This repo extends the original ExecuTorch MobileNet demo with a production-ready pipeline:  
**Python PTQ + MLflow + MinIO + Jenkins CI/CD + Android ExecuTorch inference**.

---

## 🧩 Project Title  
**MLOps Framework for Quantized Person Detection Models in Android-Based Haptic Mat Systems with STM32 Pressure Sensing**
```mermaid
flowchart LR
    subgraph DevMachine[Docker Host (Local)]
        subgraph Stack[Docker Compose Stack]
            MLflow[MLflow Tracking Server\n(http://localhost:5001)]
            Minio[(MinIO Object Store\n:9000 / :9001)]
            MySQL[(MySQL DB\nMLflow Backend Store)]
            Jenkins[Jenkins CI/CD\nMLOps-Model-Deployment Pipeline]
        end
    end

    Script[log_model_to_mlflow.py\n(PTQ + logging script)]:::code
    Android[Android App\nExecuTorch + Quantized Model]:::mobile

    %% Script -> MLflow
    Script -->|MLflow Tracking API\nlog_param / log_metric / log_artifact| MLflow

    %% MLflow -> MySQL & MinIO
    MLflow -->|runs, params,\nmetrics, experiments| MySQL
    MLflow -->|"artifact_uri" altında\nmodel_files/manifest.json\nmv2_xnnpack.pte| Minio

    %% Jenkins pipeline
    Jenkins -->|mlflow search-runs\n(en son başarılı run ID)| MLflow
    Jenkins -->|mlflow artifacts download\nmodel_files/...| Minio
    Jenkins -->|Upload to prod bucket\nmv2_xnnpack.pte + latest.json| Minio

    %% Android app
    Android -->|HTTP(S)\nGET latest.json + model.pte| Minio
    Android -->|On-device inference\nExecuTorch XNNPACK| Android

    classDef code fill:#f6f8fa,stroke:#999,stroke-width:1px;
    classDef mobile fill:#e8f7ff,stroke:#0077b6,stroke-width:1px;
```

---

## 📄 Abstract (Short)

This project builds an automated MLOps system for delivering and evaluating **on-device AI models** used in a haptic mat capable of detecting if a user is present, sitting, fully lying, or partially lying.

Pressure data from STM32 bladders is processed by a **quantized person-detection model**, enabling:

- private, offline inference  
- reduced operational cost  
- automatic app shutdown when no user is detected  
- real-time responsiveness on mobile devices  

Using **post-training quantization (PTQ)**, the system compresses models, evaluates them via MLflow, stores versions in MinIO, deploys them through Jenkins, and loads them inside an Android ExecuTorch app.

Developed with **Seroton GmbH**.

**References:**  
https://advanced.onlinelibrary.wiley.com/doi/10.1002/advs.202402461  
https://arxiv.org/abs/1712.05877  
https://github.com/google/XNNPACK  
https://www.vulkan.org/

---

## 🏗 System Architecture

Python PTQ Script → MLflow → MinIO → Jenkins CI/CD → Android ExecuTorch App
- **MLflow**: logs metrics, artifacts, manifests  
- **MinIO**: S3 storage for models  
- **Jenkins**: fetches latest MLflow run, validates SHA, deploys model  
- **ExecuTorch**: loads model directly from MinIO inside Android app  

---

## 🛠 Tech Stack

- **MLflow**, **MinIO**, **Jenkins**, **Docker**, **Python**
- **ExecuTorch**, **XNNPACK** (CPU backend)
- **Kotlin**, **Jetpack Compose**
- **STM32 Pressure Sensors**

## 📱 App Demo

<div align="center">
  <table>
    <tr>
      <td align="center">
        <b>📱 Screenshot</b><br/>
        <img src="screen_test.jpeg" alt="App Test Screenshot" width="300">
      </td>
      <td align="center">
        <b>🎥 Demo Video</b><br/>
        <img src="demo.gif" alt="App Demo Video" width="300">
      </td>
    </tr>
  </table>
</div>

The app successfully loads and tests the MobileNetV2 model with:
- ✅ Model loading from assets
- ⚡ Fast inference (25ms on device)  
- 📊 Correct output shape validation [1, 1000]
- 🏆 Top-5 prediction display
- 📈 Output statistics analysis

## 🏗️ Building the Model

The ExecuTorch model file (`mv2_xnnpack.pte`) is generated using:

```bash
python mv2_xnnpack_build.py
```

This script exports a MobileNetV2 model from PyTorch to ExecuTorch format with XNNPACK backend optimization for accelerated CPU inference.

## 🚀 Features

- **Model Loading**: Loads pre-trained MobileNetV2 model from Android assets
- **Fast Inference**: Optimized with XNNPACK backend for mobile performance
- **Real-time Testing**: Test inference with random input data (1×3×224×224)
- **Results Analysis**: 
  - Inference timing measurement
  - Output shape validation
  - Top-5 class predictions
  - Statistical analysis (min/max/mean values)
- **User-friendly UI**: Material Design 3 interface with status indicators

## 🧪 Testing Results

The app demonstrates successful ExecuTorch integration:

- **Model Loading**: ✅ Successful from assets folder
- **Inference Speed**: ⚡ 25ms average execution time
- **Output Validation**: 📊 Correct [1, 1000] shape for ImageNet classification
- **Performance**: 🚀 XNNPACK acceleration working optimally
- **Stability**: 💪 No crashes or memory issues

## 📊 Performance Metrics

- **Model Size**: ~9MB (MobileNetV2 with XNNPACK optimization)
- **Inference Time**: 25ms (tested on Samsung device)
- **Memory Usage**: Minimal overhead
- **CPU Usage**: Optimized with XNNPACK backend

---

## 🔥 Training → Quantization → Deployment Pipeline

### **1️⃣ Start MLOps stack**

```bash
docker compose up -d
2️⃣ Log model to MLflow + MinIO
bash
Kodu kopyala
python scripts/log_model_to_mlflow.py
3️⃣ Jenkins auto-deploys the latest model
Downloads MLflow artifacts

Validates SHA-256

Uploads production model to MinIO

Publishes latest.json for the Android app

4️⃣ Android app loads model from MinIO
User taps Load Model →
model is downloaded → SHA check → ExecuTorch loads → inference begins.

🚀 Key Features
🔒 100% on-device inference for privacy

⚡ Real-time quantized AI on mobile

🔁 Automated CI/CD pipeline with Jenkins

📦 Reproducible Dockerized setup

🧪 Full PTQ + benchmarking pipeline

🏢 Collaboration
Developed in collaboration with Seroton GmbH.
