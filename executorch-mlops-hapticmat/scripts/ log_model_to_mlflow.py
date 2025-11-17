import mlflow
import os
import hashlib
import json

# --- Config from environment ---

os.environ["MLFLOW_S3_ENDPOINT_URL"] = os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000")
os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID", "CHANGE_ME_MINIO_ACCESS_KEY")
os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY", "CHANGE_ME_MINIO_SECRET_KEY")

EXPERIMENT_NAME = "MobileNetV2-HapticMat-Quantized"
MODEL_NAME = "PersonDetectionQuantized"
MODEL_FILE = "app/src/main/assets/mv2_xnnpack.pte"
MANIFEST_FILE = "manifest.json"


# Helper functions ---
def calculate_sha256(file_path):
    """Calculates the SHA256 digest of the given file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

# --- MLflow Flow---
if __name__ == "__main__":
    # Check if the file exists
    if not os.path.exists(MODEL_FILE):
        print(f"HATA: Model dosyası bulunamadı: {MODEL_FILE}")
        exit(1)

    # Connect to MLflow Tracking Server
    mlflow.set_tracking_uri("http://localhost:5001")
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Start a new Run
    with mlflow.start_run(run_name="quantized_model_v1") as run:
        run_id = run.info.run_id
        print(f"--- MLflow Run ID: {run_id} ---")

        # 1. Saving Parameters (Model Development Details)
        mlflow.log_param("quantization_type", "PTQ_Int8")
        mlflow.log_param("base_model", "MobileNetV2")
        mlflow.log_param("executorch_backend", "XNNPACK")

        # 2. Recording Metrics (Simulated)
        mlflow.log_metric("top1_accuracy", 0.85)
        mlflow.log_metric("inference_latency_ms", 25.0)
        mlflow.log_metric("model_size_mib", os.path.getsize(MODEL_FILE) / (1024 * 1024))

        # 3. Saving Artifacts (Model File and Manifest)

        # Create manifest file
        sha256_hash = calculate_sha256(MODEL_FILE)

        manifest_data = {
            "model_name": MODEL_NAME,
            "version": "1.0.0",
            "sha256": sha256_hash,
            "model_file": MODEL_FILE,
            "metrics": {
                "top1_accuracy": 0.85,
                "latency_ms": 25.0
            }
        }

        # Create a temporary manifest file
        with open(MANIFEST_FILE, "w") as f:
            json.dump(manifest_data, f, indent=4)

        # Upload artifacts to MLflow (i.e. MinIO)
        mlflow.log_artifact(MODEL_FILE, "model_files")
        mlflow.log_artifact(MANIFEST_FILE, "model_files")

        print(f"Model and Manifest files saved to MinIO.")

        # 4. Saving the Model to MLflow Registry (Commented)
        # This section has been commented to bypass the Model Registry API error.
        # print(f"Model '{MODEL_NAME}' has been saved to MLflow Registry.")