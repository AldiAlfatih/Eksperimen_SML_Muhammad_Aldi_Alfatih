from __future__ import annotations

import json
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import psutil
import requests
from prometheus_client import Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST

# Endpoint MLflow Model Serving yang benar-benar menjalankan model.
# Jalankan MLflow serving di port 5002, lalu jalankan exporter ini di port 8000.
MLFLOW_SERVING_URL = os.getenv("MLFLOW_SERVING_URL", "http://127.0.0.1:5002/invocations")
EXPORTER_HOST = os.getenv("EXPORTER_HOST", "0.0.0.0")
EXPORTER_PORT = int(os.getenv("EXPORTER_PORT", "8000"))
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "30"))

# Metrics utama monitoring inference asli.
REQUEST_COUNT = Counter("model_request_total", "Total inference requests received by the serving proxy")
PREDICTION_COUNT = Counter("model_prediction_total", "Total predictions returned by the real MLflow model")
ERROR_COUNT = Counter("model_error_total", "Total inference errors from the real serving process")
LATENCY = Histogram("model_latency_seconds", "Actual end-to-end model inference latency in seconds")

# Metrics resource ketika model melayani request.
CPU_USAGE = Gauge("system_cpu_usage_percent", "CPU usage percentage")
RAM_USAGE = Gauge("system_ram_usage_percent", "RAM usage percentage")
DISK_USAGE = Gauge("system_disk_usage_percent", "Disk usage percentage")
MODEL_SERVING_UP = Gauge("model_serving_up", "1 if the MLflow serving endpoint is reachable, otherwise 0")
MODEL_LAST_STATUS_CODE = Gauge("model_last_response_status_code", "Last HTTP status code returned by MLflow serving")

# Metrics performa training terakhir, dibaca dari file MLflow bila tersedia.
MODEL_ACCURACY = Gauge("model_accuracy", "Last training model accuracy from MLflow run metrics")
MODEL_F1 = Gauge("model_f1_score", "Last training model F1 score from MLflow run metrics")
MODEL_ROC_AUC = Gauge("model_roc_auc", "Last training model ROC AUC from MLflow run metrics")


def _read_mlflow_metric(metric_name: str) -> float | None:
    """Read the latest metric value from local mlruns files."""
    project_root = Path(__file__).resolve().parents[1]
    mlruns_dir = project_root / "Membangun_model" / "mlruns"
    candidates = list(mlruns_dir.glob(f"*/**/metrics/{metric_name}"))
    if not candidates:
        return None

    latest_file = max(candidates, key=lambda p: p.stat().st_mtime)
    try:
        lines = [line.strip() for line in latest_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not lines:
            return None
        # Format metric MLflow: timestamp value step
        return float(lines[-1].split()[1])
    except Exception:
        return None


def load_training_metrics() -> None:
    metric_map = {
        "test_accuracy": MODEL_ACCURACY,
        "test_f1_score": MODEL_F1,
        "test_roc_auc": MODEL_ROC_AUC,
    }
    for metric_name, gauge in metric_map.items():
        value = _read_mlflow_metric(metric_name)
        if value is not None:
            gauge.set(value)


def update_resource_metrics() -> None:
    CPU_USAGE.set(psutil.cpu_percent(interval=0.1))
    RAM_USAGE.set(psutil.virtual_memory().percent)
    DISK_USAGE.set(psutil.disk_usage("/").percent)


def count_predictions(response_body: bytes) -> int:
    """Count predictions from MLflow JSON response."""
    try:
        payload: Any = json.loads(response_body.decode("utf-8"))
    except Exception:
        return 0

    predictions = payload.get("predictions") if isinstance(payload, dict) else payload
    if isinstance(predictions, list):
        return len(predictions)
    if predictions is None:
        return 0
    return 1


class MetricsHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        # Keep terminal output clean.
        return

    def do_GET(self) -> None:
        if self.path == "/metrics":
            update_resource_metrics()
            load_training_metrics()
            output = generate_latest()
            self.send_response(200)
            self.send_header("Content-Type", CONTENT_TYPE_LATEST)
            self.end_headers()
            self.wfile.write(output)
            return

        if self.path == "/health":
            try:
                requests.get(MLFLOW_SERVING_URL.replace("/invocations", "/ping"), timeout=5)
                MODEL_SERVING_UP.set(1)
                status = b"exporter_ok; mlflow_serving_reachable"
            except requests.RequestException:
                MODEL_SERVING_UP.set(0)
                status = b"exporter_ok; mlflow_serving_unreachable"
            self.send_response(200)
            self.end_headers()
            self.wfile.write(status)
            return

        self.send_response(404)
        self.end_headers()
        self.wfile.write(b"Use GET /metrics or POST /invocations")

    def do_POST(self) -> None:
        if self.path != "/invocations":
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Use POST /invocations")
            return

        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)
        content_type = self.headers.get("Content-Type", "application/json")

        REQUEST_COUNT.inc()
        update_resource_metrics()
        start_time = time.perf_counter()

        try:
            response = requests.post(
                MLFLOW_SERVING_URL,
                data=body,
                headers={"Content-Type": content_type},
                timeout=REQUEST_TIMEOUT,
            )
            elapsed = time.perf_counter() - start_time
            LATENCY.observe(elapsed)
            MODEL_LAST_STATUS_CODE.set(response.status_code)
            MODEL_SERVING_UP.set(1)

            if response.ok:
                prediction_total = count_predictions(response.content)
                if prediction_total > 0:
                    PREDICTION_COUNT.inc(prediction_total)
            else:
                ERROR_COUNT.inc()

            self.send_response(response.status_code)
            self.send_header("Content-Type", response.headers.get("Content-Type", "application/json"))
            self.end_headers()
            self.wfile.write(response.content)

        except requests.RequestException as exc:
            elapsed = time.perf_counter() - start_time
            LATENCY.observe(elapsed)
            ERROR_COUNT.inc()
            MODEL_SERVING_UP.set(0)
            self.send_response(502)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(exc)}).encode("utf-8"))


if __name__ == "__main__":
    load_training_metrics()
    server = ThreadingHTTPServer((EXPORTER_HOST, EXPORTER_PORT), MetricsHandler)
    print(f"Prometheus exporter proxy running at http://127.0.0.1:{EXPORTER_PORT}/metrics")
    print(f"Forwarding real inference requests to {MLFLOW_SERVING_URL}")
    server.serve_forever()
