from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import requests

# Request dikirim ke exporter proxy, bukan langsung ke MLflow.
# Dengan begitu, request, prediction, error, dan latency benar-benar tercatat di Prometheus.
EXPORTER_INFERENCE_URL = "http://127.0.0.1:8000/invocations"
DATA_PATH = Path(__file__).resolve().parents[1] / "Membangun_model" / "breast_cancer_preprocessing" / "test.csv"


def main() -> None:
    df = pd.read_csv(DATA_PATH).drop(columns=["target"]).head(3)
    payload = {"dataframe_split": df.to_dict(orient="split")}

    response = requests.post(
        EXPORTER_INFERENCE_URL,
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload),
        timeout=30,
    )

    print("Status code:", response.status_code)
    print(response.text)


if __name__ == "__main__":
    main()
