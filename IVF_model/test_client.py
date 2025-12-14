"""
Simple test client to exercise the Flask server endpoints.

Usage:
1. Start server: `python IVF_model/app.py`
2. Run this client: `python IVF_model/test_client.py`

It will call /health, then /predict_batch (if dataset exists), and show outputs.
"""
import json
import os
import time
import requests

import os

# read server port from env to match app; default to 8000
PORT = os.environ.get("PORT", "8000")
BASE = f"http://127.0.0.1:{PORT}"


def call_health():
    r = requests.get(f"{BASE}/health")
    print("/health ->", r.status_code, r.text)


def call_predict_batch():
    print("Calling /ivf/run_batch ...")
    r = requests.post(f"{BASE}/ivf/run_batch")
    print("/ivf/run_batch ->", r.status_code)
    print(r.text[:1000])


def call_predict_sample(sample):
    print("Calling /predict for single sample ...")
    # send to the HTML endpoint (it accepts form or JSON)
    try:
        r = requests.post(f"{BASE}/ivf/predict", json={"features": sample})
    except Exception:
        # fallback to form-encoded
        r = requests.post(f"{BASE}/ivf/predict", data=sample)
    print("/predict ->", r.status_code)
    try:
        print(json.dumps(r.json(), indent=2))
    except Exception as e:
        print("Failed to parse JSON response:", e)


def load_first_sample_from_test():
    path = os.path.join(os.path.dirname(__file__), "ivf_test_dataset.csv")
    if not os.path.exists(path):
        print("No test dataset found at", path)
        return None
    import pandas as pd

    df = pd.read_csv(path)
    # drop excluded columns and pick first row as dict
    exclude = ["patient_id", "embryo_health_score", "embryo_quality_class"]
    row = df.drop(columns=[c for c in exclude if c in df.columns]).iloc[0]
    # replace NaN with None for JSON serialization
    row = row.where(pd.notnull(row), None)
    sample = row.to_dict()
    return sample


if __name__ == "__main__":
    call_health()
    sample = load_first_sample_from_test()
    if sample is not None:
        call_predict_sample(sample)
    call_predict_batch()
