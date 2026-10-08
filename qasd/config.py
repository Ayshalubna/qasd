"""Settings shared by training, export and serving."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(__file__).resolve().parent / "data"
CHECKSUMS = {
    "train.csv": "b06e26ac675513959a63135f11b94ea7786ed02da65db93a5650d8838cbc664b",
    "test.csv": "d12d6e3bc4c3103966ae786dc435913c0c563dfa328f5a3646d0e62cfeeb474d",
}
BASE_MODEL = os.getenv("QASD_BASE_MODEL", "distilbert-base-uncased")
MAX_LEN = 64
SEED = 42
RUNS = Path(os.getenv("QASD_RUNS", ROOT / "runs"))           # checkpoints, per-run metrics
EXPORT = Path(os.getenv("QASD_EXPORT", ROOT / "export"))     # ONNX model served by the API and the web demo
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", (ROOT / "mlruns").as_uri())   # local file store; any MLflow server works too
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "true")
os.environ.setdefault("DO_NOT_TRACK", "true")

# The three ways of adapting the same pretrained model that are compared
METHODS = {
    "full": {"label": "Full fine-tuning", "epochs": 4, "lr": 5e-5, "batch": 32},
    "lora": {"label": "LoRA (r=8, attention q/v)", "epochs": 6, "lr": 1e-3, "batch": 32, "r": 8, "alpha": 16, "dropout": 0.1},
    "head": {"label": "Frozen encoder, train classifier only", "epochs": 8, "lr": 2e-3, "batch": 32},
}
