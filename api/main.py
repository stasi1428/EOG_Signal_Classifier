#!/usr/bin/env python3
"""
FastAPI app for serving the EOG movement classifier.
"""

import joblib
import numpy as np
from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel

# Paths
BASE_DIR   = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "eog_movement_model.joblib"

# Load trained model
model = joblib.load(MODEL_PATH)

# FastAPI app
app = FastAPI(title="EOG Movement Classification API")

# Request schema
class SignalRequest(BaseModel):
    signal: list[float]  # raw EOG signal (length ~251)

# Preprocessing (same as pipeline)
def preprocess_signal(sig):
    sig = np.array(sig)
    return (sig - np.mean(sig)) / (np.std(sig) + 1e-8)

# Feature extraction (same as pipeline)
def extract_features(sig):
    return np.array([
        np.mean(sig),
        np.std(sig),
        np.max(sig) - np.min(sig)
    ]).reshape(1, -1)

# Routes
@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": MODEL_PATH.name}

@app.post("/predict")
def predict(req: SignalRequest):
    sig_proc = preprocess_signal(req.signal)
    feats = extract_features(sig_proc)
    pred = model.predict(feats)[0]
    return {"prediction": pred}