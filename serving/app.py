"""
FastAPI service for the ONNX ensemble.

Run:  MODEL_DIR=models/ensemble uvicorn serving.app:app --port 8080
Endpoints: GET /health, GET /info, POST /predict (multipart field ``file``: a JPEG or PNG image).
Research demo only: not a medical device and not clinically validated.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile

from serving.model import OnnxEnsemble, load_onnx_ensemble
from serving.preprocess import CLASS_NAMES

DISCLAIMER = "Research demo. Not a medical device and not clinically validated."
MAX_BYTES = 10 * 1024 * 1024


def decode_image(data: bytes) -> np.ndarray:
    arr = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if arr is None:
        raise ValueError("could not decode image")
    return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)


def create_app(model: OnnxEnsemble | None = None, model_dir: str | Path | None = None) -> FastAPI:
    app = FastAPI(title="Skin lesion classifier (research demo)", version="2.1.0")
    state: dict[str, OnnxEnsemble | None] = {"model": model}

    def get_model() -> OnnxEnsemble:
        if state["model"] is None:
            directory = model_dir or os.environ.get("MODEL_DIR", "models/ensemble")
            threads = os.environ.get("ORT_THREADS")
            state["model"] = load_onnx_ensemble(directory, int(threads) if threads else None)
        return state["model"]

    @app.get("/health")
    def health() -> dict:
        try:
            get_model()
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return {"status": "ok"}

    @app.get("/info")
    def info() -> dict:
        m = get_model()
        return {"classes": list(CLASS_NAMES), "members": len(m.sessions), "input_sizes": m.sizes,
                "tta_views": 4, "description": m.description, "disclaimer": DISCLAIMER}

    # A plain ``def`` endpoint: FastAPI runs it in a worker thread, so the blocking ONNX inference
    # does not stall the event loop.
    @app.post("/predict")
    def predict(file: UploadFile = File(...)) -> dict:
        data = file.file.read(MAX_BYTES + 1)  # never hold more than the limit plus one byte
        if len(data) > MAX_BYTES:
            raise HTTPException(status_code=413, detail="image larger than 10 MB")
        try:
            image = decode_image(data)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        t0 = time.perf_counter()
        probs = get_model().predict_probs(image)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        order = np.argsort(probs)[::-1]
        return {
            "prediction": CLASS_NAMES[int(order[0])],
            "probabilities": {c: round(float(p), 6) for c, p in zip(CLASS_NAMES, probs)},
            "latency_ms": round(latency_ms, 1),
            "disclaimer": DISCLAIMER,
        }

    return app


app = create_app()
