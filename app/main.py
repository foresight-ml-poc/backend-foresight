"""FastAPI app for the ml-foresight inference service."""

from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.config import ALLOWED_ORIGINS
from app.inference import predict_batch, predict_one
from app.model_loader import get_model_bundle
from app.schemas import (
    BatchInput,
    BatchPrediction,
    Prediction,
    SignalInput,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("backend-foresight")

app = FastAPI(
    title="backend-foresight",
    description=(
        "Inference service for the Foresight ML POC. "
        "Loads the best_model.joblib from ml-foresight v1.1.0 release "
        "and exposes /predict and /predict/batch."
    ),
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _load_on_startup() -> None:
    """Eagerly load the bundle so the first request doesn't pay the cost."""
    try:
        bundle = get_model_bundle()
        log.info("Startup OK — model %s loaded with %d features",
                 bundle.model_card.get("best_model_type"),
                 len(bundle.feature_order))
    except Exception:
        log.exception("Failed to load model bundle on startup")
        # Don't raise — let /healthz report it later. Tests can mock the loader.


@app.get("/healthz")
def healthz() -> dict:
    """Liveness probe."""
    return {"status": "ok"}


@app.get("/readyz")
def readyz() -> dict:
    """Readiness probe — confirms the model bundle is loaded."""
    try:
        bundle = get_model_bundle()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Model not loaded: {exc}")
    return {
        "status": "ready",
        "model_type": bundle.model_card.get("best_model_type"),
        "model_version": bundle.model_card.get("model_version"),
        "features": len(bundle.feature_order),
    }


@app.get("/model/info")
def model_info() -> dict:
    """Return the full model_card.json for the loaded model."""
    bundle = get_model_bundle()
    return bundle.model_card


@app.post("/predict", response_model=Prediction)
def predict(signal: SignalInput) -> Prediction:
    """Predict the win probability for a single signal."""
    bundle = get_model_bundle()
    return predict_one(signal, bundle)


@app.post("/predict/batch", response_model=BatchPrediction)
def predict_batch_route(payload: BatchInput) -> BatchPrediction:
    """Predict for multiple signals in one request."""
    bundle = get_model_bundle()
    return BatchPrediction(predictions=predict_batch(payload.signals, bundle))
