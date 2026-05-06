"""Load the trained model + scaler + feature_order + model_card.

Two modes:
  1. Local dir (MODEL_LOCAL_DIR set) — read directly from disk. For dev.
  2. GitHub Release (MODEL_RELEASE_BASE_URL set) — download to ARTIFACT_CACHE_DIR.

Exposes a singleton via `get_model_bundle()`.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import joblib

from app.config import (
    ARTIFACT_CACHE_DIR,
    MODEL_LOCAL_DIR,
    MODEL_RELEASE_BASE_URL,
    MODEL_VERSION,
)

logger = logging.getLogger(__name__)

ARTIFACT_FILES = ["best_model.joblib", "scaler.pkl", "feature_order.pkl",
                  "model_card.json"]


@dataclass
class ModelBundle:
    model: Any
    scaler: Any
    feature_order: list[str]
    model_card: dict


_bundle: ModelBundle | None = None


def _resolve_local_path(filename: str) -> Path | None:
    """Resolve a single artifact via either local dir or GitHub Release download."""
    if MODEL_LOCAL_DIR:
        p = Path(MODEL_LOCAL_DIR) / filename
        if not p.exists():
            raise FileNotFoundError(f"Local model file missing: {p}")
        return p

    if MODEL_RELEASE_BASE_URL:
        ARTIFACT_CACHE_DIR.mkdir(exist_ok=True)
        cached = ARTIFACT_CACHE_DIR / filename
        if cached.exists():
            return cached
        url = f"{MODEL_RELEASE_BASE_URL.rstrip('/')}/{filename}"
        logger.info("Downloading %s from %s", filename, url)
        with httpx.stream("GET", url, follow_redirects=True, timeout=60) as r:
            r.raise_for_status()
            with cached.open("wb") as f:
                for chunk in r.iter_bytes():
                    f.write(chunk)
        return cached

    raise RuntimeError(
        "Neither MODEL_LOCAL_DIR nor MODEL_RELEASE_BASE_URL is set. "
        "Configure one in .env or .env.local."
    )


def load_bundle() -> ModelBundle:
    """Load all 4 artifacts and return a populated ModelBundle."""
    paths = {f: _resolve_local_path(f) for f in ARTIFACT_FILES}

    model = joblib.load(paths["best_model.joblib"])
    scaler = joblib.load(paths["scaler.pkl"])
    feature_order = joblib.load(paths["feature_order.pkl"])
    with paths["model_card.json"].open() as f:
        card = json.load(f)

    logger.info(
        "Loaded model bundle: type=%s version=%s features=%d",
        card.get("best_model_type", "?"),
        card.get("model_version", MODEL_VERSION),
        len(feature_order),
    )
    return ModelBundle(
        model=model,
        scaler=scaler,
        feature_order=feature_order,
        model_card=card,
    )


def get_model_bundle() -> ModelBundle:
    """Return the singleton bundle, loading on first call."""
    global _bundle
    if _bundle is None:
        _bundle = load_bundle()
    return _bundle


def reset_bundle() -> None:
    """Test-only helper: force reload on next get_model_bundle()."""
    global _bundle
    _bundle = None
