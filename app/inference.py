"""Convert a SignalInput into a 19-dim feature vector and run prediction."""

from __future__ import annotations

import numpy as np

from app.schemas import Prediction, SignalInput


def signal_to_vector(signal: SignalInput, feature_order: list[str]) -> np.ndarray:
    """Build the feature vector in the exact order the model expects.

    Derived features:
      - is_buy_yes (binary)
      - market_price_centered = abs(market_price_at_signal - 0.5)
      - bucket one-hot (drop_first=True semantics — 'crypto' is the dropped baseline)
    """
    is_buy_yes = 1 if signal.direction == "BUY_YES" else 0
    market_price_centered = abs(signal.market_price_at_signal - 0.5)

    raw_lookup: dict[str, float] = {
        "cosine_score": signal.cosine_score,
        "impact_strength": signal.impact_strength,
        "llm_confidence": signal.llm_confidence,
        "ambiguity_score": signal.ambiguity_score,
        "specificity_score": signal.specificity_score,
        "articles_count": float(signal.articles_count),
        "unique_sources_count": float(signal.unique_sources_count),
        "tier_1_count": float(signal.tier_1_count),
        "tier_2_count": float(signal.tier_2_count),
        "tier_3_count": float(signal.tier_3_count),
        "is_buy_yes": float(is_buy_yes),
        "market_price_centered": market_price_centered,
        "hour_of_day": float(signal.hour_of_day),
    }

    # bucket one-hot — every column starts with "bucket_"
    for col in feature_order:
        if col.startswith("bucket_"):
            bucket_name = col.removeprefix("bucket_")
            raw_lookup[col] = 1.0 if signal.bucket == bucket_name else 0.0

    # Build vector in exact order
    missing = [c for c in feature_order if c not in raw_lookup]
    if missing:
        raise ValueError(f"Feature(s) missing in signal_to_vector: {missing}")

    vector = np.array([raw_lookup[c] for c in feature_order], dtype=np.float32)
    return vector.reshape(1, -1)


def predict_one(signal: SignalInput, bundle) -> Prediction:
    """Predict a single signal."""
    raw = signal_to_vector(signal, bundle.feature_order)
    scaled = bundle.scaler.transform(raw)

    if hasattr(bundle.model, "predict_proba"):
        proba = float(bundle.model.predict_proba(scaled)[0, 1])
    else:
        proba = float(bundle.model.predict(scaled)[0])

    label = 1 if proba >= 0.5 else 0
    card = bundle.model_card

    return Prediction(
        probability_winning=proba,
        predicted_label=label,
        threshold_used=0.5,
        model_version=card.get("model_version", "unknown"),
        model_type=card.get("best_model_type", "unknown"),
    )


def predict_batch(signals: list[SignalInput], bundle) -> list[Prediction]:
    return [predict_one(s, bundle) for s in signals]
