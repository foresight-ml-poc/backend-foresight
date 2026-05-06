"""Pydantic schemas for the /predict endpoint.

Two layers:
  - SignalInput: the user-facing "raw" signal (what Foresight has at emission time)
  - The 19-dim feature vector is computed from SignalInput inside inference.py

Output:
  - Prediction: probability + label + model metadata
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Bucket = Literal["geopolitics", "politics", "sports", "crypto",
                  "economics", "science", "other"]
Direction = Literal["BUY_YES", "BUY_NO"]


class SignalInput(BaseModel):
    """Raw signal data, as it would be emitted by Foresight."""

    direction: Direction
    market_price_at_signal: float = Field(ge=0.0, le=1.0)

    # 4 LLM features (from event_market_analysis)
    impact_strength: float = Field(ge=0.0, le=1.0)
    llm_confidence: float = Field(ge=0.0, le=1.0)
    ambiguity_score: float = Field(ge=0.0, le=1.0)
    specificity_score: float = Field(ge=0.0, le=1.0)

    # Retrieval quality
    cosine_score: float = Field(ge=0.0, le=1.0)

    # Source tier counts
    tier_1_count: int = Field(ge=0)
    tier_2_count: int = Field(ge=0)
    tier_3_count: int = Field(ge=0)

    # Event context
    bucket: Bucket
    articles_count: int = Field(ge=1)
    unique_sources_count: int = Field(ge=1)

    # Time of day (signal emission)
    hour_of_day: int = Field(ge=0, le=23)


class Prediction(BaseModel):
    """Resolves the `model_*` field-name warning from Pydantic v2."""
    model_config = {"protected_namespaces": ()}

    probability_winning: float = Field(ge=0.0, le=1.0)
    predicted_label: int = Field(ge=0, le=1)
    threshold_used: float = 0.5
    model_version: str
    model_type: str


class BatchInput(BaseModel):
    signals: list[SignalInput]


class BatchPrediction(BaseModel):
    predictions: list[Prediction]
