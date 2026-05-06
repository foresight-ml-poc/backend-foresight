"""Backend configuration via env vars."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")
load_dotenv(PROJECT_ROOT / ".env.local")

# Model loading: prefer local dir for dev, fall back to GitHub Release URLs
MODEL_LOCAL_DIR = os.environ.get("MODEL_LOCAL_DIR", "").strip()
MODEL_RELEASE_BASE_URL = os.environ.get("MODEL_RELEASE_BASE_URL", "").strip()
MODEL_VERSION = os.environ.get("MODEL_VERSION", "v1.1.0").strip()

# CORS
_origins = os.environ.get("ALLOWED_ORIGINS", "http://localhost:5173").strip()
ALLOWED_ORIGINS = [o.strip() for o in _origins.split(",") if o.strip()]

# Where downloaded artifacts get cached when MODEL_RELEASE_BASE_URL is used
ARTIFACT_CACHE_DIR = PROJECT_ROOT / ".artifacts"
