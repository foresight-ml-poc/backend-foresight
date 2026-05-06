# backend-foresight

API FastAPI qui sert le modèle ML entraîné dans [ml-foresight](https://github.com/foresight-ml-poc/ml-foresight). Au démarrage, télécharge `best_model.joblib`, `scaler.pkl`, `feature_order.pkl`, et `model_card.json` depuis la [GitHub Release v1.1.0](https://github.com/foresight-ml-poc/ml-foresight/releases/tag/v1.1.0).

## Endpoints

- `GET /healthz` — liveness
- `GET /readyz` — readiness (modèle chargé)
- `GET /model/info` — renvoie `model_card.json`
- `POST /predict` — prédit un signal (input : 14 champs)
- `POST /predict/batch` — N signaux

## Quickstart

```bash
pip install -r requirements.txt

# Pour dev local : pointer vers le repo ml-foresight cloné à côté
echo "MODEL_LOCAL_DIR=/Users/vadim/foresight-ml-poc/ml-foresight/models" > .env.local

# Pour prod : utiliser la GitHub Release
# echo "MODEL_RELEASE_BASE_URL=https://github.com/foresight-ml-poc/ml-foresight/releases/download/v1.1.0" > .env.local

uvicorn app.main:app --reload --port 8000
# → http://localhost:8000/docs
```

## Test rapide

```bash
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "direction": "BUY_YES",
    "market_price_at_signal": 0.42,
    "impact_strength": 0.75,
    "llm_confidence": 0.82,
    "ambiguity_score": 0.18,
    "specificity_score": 0.7,
    "cosine_score": 0.65,
    "tier_1_count": 2,
    "tier_2_count": 1,
    "tier_3_count": 0,
    "bucket": "geopolitics",
    "articles_count": 4,
    "unique_sources_count": 3,
    "hour_of_day": 14
  }'
```

→ `{"probability_winning": 0.21, "predicted_label": 0, "model_version": "v1.1.0", ...}`

## Architecture (3 repos)

- [ml-foresight](https://github.com/foresight-ml-poc/ml-foresight) — pipeline ML
- **backend-foresight** (ici)
- [frontend-foresight](https://github.com/foresight-ml-poc/frontend-foresight) — démo React
