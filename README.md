# backend-foresight

> Service FastAPI d'inférence pour le POC ML Foresight.
> Projet école Albert School · 2026-05.

![Status](https://img.shields.io/badge/status-design--phase-orange) · Python 3.11 · FastAPI

## Rôle

API REST qui charge le modèle ML entraîné dans
[ml-foresight](https://github.com/foresight-ml-poc/ml-foresight) et expose des
endpoints d'inférence pour le frontend (et tout autre client).

Au démarrage, télécharge `best_model.pkl`, `scaler.pkl`, et `model_card.json`
depuis la dernière GitHub Release du repo `ml-foresight`.

## Endpoints prévus

| Méthode | Endpoint | Description |
|---|---|---|
| `GET`  | `/healthz` | Liveness probe |
| `GET`  | `/readyz`  | Readiness probe (modèle chargé en mémoire) |
| `GET`  | `/model/info` | Renvoie `model_card.json` (version, features, métriques) |
| `POST` | `/predict` | Prédit pour un signal (~20 features) |
| `POST` | `/predict/batch` | Prédit pour N signaux en une requête |

## Architecture (3 repos)

| Repo | Rôle |
|---|---|
| [ml-foresight](https://github.com/foresight-ml-poc/ml-foresight) | Pipeline ML, training, évaluation, Streamlit dashboard |
| **backend-foresight** *(ici)* | API FastAPI d'inférence |
| [frontend-foresight](https://github.com/foresight-ml-poc/frontend-foresight) | Demo React |

## Documentation

- 📄 **[Design document complet](./docs/specs/2026-05-05-design.md)** —
  voir notamment §9 pour les détails de ce repo

## Quickstart (à venir)

```bash
# 1. Setup
pip install -r requirements.txt
cp .env.example .env   # renseigner MODEL_RELEASE_URL et MODEL_VERSION

# 2. Lancer en dev
uvicorn app.main:app --reload --port 8000
# → http://localhost:8000/docs (Swagger UI)
# → http://localhost:8000/healthz
```

## Status

🚧 **Phase de design.** Code à venir. Voir le
[design doc](./docs/specs/2026-05-05-design.md) pour le plan complet.

## License

MIT
