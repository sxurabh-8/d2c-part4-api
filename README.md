# Part 4 — FastAPI Churn Scoring Service

Production-style FastAPI service that loads the Part 3 model and exposes prediction endpoints for the internal CRM.

## Endpoints
| Method | Path | Purpose |
|---|---|---|
| GET  | `/health`        | Health check + model-loaded status |
| POST | `/predict`       | Score one customer |
| POST | `/batch_predict` | Score up to 1,000 customers |

## Project Structure
```
part4_api/
├── app/main.py          # FastAPI app, Pydantic schemas, scoring logic
├── train_model.py       # (re)trains and writes model.pkl from ../data/
├── tests/test_api.py    # 6 API tests
├── model.pkl            # serialized model artifact (joblib)
├── requirements.txt
├── Dockerfile
├── monitoring_plan.md
└── README.md
```

## Setup
```bash
pip install -r requirements.txt
# model.pkl is already included. To retrain from the raw snapshot:
python train_model.py
```

## Run the API
```bash
uvicorn app.main:app --reload --port 8000
# OpenAPI docs: http://localhost:8000/docs
```

## Run tests
```bash
pytest -q
```

## Docker
```bash
docker build -t churn-api .
docker run -p 8000:8000 churn-api
```

## Sample Request (`POST /predict`)
```json
{
  "customer_id": "CUST00001",
  "city_tier": "Tier 1",
  "age_group": "18-24",
  "acquisition_channel": "Instagram",
  "loyalty_tier": "Silver",
  "preferred_category": "Makeup",
  "marketing_consent": "Yes",
  "recency_days": 107,
  "frequency_180d": 1,
  "monetary_180d": 362.73,
  "return_rate_180d": 0.0,
  "avg_discount_pct_180d": 0.23,
  "avg_rating_180d": 3.0,
  "category_diversity_180d": 1,
  "ticket_count_90d": 0,
  "negative_ticket_rate_90d": 0,
  "avg_resolution_hours_90d": 0,
  "days_since_signup": 524,
  "sessions_30d": 1,
  "product_views_30d": 4,
  "cart_adds_30d": 0,
  "wishlist_adds_30d": 0,
  "abandoned_carts_30d": 0,
  "email_opens_30d": 2,
  "campaign_clicks_30d": 0,
  "last_visit_days_ago": 20
}
```

## Sample Response
```json
{
  "customer_id": "CUST00001",
  "churn_probability": 0.78,
  "predicted_class": 1,
  "risk_level": "high",
  "risk_explanation": "Elevated churn risk: long inactivity (107 days since last order); only 1 web session in 30 days."
}
```

## Sample Batch Request (`POST /batch_predict`)
```json
{ "customers": [ { ... features ... }, { ... features ... } ] }
```

## Model / Data Notes
- The model is a `GradientBoostingClassifier` trained on `data/rfm_modeling_snapshot.csv` (snapshot date 2025-09-30).
- Threshold tuned on validation set for best F1 (see `metrics.json` in Part 3).
- All features are pre-snapshot — no label leakage.
- See `monitoring_plan.md` for drift/retraining plan and responsible-use guidance.
