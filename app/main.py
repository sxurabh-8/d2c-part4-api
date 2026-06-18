"""FastAPI churn-scoring service."""
from __future__ import annotations
from pathlib import Path
from typing import List, Literal, Optional
import joblib, pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict

MODEL_PATH = Path(__file__).resolve().parent.parent / "model.pkl"

app = FastAPI(title="D2C Churn Scoring API", version="1.0.0",
              description="Returns 60-day churn probability for D2C customers.")

_artifact = None
def get_artifact():
    global _artifact
    if _artifact is None:
        if not MODEL_PATH.exists():
            raise RuntimeError(f"Model file not found at {MODEL_PATH}. Run train_model.py first.")
        _artifact = joblib.load(MODEL_PATH)
    return _artifact


class CustomerFeatures(BaseModel):
    model_config = ConfigDict(extra="allow")  # accept any extra cols, we filter to feature_cols
    customer_id: Optional[str] = None
    city_tier: str = Field(..., examples=["Tier 1"])
    age_group: str = Field(..., examples=["25-34"])
    acquisition_channel: str = Field(..., examples=["Instagram"])
    loyalty_tier: str = Field("None", examples=["Silver", "Gold", "None"])
    preferred_category: str = Field(..., examples=["Skin Care"])
    marketing_consent: str = Field(..., examples=["Yes", "No"])
    recency_days: float = Field(..., ge=0)
    frequency_180d: float = Field(..., ge=0)
    monetary_180d: float = Field(..., ge=0)
    return_rate_180d: float = Field(0.0, ge=0, le=1)
    avg_discount_pct_180d: float = Field(0.0, ge=0, le=1)
    avg_rating_180d: float = Field(0.0)
    category_diversity_180d: float = Field(0.0, ge=0)
    ticket_count_90d: float = Field(0.0, ge=0)
    negative_ticket_rate_90d: float = Field(0.0, ge=0, le=1)
    avg_resolution_hours_90d: float = Field(0.0, ge=0)
    days_since_signup: float = Field(..., ge=0)
    sessions_30d: float = Field(0.0, ge=0)
    product_views_30d: float = Field(0.0, ge=0)
    cart_adds_30d: float = Field(0.0, ge=0)
    wishlist_adds_30d: float = Field(0.0, ge=0)
    abandoned_carts_30d: float = Field(0.0, ge=0)
    email_opens_30d: float = Field(0.0, ge=0)
    campaign_clicks_30d: float = Field(0.0, ge=0)
    last_visit_days_ago: float = Field(0.0, ge=0)


class PredictionResponse(BaseModel):
    customer_id: Optional[str] = None
    churn_probability: float
    predicted_class: int
    risk_level: Literal["low", "medium", "high"]
    risk_explanation: str


class BatchRequest(BaseModel):
    customers: List[CustomerFeatures]


def _explain(row: dict, proba: float) -> str:
    reasons = []
    if row.get("recency_days", 0) > 90:
        reasons.append(f"long inactivity ({int(row['recency_days'])} days since last order)")
    if row.get("sessions_30d", 0) == 0:
        reasons.append("zero web/app sessions in last 30 days")
    if row.get("ticket_count_90d", 0) >= 2:
        reasons.append(f"{int(row['ticket_count_90d'])} support tickets in 90 days")
    if row.get("negative_ticket_rate_90d", 0) >= 0.5:
        reasons.append("majority of recent support tickets were negative")
    if row.get("frequency_180d", 0) == 0:
        reasons.append("no orders in last 180 days")
    if not reasons:
        if proba < 0.3:
            return "Recent activity and order frequency indicate low churn risk."
        return "No single strong signal — risk driven by combined feature interactions."
    return "Elevated churn risk: " + "; ".join(reasons) + "."


def _score(features_list: List[dict]) -> List[dict]:
    art = get_artifact()
    feat_cols = art["feature_cols"]
    cat_cols = art["cat_cols"]; num_cols = art["num_cols"]
    df = pd.DataFrame(features_list)
    for c in feat_cols:
        if c not in df.columns:
            df[c] = "Unknown" if c in cat_cols else 0
    df[cat_cols] = df[cat_cols].fillna("Unknown").astype(str)
    df[num_cols] = df[num_cols].fillna(0)
    X = df[feat_cols]
    probas = art["model"].predict_proba(X)[:, 1]
    thr = art["threshold"]
    out = []
    for i, p in enumerate(probas):
        p = float(p)
        if p < 0.3: lvl = "low"
        elif p < 0.6: lvl = "medium"
        else: lvl = "high"
        out.append({
            "customer_id": features_list[i].get("customer_id"),
            "churn_probability": round(p, 4),
            "predicted_class": int(p >= thr),
            "risk_level": lvl,
            "risk_explanation": _explain(features_list[i], p),
        })
    return out


@app.get("/health")
def health():
    try:
        get_artifact()
        return {"status": "ok", "model_loaded": True}
    except Exception as e:
        return {"status": "degraded", "model_loaded": False, "detail": str(e)}


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: CustomerFeatures):
    try:
        result = _score([payload.model_dump()])[0]
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scoring failed: {e}")


@app.post("/batch_predict", response_model=List[PredictionResponse])
def batch_predict(payload: BatchRequest):
    if not payload.customers:
        raise HTTPException(status_code=400, detail="customers list is empty")
    if len(payload.customers) > 1000:
        raise HTTPException(status_code=400, detail="batch limit is 1000 customers")
    try:
        return _score([c.model_dump() for c in payload.customers])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scoring failed: {e}")
