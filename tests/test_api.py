"""API test suite. Run with: pytest -q"""
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

SAMPLE_HIGH_RISK = {
    "customer_id": "TEST_HIGH",
    "city_tier": "Tier 2", "age_group": "25-34", "acquisition_channel": "Marketplace",
    "loyalty_tier": "None", "preferred_category": "Skin Care", "marketing_consent": "Yes",
    "recency_days": 250, "frequency_180d": 0, "monetary_180d": 0,
    "return_rate_180d": 0, "avg_discount_pct_180d": 0, "avg_rating_180d": 0,
    "category_diversity_180d": 0, "ticket_count_90d": 0, "negative_ticket_rate_90d": 0,
    "avg_resolution_hours_90d": 0, "days_since_signup": 500, "sessions_30d": 0,
    "product_views_30d": 0, "cart_adds_30d": 0, "wishlist_adds_30d": 0,
    "abandoned_carts_30d": 0, "email_opens_30d": 0, "campaign_clicks_30d": 0,
    "last_visit_days_ago": 60,
}
SAMPLE_LOW_RISK = {
    "customer_id": "TEST_LOW",
    "city_tier": "Tier 1", "age_group": "25-34", "acquisition_channel": "Organic",
    "loyalty_tier": "Gold", "preferred_category": "Skin Care", "marketing_consent": "Yes",
    "recency_days": 5, "frequency_180d": 6, "monetary_180d": 4500,
    "return_rate_180d": 0, "avg_discount_pct_180d": 0.2, "avg_rating_180d": 4.5,
    "category_diversity_180d": 3, "ticket_count_90d": 0, "negative_ticket_rate_90d": 0,
    "avg_resolution_hours_90d": 0, "days_since_signup": 400, "sessions_30d": 12,
    "product_views_30d": 60, "cart_adds_30d": 3, "wishlist_adds_30d": 2,
    "abandoned_carts_30d": 1, "email_opens_30d": 8, "campaign_clicks_30d": 2,
    "last_visit_days_ago": 2,
}

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] in ("ok", "degraded")

def test_predict_high_risk():
    r = client.post("/predict", json=SAMPLE_HIGH_RISK)
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["churn_probability"] <= 1
    assert body["predicted_class"] in (0, 1)
    assert body["risk_level"] in ("low", "medium", "high")
    assert body["churn_probability"] > 0.5  # dormant profile should score high

def test_predict_low_risk():
    r = client.post("/predict", json=SAMPLE_LOW_RISK)
    assert r.status_code == 200
    body = r.json()
    assert body["churn_probability"] < 0.5  # engaged Gold customer should score low

def test_batch_predict():
    r = client.post("/batch_predict", json={"customers": [SAMPLE_HIGH_RISK, SAMPLE_LOW_RISK]})
    assert r.status_code == 200
    body = r.json()
    assert len(body) == 2
    assert body[0]["churn_probability"] > body[1]["churn_probability"]

def test_validation_error():
    bad = dict(SAMPLE_LOW_RISK); bad["recency_days"] = -5
    r = client.post("/predict", json=bad)
    assert r.status_code == 422

def test_empty_batch():
    r = client.post("/batch_predict", json={"customers": []})
    assert r.status_code == 400
