# Monitoring & Responsible-Use Plan

## What to Monitor After Deployment

### 1. Data Drift (input)
- Track distribution of every feature weekly vs. the training distribution.
  - Numeric: Population Stability Index (PSI). Alert if PSI > 0.2 on any feature.
  - Categorical: Chi-squared distribution shift on `acquisition_channel`, `city_tier`, `loyalty_tier`.
- Watch for new categorical values (one-hot encoder maps them to all-zero — silently degrades).

### 2. Prediction Drift (output)
- Daily distribution of `churn_probability` (mean, P50, P90).
- Daily share of predictions in each `risk_level` bucket.
- Alert if `high` share moves >10 percentage points vs. trailing 30-day baseline.

### 3. Model Performance (when labels arrive, 60 days later)
- Rolling 60-day PR-AUC and recall@chosen-threshold.
- Confusion matrix split by acquisition channel and loyalty tier — to catch fairness regressions.
- Alert if PR-AUC drops > 0.05 vs. last quarter.

### 4. Business Outcomes
- Compare 60-day churn rate of treated `high`-risk customers vs. a hold-out control group.
- Track incremental revenue retained per ₹ spent on retention by segment.
- Track Champions/Loyal AOV — guard against discount cannibalization.

### 5. API Health
- p50 / p95 latency on `/predict` and `/batch_predict`.
- 5xx error rate; alert if > 0.5% over 5 min.
- Model-loading failures (logged from `/health`).

### 6. Retraining Triggers
- PSI > 0.2 on any top-5 importance feature.
- PR-AUC drop > 0.05 vs. baseline for 2 consecutive weeks.
- Quarterly retrain regardless (calendar-based).
- Any change in upstream feature definition (recency window, support window).

## Responsible-Use Note for the Retention Team

**Do:**
- Use predictions to **prioritize** outreach within the retention budget.
- Combine with segment context (At-Risk High Value vs. Hibernating) before deciding the offer.
- Treat the score as one input — not a verdict.

**Do NOT:**
- Use the score to deny service, change pricing, or alter product access for any customer.
- Share raw probabilities with the customer.
- Run a campaign exclusively against a single demographic slice (city_tier / age_group) without a fairness review.
- Treat a `low` risk score as a guarantee — see false-negative examples in Part 3 `error_analysis.md`.

**Escalation:** If a customer appears repeatedly in `high` risk but has filed complaints, route to CS before sending any marketing offer.
