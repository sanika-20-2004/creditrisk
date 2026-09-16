"""
Central configuration — all feature names, DB settings, thresholds.
Imported by every module so nothing ever mismatches.
"""
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Exact features the XGBoost model is trained on ────────────────────────────
MODEL_FEATURES = [
    "age", "income_scaled", "loan_amount_scaled", "loan_tenure",
    "avg_monthly_credit_scaled", "avg_monthly_debit_scaled",
    "salary_regularity", "bill_payment_score", "merchant_diversity",
    "existing_emis", "overdraft_count", "savings_ratio",
    "is_postpaid", "recharge_regularity", "avg_recharge_amount_scaled",
    "num_financial_apps", "night_usage_ratio", "account_age_months",
    "home_stability_months", "location_changes_6m", "work_regularity",
    "home_work_distance_km_scaled", "vague_purpose_flag",
    "debt_to_income", "credit_debit_ratio", "emi_burden_ratio",
    "stability_score", "mobile_trust_score", "behavioral_risk_score",
    "loan_to_income_ratio", "city_tier_encoded", "employment_encoded",
] + [f"text_dim_{i}" for i in range(20)]

RISK_THRESHOLDS = {"low": 0.30, "medium": 0.55, "high": 0.75}

DB_PATH  = os.path.join(BASE_DIR, "data", "credit_risk.db")
DB_URL   = f"sqlite:///{DB_PATH}"

MODEL_PATH = os.path.join(BASE_DIR, "models", "saved", "xgb_model.pkl")
TFIDF_PATH = os.path.join(BASE_DIR, "models", "saved", "tfidf.pkl")
SVD_PATH   = os.path.join(BASE_DIR, "models", "saved", "svd.pkl")
