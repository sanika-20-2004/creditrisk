"""
Step 6 — Flask REST API  (UPDATED — unified assign_tier driven by best_threshold.pkl)
Run:  python api/app.py
Test: curl -X POST http://localhost:5000/predict \
      -H "Content-Type: application/json" \
      -d @api/sample_input.json
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
from sqlalchemy import create_engine
from datetime import datetime
from sklearn.calibration import CalibratedClassifierCV

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH    = os.path.join(BASE_DIR, "data", "credit_risk.db")
DB_URL     = f"sqlite:///{DB_PATH}"
MODELS_DIR = os.path.join(BASE_DIR, "models", "saved")

app = Flask(__name__)
CORS(app)  # ← must be here, once, right after app is created

print("Loading model artifacts...")
cal_model  = joblib.load(os.path.join(MODELS_DIR, "xgb_model.pkl"))
tfidf      = joblib.load(os.path.join(MODELS_DIR, "tfidf.pkl"))
svd        = joblib.load(os.path.join(MODELS_DIR, "svd.pkl"))
feat_names = joblib.load(os.path.join(MODELS_DIR, "feature_names.pkl"))
thresh_path = os.path.join(MODELS_DIR, "best_threshold.pkl")
BEST_THRESHOLD = joblib.load(thresh_path) if os.path.exists(thresh_path) else 0.15
print(f"Model loaded. F1-optimal threshold = {BEST_THRESHOLD:.3f}")

# ── Unified tier thresholds (mirrors streamlit_app.py exactly) ────────────────
LOW_THRESHOLD  = 0.10
MID_THRESHOLD  = BEST_THRESHOLD          # e.g. 0.15
HIGH_THRESHOLD = BEST_THRESHOLD * 2      # e.g. 0.30

def assign_tier(p):
    """Single source of truth for risk tier assignment."""
    if p < LOW_THRESHOLD:  return "Low Risk"
    if p < MID_THRESHOLD:  return "Medium Risk"
    if p < HIGH_THRESHOLD: return "High Risk"
    return "Very High Risk"

def to_display_score(p):
    """Rescale raw model probability to a user-friendly 0–100 score.
    Model thresholds are completely unchanged. Only the displayed number changes.
    Low < 33 | Medium 33–66 | High 66–91 | Very High 91–100
    """
    if p < 0.10: return (p / 0.10) * 33
    if p < 0.30: return 33 + ((p - 0.10) / 0.20) * 33
    if p < 0.60: return 66 + ((p - 0.30) / 0.30) * 25
    return        91 + ((p - 0.60) / 0.40) * 9

ACTION_MAP = {
    "Low Risk":       "Auto-approve",
    "Medium Risk":    "Manual review",
    "High Risk":      "Reject or collateral",
    "Very High Risk": "Reject",
}

def norm(v, lo, hi):
    return float(max(0.0, min(1.0, (v - lo) / (hi - lo + 1e-9))))

def build_features(data):
    income        = float(data.get("income", 45_000))
    loan_amount   = float(data.get("loan_amount", 250_000))
    tenure        = int(data.get("loan_tenure", 24))
    employment    = data.get("employment_type", "salaried")
    city_tier     = int(data.get("city_tier", 2))
    salary_reg    = float(data.get("salary_regularity", 0.75))
    bill_score    = float(data.get("bill_payment_score", 0.75))
    savings_ratio = float(data.get("savings_ratio", 0.20))
    existing_emis = int(data.get("existing_emis", 1))
    overdrafts    = int(data.get("overdraft_count", 0))
    is_postpaid   = int(data.get("is_postpaid", 1))
    recharge_reg  = float(data.get("recharge_regularity", 0.75))
    avg_recharge  = float(data.get("avg_recharge_amount", 399))
    num_fin_apps  = int(data.get("num_financial_apps", 3))
    night_usage   = float(data.get("night_usage_ratio", 0.20))
    home_stab     = int(data.get("home_stability_months", 18))
    loc_changes   = int(data.get("location_changes_6m", 1))
    work_reg      = float(data.get("work_regularity", 0.75))
    acc_age       = int(data.get("account_age_months", 24))
    loan_text     = str(data.get("loan_purpose_text", "working cash loans"))

    income_s   = norm(income, 5_000, 500_000)
    loan_s     = norm(loan_amount, 10_000, 5_000_000)
    recharge_s = norm(avg_recharge, 50, 1_500)
    city_enc   = {1: 1.0, 2: 0.6, 3: 0.3}.get(city_tier, 0.6)
    emp_enc    = {"salaried": 1.0, "self_employed": 0.7, "gig": 0.4, "student": 0.2}.get(employment, 0.7)

    credit       = income * 1.05
    debit        = income * (1 - savings_ratio)
    dti          = loan_amount / (income + 1)
    emi_burden   = (existing_emis * debit * 0.1) / (income + 1)
    credit_debit = credit / (debit + 1)

    stability    = (home_stab / 60) * 0.5 + work_reg * 0.3 + max(0, 1 - loc_changes / 5) * 0.2
    mobile_trust = (is_postpaid * 0.3 + recharge_reg * 0.3
                    + min(num_fin_apps / 10, 1) * 0.2 + (1 - night_usage) * 0.2)
    beh_risk     = overdrafts * 0.3 + (1 - bill_score) * 0.3 + (1 - salary_reg) * 0.4

    m_emi    = loan_s / (tenure + 1)
    rep_inc  = m_emi / (income_s + 1e-9)

    vague_words = ["urgent", "miscellaneous", "general", "personal use", "various", "soon"]
    vague_flag  = int(any(w in loan_text.lower() for w in vague_words))

    row = {
        "age":                          35,
        "income_scaled":                income_s,
        "loan_amount_scaled":           loan_s,
        "loan_tenure":                  tenure,
        "avg_monthly_credit_scaled":    norm(credit, 5_000, 600_000),
        "avg_monthly_debit_scaled":     norm(debit, 3_000, 550_000),
        "salary_regularity":            salary_reg,
        "bill_payment_score":           bill_score,
        "merchant_diversity":           10,
        "existing_emis":                existing_emis,
        "overdraft_count":              overdrafts,
        "savings_ratio":                savings_ratio,
        "is_postpaid":                  is_postpaid,
        "recharge_regularity":          recharge_reg,
        "avg_recharge_amount_scaled":   recharge_s,
        "num_financial_apps":           num_fin_apps,
        "night_usage_ratio":            night_usage,
        "account_age_months":           acc_age,
        "home_stability_months":        home_stab,
        "location_changes_6m":          loc_changes,
        "work_regularity":              work_reg,
        "home_work_distance_km_scaled": 0.1,
        "vague_purpose_flag":           vague_flag,
        "debt_to_income":               dti,
        "credit_debit_ratio":           credit_debit,
        "emi_burden_ratio":             emi_burden,
        "stability_score":              stability,
        "mobile_trust_score":           mobile_trust,
        "behavioral_risk_score":        beh_risk,
        "loan_to_income_ratio":         dti,
        "city_tier_encoded":            city_enc,
        "employment_encoded":           emp_enc,
        "risk_x_dti":                   beh_risk * dti,
        "income_x_stability":           income_s * stability,
        "repayment_to_income":          rep_inc,
        "monthly_emi_estimate":         m_emi,
        "savings_x_stability":          savings_ratio * stability,
    }

    # NLP text features
    text_svd = svd.transform(tfidf.transform([loan_text]))
    for i, v in enumerate(text_svd[0]):
        row[f"text_dim_{i}"] = v

    X = pd.DataFrame([row])
    for col in feat_names:
        if col not in X.columns:
            X[col] = 0.0
    return X[feat_names]

def log_to_db(data, proba, tier):
    """Non-blocking DB write — prints warning on failure instead of raising."""
    try:
        engine = create_engine(DB_URL, echo=False)
        pd.DataFrame([{
            "income":              data.get("income", 0),
            "loan_amount":         data.get("loan_amount", 0),
            "employment_type":     data.get("employment_type", ""),
            "loan_purpose_text":   data.get("loan_purpose_text", ""),
            "default_probability": round(proba, 4),
            "risk_tier":           tier,
            "recommendation":      ACTION_MAP[tier],
            "shap_reason_1":       "",
            "shap_reason_2":       "",
            "shap_reason_3":       "",
            "created_at":          datetime.utcnow().isoformat(),
        }]).to_sql("live_prediction_log", engine, if_exists="append", index=False)
    except Exception as e:
        print(f"[WARN] DB log failed: {e}")

# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status":         "ok",
        "model":          "CreditRisk-XGBoost-v2",
        "features":       len(feat_names),
        "best_threshold": round(BEST_THRESHOLD, 3),
        "tier_boundaries": {
            "Low Risk":       f"p < {LOW_THRESHOLD:.2f}",
            "Medium Risk":    f"{LOW_THRESHOLD:.2f} ≤ p < {MID_THRESHOLD:.2f}",
            "High Risk":      f"{MID_THRESHOLD:.2f} ≤ p < {HIGH_THRESHOLD:.2f}",
            "Very High Risk": f"p ≥ {HIGH_THRESHOLD:.2f}",
        }
    })

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data  = request.get_json(force=True)
        X     = build_features(data)
        proba = float(cal_model.predict_proba(X)[0][1])
        tier  = assign_tier(proba)

        result = {
            "applicant_id":        data.get("applicant_id", "N/A"),
            "default_probability": round(proba, 4),
            "display_score":       round(to_display_score(proba), 1),
            "risk_tier":           tier,
            "recommendation":      ACTION_MAP[tier],
            "tier_boundaries": {
                "low_max":   LOW_THRESHOLD,
                "mid_max":   MID_THRESHOLD,
                "high_max":  HIGH_THRESHOLD,
            },
        }
        log_to_db(data, proba, tier)
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/batch_predict", methods=["POST"])
def batch_predict():
    try:
        records = request.get_json(force=True)
        results = []
        for rec in records:
            X    = build_features(rec)
            prob = float(cal_model.predict_proba(X)[0][1])
            tier = assign_tier(prob)
            results.append({
                "applicant_id":        rec.get("applicant_id", "N/A"),
                "default_probability": round(prob, 4),
                "risk_tier":           tier,
                "recommendation":      ACTION_MAP[tier],
            })
        return jsonify({"count": len(results), "results": results})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)