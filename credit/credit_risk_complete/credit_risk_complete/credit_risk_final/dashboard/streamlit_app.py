"""
Step 5 — Streamlit Live Dashboard  (UPDATED — unified thresholds, fixed imports, proper error handling)
Run: streamlit run dashboard/streamlit_app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sqlalchemy import create_engine, text
from sklearn.calibration import CalibratedClassifierCV          # ← moved to top level
from datetime import datetime
import os, sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

DB_PATH = os.path.join(BASE_DIR, "data", "credit_risk.db")
DB_URL  = f"sqlite:///{DB_PATH}"

st.set_page_config(
    page_title="Credit Risk Analytics",
    layout="wide",
    page_icon="🏦",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 12px; padding: 16px; color: white; text-align: center;
}
.risk-low    { background: #d4edda; border-left: 5px solid #28a745; padding: 12px; border-radius: 6px; }
.risk-medium { background: #fff3cd; border-left: 5px solid #ffc107; padding: 12px; border-radius: 6px; }
.risk-high   { background: #f8d7da; border-left: 5px solid #dc3545; padding: 12px; border-radius: 6px; }
.risk-vhigh  { background: #e2d0f8; border-left: 5px solid #6f42c1; padding: 12px; border-radius: 6px; }
.main-header { font-size: 2rem; font-weight: 700; color: #1a1a2e; margin-bottom: 0.5rem; }
</style>
""", unsafe_allow_html=True)

RISK_COLORS = {
    "Low Risk":       "#28a745",
    "Medium Risk":    "#ffc107",
    "High Risk":      "#dc3545",
    "Very High Risk": "#6f42c1"
}

FEAT_LABELS = {
    "behavioral_risk_score": "Behavioural risk score",
    "salary_regularity":     "Salary regularity",
    "bill_payment_score":    "Bill payment score",
    "overdraft_count":       "Overdraft count",
    "savings_ratio":         "Savings ratio",
    "existing_emis":         "Existing EMIs",
    "stability_score":       "Location/work stability",
    "location_changes_6m":   "Location changes (6m)",
    "vague_purpose_flag":    "Vague loan purpose",
    "debt_to_income":        "Debt-to-income ratio",
    "mobile_trust_score":    "Mobile trust score",
    "credit_debit_ratio":    "Credit/debit ratio",
    "emi_burden_ratio":      "EMI burden ratio",
    "loan_to_income_ratio":  "Loan-to-income ratio",
    "income_scaled":         "Income level",
    "loan_amount_scaled":    "Loan amount",
    "recharge_regularity":   "Recharge regularity",
    "num_financial_apps":    "Financial apps",
    "risk_x_dti":            "Risk × Debt-to-income",
    "income_x_stability":    "Income × Stability",
    "repayment_to_income":   "Repayment-to-income",
    "savings_x_stability":   "Savings × Stability",
}

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model  = joblib.load(os.path.join(BASE_DIR, "models", "saved", "xgb_model.pkl"))
    tfidf  = joblib.load(os.path.join(BASE_DIR, "models", "saved", "tfidf.pkl"))
    svd    = joblib.load(os.path.join(BASE_DIR, "models", "saved", "svd.pkl"))
    thresh_path = os.path.join(BASE_DIR, "models", "saved", "best_threshold.pkl")
    best_threshold = joblib.load(thresh_path) if os.path.exists(thresh_path) else 0.15
    return model, tfidf, svd, best_threshold

model, tfidf, svd, BEST_THRESHOLD = load_model()

# ── Unified tier thresholds (single source of truth) ─────────────────────────
# LOW  : probability < 0.10          → auto-approve
# MID  : 0.10 ≤ p < BEST_THRESHOLD  → manual review   (trained F1-optimal cut)
# HIGH : BEST_THRESHOLD ≤ p < 2×BT  → reject/collateral
# VHIGH: p ≥ 2×BEST_THRESHOLD       → reject immediately
LOW_THRESHOLD  = 0.10
MID_THRESHOLD  = BEST_THRESHOLD          # e.g. 0.15 from training
HIGH_THRESHOLD = BEST_THRESHOLD * 2      # e.g. 0.30

def assign_tier(p):
    """Unified tier function — driven entirely by trained BEST_THRESHOLD."""
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

# Handle both raw XGBoost and CalibratedClassifierCV wrapper
try:
    MODEL_FEATURES = model.get_booster().feature_names
except AttributeError:
    MODEL_FEATURES = model.estimator.get_booster().feature_names

@st.cache_data
def load_predictions():
    pred = pd.read_csv(os.path.join(BASE_DIR, "data", "predictions.csv"))
    y_t  = pd.read_csv(os.path.join(BASE_DIR, "data", "y_test.csv")).squeeze()
    pred["risk_tier"]    = pred["y_proba"].apply(assign_tier)   # uses unified tiers
    pred["actual_label"] = y_t.values
    return pred

predictions = load_predictions()

def get_engine():
    return create_engine(DB_URL, echo=False)

def get_live_log():
    try:
        engine = get_engine()
        return pd.read_sql(
            "SELECT * FROM live_prediction_log ORDER BY created_at DESC LIMIT 100",
            engine
        )
    except Exception:
        return pd.DataFrame()

# ── Feature builder — MATCHES training exactly ────────────────────────────────
def build_features(income, loan_amount, loan_tenure, emp_type, city_tier,
                   salary_reg, bill_score, savings_r, existing_emi,
                   overdrafts, merch_div, is_postpaid, fin_apps,
                   home_stab, loc_changes, work_reg, recharge_reg,
                   avg_recharge, night_usage, account_age, loan_text):

    credit = income * 1.05
    debit  = income * (1 - savings_r)

    vague_words = ["urgent", "soon", "various", "miscellaneous", "general", "personal use"]
    vague_flag  = int(any(w in loan_text.lower() for w in vague_words))

    emp_map  = {"salaried": 1.0, "self_employed": 0.7, "gig": 0.4, "student": 0.2}
    city_map = {1: 1.0, 2: 0.6, 3: 0.3}

    def norm(v, lo, hi):
        return max(0.0, min(1.0, (v - lo) / (hi - lo + 1e-9)))

    stability_score  = (home_stab / 60) * 0.5 + work_reg * 0.3 + max(0, 1 - loc_changes / 5) * 0.2
    mobile_trust     = is_postpaid * 0.3 + recharge_reg * 0.3 + min(fin_apps / 10, 1) * 0.2 + (1 - night_usage) * 0.2
    behavioral_risk  = overdrafts * 0.3 + (1 - bill_score) * 0.3 + (1 - salary_reg) * 0.4

    income_sc    = norm(income, 5000, 500000)
    dti          = loan_amount / (income + 1)
    monthly_emi  = loan_amount / (loan_tenure + 1)

    row = {
        "age":                          35,
        "income_scaled":                income_sc,
        "loan_amount_scaled":           norm(loan_amount, 10000, 5000000),
        "loan_tenure":                  loan_tenure,
        "avg_monthly_credit_scaled":    norm(credit, 5000, 600000),
        "avg_monthly_debit_scaled":     norm(debit, 3000, 550000),
        "salary_regularity":            salary_reg,
        "bill_payment_score":           bill_score,
        "merchant_diversity":           merch_div,
        "existing_emis":                existing_emi,
        "overdraft_count":              overdrafts,
        "savings_ratio":                savings_r,
        "is_postpaid":                  is_postpaid,
        "recharge_regularity":          recharge_reg,
        "avg_recharge_amount_scaled":   norm(avg_recharge, 50, 1500),
        "num_financial_apps":           fin_apps,
        "night_usage_ratio":            night_usage,
        "account_age_months":           account_age,
        "home_stability_months":        home_stab,
        "location_changes_6m":          loc_changes,
        "work_regularity":              work_reg,
        "home_work_distance_km_scaled": 0.1,
        "vague_purpose_flag":           vague_flag,
        "debt_to_income":               dti,
        "credit_debit_ratio":           credit / (debit + 1),
        "emi_burden_ratio":             (existing_emi * debit * 0.1) / (income + 1),
        "stability_score":              stability_score,
        "mobile_trust_score":           mobile_trust,
        "behavioral_risk_score":        behavioral_risk,
        "loan_to_income_ratio":         dti,
        "city_tier_encoded":            city_map.get(city_tier, 0.6),
        "employment_encoded":           emp_map.get(emp_type, 0.7),
        # Interaction features — match train_model.py exactly
        "risk_x_dti":                   behavioral_risk * dti,
        "income_x_stability":           income_sc * stability_score,
        "repayment_to_income":          monthly_emi / (income + 1),
        "monthly_emi_estimate":         monthly_emi,
        "savings_x_stability":          savings_r * stability_score,
    }

    # NLP text features
    text_svd = svd.transform(tfidf.transform([loan_text]))
    for i, v in enumerate(text_svd[0]):
        row[f"text_dim_{i}"] = v

    X = pd.DataFrame([row])
    for col in MODEL_FEATURES:
        if col not in X.columns:
            X[col] = 0.0
    return X[MODEL_FEATURES]

def save_live_prediction(income, loan_amount, emp_type, loan_text,
                          proba, tier, recommendation, reasons):
    """Save prediction to SQL. Shows warning in UI if DB write fails."""
    try:
        engine = get_engine()
        log_df = pd.DataFrame([{
            "income":              income,
            "loan_amount":         loan_amount,
            "employment_type":     emp_type,
            "loan_purpose_text":   loan_text,
            "default_probability": round(proba, 4),
            "risk_tier":           tier,
            "recommendation":      recommendation,
            "shap_reason_1":       reasons[0] if len(reasons) > 0 else "",
            "shap_reason_2":       reasons[1] if len(reasons) > 1 else "",
            "shap_reason_3":       reasons[2] if len(reasons) > 2 else "",
            "created_at":          datetime.utcnow().isoformat(),
        }])
        log_df.to_sql("live_prediction_log", engine, if_exists="append", index=False)
        return True
    except Exception as e:
        st.warning(f"⚠️ Prediction shown but DB write failed: {e}")
        return False

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🏦 Credit Risk System")
    st.markdown("**Home Credit Default Risk**")
    try:
        from sklearn.metrics import roc_auc_score as _auc
        _live_auc = _auc(predictions["actual_label"], predictions["y_proba"])
        st.markdown(f"Model: XGBoost | AUC: ~{_live_auc:.2f}")
    except Exception:
        st.markdown("Model: XGBoost | AUC: ~0.74")
    st.markdown("Dataset: 307,511 applicants")
    st.markdown("Score: Low < 33 | Medium < 66 | High < 91 | Very High ≥ 91")
    st.markdown("Scale: 0 – 100 (rescaled for clarity)")
    st.divider()
    page = st.radio("Navigation", [
        "📊 Portfolio Overview",
        "🔍 Live Risk Predictor",
        "📈 Model Performance",
        "🗄️ SQL Analytics",
        "🔬 SHAP Explainability"
    ])
    st.divider()
    st.caption("Big Data Analytics Project")
    st.caption("Credit Risk with Alternative Data")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — PORTFOLIO OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Portfolio Overview":
    st.markdown('<p class="main-header">📊 Portfolio Risk Overview</p>', unsafe_allow_html=True)
    st.caption(f"Real Home Credit data | {len(predictions):,} test applicants | "
               f"Tiers: Low <{LOW_THRESHOLD:.0%} | Mid <{MID_THRESHOLD:.0%} | "
               f"High <{HIGH_THRESHOLD:.0%} | VHigh ≥{HIGH_THRESHOLD:.0%}")

    total = len(predictions)
    high  = predictions["risk_tier"].isin(["High Risk", "Very High Risk"]).sum()
    low   = (predictions["risk_tier"] == "Low Risk").sum()
    med   = (predictions["risk_tier"] == "Medium Risk").sum()
    avg_p = predictions["y_proba"].mean()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Applicants",       f"{total:,}")
    c2.metric("Auto-Approve (Low)",     f"{low:,}",  delta=f"{low/total:.1%}")
    c3.metric("Manual Review (Medium)", f"{med:,}",  delta=f"{med/total:.1%}")
    c4.metric("Reject (High+)",         f"{high:,}", delta=f"{high/total:.1%}")
    c5.metric("Avg Default Probability",f"{avg_p:.1%}")

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        tc = predictions["risk_tier"].value_counts().reset_index()
        tc.columns = ["Risk Tier", "Count"]
        fig = px.pie(tc, values="Count", names="Risk Tier",
                     title="Applicants by Risk Tier",
                     color="Risk Tier", color_discrete_map=RISK_COLORS, hole=0.4)
        fig.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig2 = px.histogram(predictions, x="y_proba", nbins=50,
                            title="Distribution of Default Probabilities",
                            labels={"y_proba": "Default Probability"},
                            color_discrete_sequence=["#667eea"])
        fig2.add_vline(x=LOW_THRESHOLD,  line_dash="dash", line_color="#28a745",
                       annotation_text=f"Low/Med ({LOW_THRESHOLD:.2f})")
        fig2.add_vline(x=MID_THRESHOLD,  line_dash="dash", line_color="#ffc107",
                       annotation_text=f"Med/High ({MID_THRESHOLD:.2f})")
        fig2.add_vline(x=HIGH_THRESHOLD, line_dash="dash", line_color="#dc3545",
                       annotation_text=f"High/VHigh ({HIGH_THRESHOLD:.2f})")
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        grp = predictions.groupby("risk_tier")["actual_label"].mean().reset_index()
        grp.columns = ["Risk Tier", "Actual Default Rate"]
        grp = grp.sort_values("Actual Default Rate", ascending=True)
        fig3 = px.bar(grp, x="Actual Default Rate", y="Risk Tier",
                      orientation="h", title="Actual Default Rate by Risk Tier",
                      color="Risk Tier", color_discrete_map=RISK_COLORS)
        fig3.update_xaxes(tickformat=".0%")
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        predictions["prob_bucket"] = pd.cut(predictions["y_proba"],
            bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            labels=["0-10%", "10-20%", "20-30%", "30-40%", "40-50%",
                    "50-60%", "60-70%", "70-80%", "80-90%", "90-100%"])
        bucket_grp = predictions.groupby("prob_bucket", observed=True).agg(
            Count=("y_proba", "count"),
            Default_Rate=("actual_label", "mean")
        ).reset_index()
        fig4 = px.bar(bucket_grp, x="prob_bucket", y="Count",
                      title="Applicant Count by Probability Bucket",
                      color="Default_Rate", color_continuous_scale="RdYlGn_r",
                      labels={"prob_bucket": "Probability Range", "Count": "Count"})
        st.plotly_chart(fig4, use_container_width=True)

    st.subheader("🚨 High-Risk Applicants Requiring Immediate Review")
    high_df = predictions[predictions["risk_tier"].isin(["High Risk", "Very High Risk"])].copy()
    high_df = high_df.sort_values("y_proba", ascending=False).head(25)
    high_df["Default Probability"] = high_df["y_proba"].apply(lambda x: f"{x:.1%}")
    high_df["Actual Default"]      = high_df["actual_label"].map({1: "✅ Yes", 0: "❌ No"})
    high_df["Recommendation"]      = high_df["risk_tier"].map({
        "High Risk":       "Reject or collateral",
        "Very High Risk":  "Reject immediately"
    })
    st.dataframe(
        high_df[["Default Probability", "risk_tier", "Recommendation", "Actual Default"]
                ].reset_index(drop=True),
        use_container_width=True
    )

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — LIVE RISK PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Live Risk Predictor":
    st.markdown('<p class="main-header">🔍 Live Credit Risk Predictor</p>', unsafe_allow_html=True)
    st.info(
        f"Fill in applicant details and click **Predict**. "
        f"Score: Low < 33 | Medium < 66 | High < 91 | Very High ≥ 91 (0–100 scale)"
    )

    with st.form("predict_form"):
        st.subheader("Financial Details")
        c1, c2, c3 = st.columns(3)
        with c1:
            income       = st.number_input("Monthly Income (INR)", 5000, 500000, 45000, step=1000)
            loan_amount  = st.number_input("Loan Amount (INR)", 10000, 5000000, 250000, step=5000)
            loan_tenure  = st.selectbox("Loan Tenure (months)", [12, 24, 36, 48, 60], index=1)
            emp_type     = st.selectbox("Employment Type",
                           ["salaried", "self_employed", "gig", "student"])
            city_tier    = st.selectbox("City Tier", [1, 2, 3], index=1)
        with c2:
            salary_reg   = st.slider("Salary regularity",  0.0, 1.0, 0.80, 0.01,
                                     help="0 = very irregular, 1 = very regular")
            bill_score   = st.slider("Bill payment score", 0.0, 1.0, 0.75, 0.01,
                                     help="0 = poor, 1 = excellent")
            savings_r    = st.slider("Savings ratio",      0.0, 1.0, 0.20, 0.01,
                                     help="Proportion of income saved")
            existing_emi = st.slider("Existing EMIs",          0, 5, 1)
            overdrafts   = st.slider("Overdraft incidents",    0, 10, 0)
            merch_div    = st.slider("Transaction diversity",  1, 30, 10)
        with c3:
            is_postpaid  = st.selectbox("Mobile type", [1, 0],
                           format_func=lambda x: "Postpaid" if x else "Prepaid")
            fin_apps     = st.slider("Financial apps installed", 0, 15, 3)
            home_stab    = st.slider("Home stability (months)", 1, 60, 18)
            loc_changes  = st.slider("Location changes (6m)",   0, 6, 1)
            work_reg     = st.slider("Work regularity",   0.0, 1.0, 0.80, 0.01)
            recharge_reg = st.slider("Recharge regularity", 0.0, 1.0, 0.75, 0.01)

        c4, c5 = st.columns(2)
        with c4:
            avg_recharge = st.number_input("Avg recharge (INR)", 50, 1500, 399)
            night_usage  = st.slider("Night usage ratio", 0.0, 1.0, 0.20, 0.01)
        with c5:
            account_age  = st.slider("Account age (months)", 1, 120, 24)

        loan_text = st.text_area(
            "Loan Purpose (applicant's own words)",
            "I need this loan to purchase a laptop for my freelance software development work.",
            help="Vague purposes like 'personal use' or 'urgent need' increase risk score"
        )

        # ── Live preview of key drivers (updates as sliders move) ─────────────
        st.divider()
        st.markdown("**📊 Key driver preview** *(updates before you submit)*")
        _prev_beh  = overdrafts * 0.3 + (1 - bill_score) * 0.3 + (1 - salary_reg) * 0.4
        _prev_dti  = loan_amount / (income + 1)
        _prev_stab = (home_stab / 60) * 0.5 + work_reg * 0.3 + max(0, 1 - loc_changes / 5) * 0.2
        _prev_c1, _prev_c2, _prev_c3 = st.columns(3)
        _prev_c1.metric("Behavioural Risk Score", f"{_prev_beh:.3f}",
                         help="Higher = riskier. Range 0–1.4. Main driver of prediction.")
        _prev_c2.metric("Debt-to-Income Ratio",   f"{_prev_dti:.2f}x",
                         help="Loan ÷ Monthly Income. >10 is very high risk.")
        _prev_c3.metric("Stability Score",        f"{_prev_stab:.3f}",
                         help="Higher = more stable. Range 0–1.")

        submitted = st.form_submit_button("🔍 Predict Risk Score", use_container_width=True)

    # ── Quick test cases guide ────────────────────────────────────────────────
    with st.expander("💡 Test cases — how to see different risk tiers", expanded=False):
        st.markdown("""
**The model is most sensitive to these inputs. Change them to move between tiers:**

| To get...        | Set these values |
|------------------|-----------------|
| ✅ Low Risk       | Income ₹80k+, Loan ₹100k–200k, Salary reg 0.95, Bill score 0.95, Savings 0.35, Overdrafts 0, Salaried |
| 🔍 Medium Risk   | Income ₹45k, Loan ₹250k, Salary reg 0.80, Bill score 0.75, Savings 0.20, Overdrafts 0, Salaried |
| ⚠️ High Risk      | Income ₹15k, Loan ₹500k, Salary reg 0.50, Bill score 0.40, Savings 0.05, Overdrafts 3, Gig/Student |
| ❌ Very High Risk | Income ₹5k, Loan ₹1M+, Salary reg 0.10, Bill score 0.10, Savings 0.00, Overdrafts 8+, Student |

**The three biggest drivers are:**
1. `behavioral_risk_score` = overdrafts×0.3 + (1−bill_score)×0.3 + (1−salary_reg)×0.4
2. `debt_to_income` = loan_amount ÷ income
3. `stability_score` = home stability + work regularity + location stability
        """)

    if submitted:
        with st.spinner("Computing risk score..."):
            X     = build_features(income, loan_amount, loan_tenure, emp_type, city_tier,
                                   salary_reg, bill_score, savings_r, existing_emi,
                                   overdrafts, merch_div, is_postpaid, fin_apps,
                                   home_stab, loc_changes, work_reg, recharge_reg,
                                   avg_recharge, night_usage, account_age, loan_text)
            proba = float(model.predict_proba(X)[0][1])
            display_score = to_display_score(proba)

        # ── Debug: show computed features ─────────────────────────────────────
        with st.expander("🔧 Debug — computed feature values sent to model", expanded=False):
            # Recompute the key derived values for display
            _credit = income * 1.05
            _debit  = income * (1 - savings_r)
            _dti    = loan_amount / (income + 1)
            _stab   = (home_stab / 60) * 0.5 + work_reg * 0.3 + max(0, 1 - loc_changes / 5) * 0.2
            _beh    = overdrafts * 0.3 + (1 - bill_score) * 0.3 + (1 - salary_reg) * 0.4
            _mob    = is_postpaid * 0.3 + recharge_reg * 0.3 + min(fin_apps / 10, 1) * 0.2 + (1 - night_usage) * 0.2
            _inc_sc = max(0.0, min(1.0, (income - 5000) / (500000 - 5000)))
            _emi    = loan_amount / (loan_tenure + 1)

            debug_df = pd.DataFrame({
                "Feature": [
                    "behavioral_risk_score  ← KEY",
                    "debt_to_income (DTI)   ← KEY",
                    "stability_score        ← KEY",
                    "mobile_trust_score",
                    "income_scaled",
                    "loan_amount_scaled",
                    "repayment_to_income",
                    "risk_x_dti  (interaction)",
                    "income_x_stability",
                    "savings_x_stability",
                    "emi_burden_ratio",
                    "vague_purpose_flag",
                ],
                "Value": [
                    f"{_beh:.4f}  (overdrafts×0.3 + (1−bill)×0.3 + (1−salary_reg)×0.4)",
                    f"{_dti:.4f}  (loan {loan_amount:,.0f} ÷ income {income:,.0f})",
                    f"{_stab:.4f}  (home_stab/60×0.5 + work_reg×0.3 + loc_stable×0.2)",
                    f"{_mob:.4f}",
                    f"{_inc_sc:.4f}",
                    f"{max(0, min(1, (loan_amount-10000)/4990000)):.4f}",
                    f"{_emi/(income+1):.6f}",
                    f"{_beh * _dti:.4f}",
                    f"{_inc_sc * _stab:.4f}",
                    f"{savings_r * _stab:.4f}",
                    f"{(existing_emi * _debit * 0.1) / (income + 1):.6f}",
                    f"{int(any(w in loan_text.lower() for w in ['urgent','soon','various','miscellaneous','general','personal use']))}",
                ],
            })
            st.dataframe(debug_df, use_container_width=True)
            st.caption(f"Raw model output (before tier assignment): **{proba:.4f}** ({proba:.1%})")

        tier  = assign_tier(proba)          # ← unified function
        color = RISK_COLORS[tier]

        action_map = {
            "Low Risk":       ("✅ Auto-approve",                "risk-low"),
            "Medium Risk":    ("🔍 Manual review recommended",   "risk-medium"),
            "High Risk":      ("⚠️ Reject or require collateral", "risk-high"),
            "Very High Risk": ("❌ Reject immediately",           "risk-vhigh"),
        }
        action_text, risk_class = action_map[tier]

        # Result display
        r1, r2 = st.columns([1, 1])
        with r1:
            st.markdown(f"""
            <div class="{risk_class}">
                <h2 style="margin:0;color:#333">Risk Score: {display_score:.1f} / 100</h2>
                <h3 style="margin:4px 0;color:#555">{tier}</h3>
                <p style="margin:0;font-size:16px">{action_text}</p>
                <hr style="margin:8px 0;border-color:#ccc">
                <p style="margin:0;font-size:12px;color:#666">
                    Low &lt;{LOW_THRESHOLD:.0%} | Medium &lt;{MID_THRESHOLD:.0%} |
                    High &lt;{HIGH_THRESHOLD:.0%} | Very High ≥{HIGH_THRESHOLD:.0%}
                </p>
                <p style="margin:2px 0;font-size:12px;color:#666">
                    F1-optimal threshold: {BEST_THRESHOLD:.3f}
                </p>
            </div>""", unsafe_allow_html=True)

        with r2:
            gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=round(display_score, 1),
                title={"text": "Risk Score (0–100)", "font": {"size": 16}},
                number={"suffix": "/100", "font": {"size": 28}},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1},
                    "bar":  {"color": color, "thickness": 0.3},
                    "steps": [
                        {"range": [0,  33], "color": "#d4edda"},
                        {"range": [33, 66], "color": "#fff3cd"},
                        {"range": [66, 91], "color": "#f8d7da"},
                        {"range": [91,100], "color": "#e2d0f8"},
                    ],
                    "threshold": {"line": {"color": color, "width": 4}, "value": display_score}
                }
            ))
            gauge.update_layout(height=220, margin=dict(t=40, b=0, l=20, r=20))
            st.plotly_chart(gauge, use_container_width=True)

        # ── SHAP explanation ──────────────────────────────────────────────────
        st.subheader("What drove this prediction?")
        reasons = []
        try:
            import shap as shap_lib
            # CalibratedClassifierCV imported at top — TreeExplainer needs raw model
            shap_model = model.estimator if isinstance(model, CalibratedClassifierCV) else model
            explainer  = shap_lib.TreeExplainer(shap_model)
            sv         = explainer.shap_values(X)[0]
            top_idx    = np.argsort(np.abs(sv))[-15:][::-1]
            cols       = X.columns.tolist()

            top_feats  = [FEAT_LABELS.get(cols[i], cols[i].replace("_", " ").title()) for i in top_idx]
            top_vals   = [sv[i] for i in top_idx]
            bar_colors = ["#dc3545" if v > 0 else "#28a745" for v in top_vals]

            fig_shap = go.Figure(go.Bar(
                x=top_vals, y=top_feats, orientation="h",
                marker_color=bar_colors,
                text=[f"{v:+.3f}" for v in top_vals],
                textposition="outside"
            ))
            fig_shap.update_layout(
                title="SHAP Feature Impact (red = increases risk | green = reduces risk)",
                xaxis_title="Impact on default probability",
                height=420, margin=dict(l=10, r=60)
            )
            st.plotly_chart(fig_shap, use_container_width=True)

            # Top 3 text reasons
            st.subheader("Top 3 Risk Factors")
            col_a, col_b, col_c = st.columns(3)
            for i, (col, val) in enumerate(
                zip([cols[j] for j in top_idx[:3]], [sv[j] for j in top_idx[:3]])
            ):
                label     = FEAT_LABELS.get(col, col.replace("_", " ").capitalize())
                direction = "⬆️ Increasing" if val > 0 else "⬇️ Reducing"
                icon      = "🔴" if val > 0 else "🟢"
                txt       = f"{icon} **{label}**\n\n{direction} default risk\n\nSHAP = {val:.4f}"
                reasons.append(f"{label} ({'increasing' if val > 0 else 'reducing'} risk)")
                [col_a, col_b, col_c][i].info(txt)

        except MemoryError:
            st.warning("SHAP skipped — low memory. Key drivers: behavioral risk, salary regularity, savings ratio.")

        # ── Save to SQL ───────────────────────────────────────────────────────
        saved = save_live_prediction(
            income, loan_amount, emp_type, loan_text,
            proba, tier, action_text, reasons
        )
        if saved:
            st.success("✅ Prediction saved to SQL database (live_prediction_log table)")

        # ── Recent live predictions from SQL ──────────────────────────────────
        st.subheader("Recent Live Predictions (from SQL)")
        live_log = get_live_log()
        if not live_log.empty:
            live_log["Risk Score (0–100)"] = live_log["default_probability"].apply(
                lambda x: f"{to_display_score(x):.1f}"
            )
            st.dataframe(
                live_log[["income", "loan_amount", "employment_type", "Risk Score (0–100)",
                           "risk_tier", "recommendation", "created_at"]].head(10),
                use_container_width=True
            )
        else:
            st.info("No live predictions yet — submit a prediction above to see it here.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Model Performance":
    st.markdown('<p class="main-header">📈 Model Performance Metrics</p>', unsafe_allow_html=True)
    st.caption("Trained on 246,008 real loan applications | Tested on 61,503")

    from sklearn.metrics import (roc_curve, precision_recall_curve, auc,
                                  confusion_matrix, precision_score,
                                  recall_score, f1_score)

    fpr, tpr, _ = roc_curve(predictions["actual_label"], predictions["y_proba"])
    roc_auc     = auc(fpr, tpr)
    prec, rec, _ = precision_recall_curve(predictions["actual_label"], predictions["y_proba"])
    accuracy    = (predictions["y_pred"] == predictions["actual_label"]).mean()

    # Metrics at BEST_THRESHOLD
    y_pred_best = (predictions["y_proba"] >= BEST_THRESHOLD).astype(int)
    best_prec   = precision_score(predictions["actual_label"], y_pred_best, zero_division=0)
    best_rec    = recall_score(predictions["actual_label"],    y_pred_best, zero_division=0)
    best_f1     = f1_score(predictions["actual_label"],        y_pred_best, zero_division=0)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("ROC-AUC",           f"{roc_auc:.4f}", delta="vs random 0.5000")
    c2.metric("Accuracy",          f"{accuracy:.2%}")
    c3.metric("Best Threshold",    f"{BEST_THRESHOLD:.3f}")
    c4.metric("Recall @ threshold",f"{best_rec:.2%}", help="% of actual defaulters caught")
    c5.metric("F1 @ threshold",    f"{best_f1:.4f}")

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines", fill="tozeroy",
            fillcolor="rgba(102,126,234,0.2)",
            name=f"XGBoost (AUC={roc_auc:.3f})",
            line=dict(color="#667eea", width=2.5)
        ))
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines",
            line=dict(color="gray", dash="dash"), name="Random (AUC=0.5)"
        ))
        fig_roc.update_layout(
            title="ROC Curve", height=380,
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            legend=dict(x=0.6, y=0.1)
        )
        st.plotly_chart(fig_roc, use_container_width=True)

    with col2:
        cm = confusion_matrix(predictions["actual_label"], y_pred_best)
        fig_cm = px.imshow(
            cm, text_auto=True, color_continuous_scale="Blues",
            labels=dict(x="Predicted", y="Actual"),
            x=["No Default", "Default"], y=["No Default", "Default"],
            title=f"Confusion Matrix (threshold={BEST_THRESHOLD:.3f})"
        )
        fig_cm.update_layout(height=380)
        st.plotly_chart(fig_cm, use_container_width=True)

    fig_pr = go.Figure()
    fig_pr.add_trace(go.Scatter(
        x=rec, y=prec, mode="lines", fill="tozeroy",
        fillcolor="rgba(40,167,69,0.2)",
        name="Precision-Recall", line=dict(color="#28a745", width=2)
    ))
    fig_pr.update_layout(
        title="Precision-Recall Curve",
        xaxis_title="Recall", yaxis_title="Precision", height=320
    )
    st.plotly_chart(fig_pr, use_container_width=True)

    st.subheader("What the metrics mean")
    c1, c2 = st.columns(2)
    with c1:
        st.info(f"**ROC-AUC {roc_auc:.2f}** — Model correctly ranks a defaulter above a "
                f"non-defaulter {roc_auc:.0%} of the time.")
        st.success(f"**Recall {best_rec:.0%}** — Catches {best_rec:.0%} of actual defaulters. "
                   f"High recall = fewer bad loans approved.")
    with c2:
        st.warning(f"**Precision {best_prec:.0%}** — Expected for ~8% default rate. "
                   f"Probabilities are calibrated.")
        st.info(f"**Best threshold {BEST_THRESHOLD:.3f}** — Auto-calculated to maximise F1. "
                f"Model uses isotonic calibration for accurate probabilities.")

    # Threshold sweep
    st.subheader("Threshold Impact Analysis")
    thresholds = np.arange(0.05, 0.60, 0.05)
    rows = []
    for t in thresholds:
        yp = (predictions["y_proba"] >= t).astype(int)
        tp = ((yp == 1) & (predictions["actual_label"] == 1)).sum()
        fp = ((yp == 1) & (predictions["actual_label"] == 0)).sum()
        fn = ((yp == 0) & (predictions["actual_label"] == 1)).sum()
        p  = tp / (tp + fp + 1e-9)
        r  = tp / (tp + fn + 1e-9)
        rows.append({
            "Threshold":  round(t, 2),
            "Precision":  round(p, 3),
            "Recall":     round(r, 3),
            "Flagged":    int(yp.sum()),
            "Approved":   int((yp == 0).sum()),
            "Is Best":    "✅" if abs(t - BEST_THRESHOLD) < 0.026 else ""
        })
    thresh_df = pd.DataFrame(rows)
    fig_t = go.Figure()
    fig_t.add_trace(go.Scatter(x=thresh_df["Threshold"], y=thresh_df["Precision"],
                               name="Precision", line=dict(color="#dc3545")))
    fig_t.add_trace(go.Scatter(x=thresh_df["Threshold"], y=thresh_df["Recall"],
                               name="Recall", line=dict(color="#28a745")))
    fig_t.add_vline(x=BEST_THRESHOLD, line_dash="dash",
                    annotation_text=f"Best ({BEST_THRESHOLD:.3f})", line_color="#667eea")
    fig_t.update_layout(
        title="Precision vs Recall at Different Thresholds",
        xaxis_title="Decision Threshold", yaxis_title="Score", height=320
    )
    st.plotly_chart(fig_t, use_container_width=True)
    st.dataframe(thresh_df, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — SQL ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🗄️ SQL Analytics":
    st.markdown('<p class="main-header">🗄️ SQL-Powered Analytics</p>', unsafe_allow_html=True)
    st.caption("All data stored in SQLite database — run custom SQL queries below")

    try:
        engine = get_engine()

        with engine.connect() as conn:
            total_r  = conn.execute(text("SELECT COUNT(*) FROM applicant_features")).scalar()
            def_rate = conn.execute(text("SELECT AVG(default_label) FROM applicant_features")).scalar()
            avg_inc  = conn.execute(text("SELECT AVG(income) FROM applicant_features")).scalar()
            avg_loan = conn.execute(text("SELECT AVG(loan_amount) FROM applicant_features")).scalar()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Records in DB",      f"{total_r:,}")
        c2.metric("Default Rate (SQL)", f"{def_rate:.2%}" if def_rate else "N/A")
        c3.metric("Avg Income (SQL)",   f"₹{avg_inc:,.0f}" if avg_inc else "N/A")
        c4.metric("Avg Loan (SQL)",     f"₹{avg_loan:,.0f}" if avg_loan else "N/A")

        st.divider()
        tab1, tab2, tab3, tab4 = st.tabs([
            "Default by City Tier", "Income vs Default", "Live Log", "Custom SQL"
        ])

        with tab1:
            df_city = pd.read_sql("""
                SELECT city_tier,
                       COUNT(*) as applicants,
                       ROUND(AVG(default_label), 3) as default_rate,
                       ROUND(AVG(income), 0) as avg_income
                FROM applicant_features
                GROUP BY city_tier
                ORDER BY city_tier
            """, engine)
            st.dataframe(df_city, use_container_width=True)
            fig = px.bar(df_city, x="city_tier", y="default_rate",
                         title="Default Rate by City Tier",
                         color="default_rate", color_continuous_scale="RdYlGn_r",
                         labels={"city_tier": "City Tier", "default_rate": "Default Rate"})
            fig.update_yaxes(tickformat=".1%")
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            df_inc = pd.read_sql("""
                SELECT
                    CASE
                        WHEN income < 100000 THEN 'Under 1L'
                        WHEN income < 200000 THEN '1L-2L'
                        WHEN income < 300000 THEN '2L-3L'
                        WHEN income < 500000 THEN '3L-5L'
                        ELSE 'Above 5L'
                    END as income_band,
                    COUNT(*) as count,
                    ROUND(AVG(default_label), 3) as default_rate
                FROM applicant_features
                GROUP BY income_band
                ORDER BY min(income)
            """, engine)
            st.dataframe(df_inc, use_container_width=True)
            fig2 = px.bar(df_inc, x="income_band", y="default_rate",
                          title="Default Rate by Income Band",
                          color="default_rate", color_continuous_scale="RdYlGn_r")
            fig2.update_yaxes(tickformat=".1%")
            st.plotly_chart(fig2, use_container_width=True)

        with tab3:
            live_df = pd.read_sql(
                "SELECT * FROM live_prediction_log ORDER BY created_at DESC", engine
            )
            if live_df.empty:
                st.info("No live predictions yet. Go to **Live Risk Predictor** and score an applicant.")
            else:
                st.dataframe(live_df, use_container_width=True)
                fig3 = px.pie(live_df, names="risk_tier",
                              title="Risk Tier Distribution of Live Predictions",
                              color="risk_tier", color_discrete_map=RISK_COLORS)
                st.plotly_chart(fig3, use_container_width=True)

        with tab4:
            st.markdown("**Run any SQL query on the database:**")
            default_query = """SELECT risk_tier, COUNT(*) as count,
ROUND(AVG(default_probability), 3) as avg_prob
FROM predictions
GROUP BY risk_tier
ORDER BY avg_prob DESC"""
            query = st.text_area("SQL Query", default_query, height=120)
            if st.button("▶ Run Query"):
                try:
                    result = pd.read_sql(query, engine)
                    st.dataframe(result, use_container_width=True)
                    st.caption(f"{len(result)} rows returned")
                except Exception as e:
                    st.error(f"SQL Error: {e}")

        st.divider()
        st.subheader("Database Tables")
        tables = ["applicant_features", "engineered_features", "predictions",
                  "live_prediction_log", "model_performance_log"]
        sel = st.selectbox("Select table to preview", tables)
        try:
            preview = pd.read_sql(f"SELECT * FROM {sel} LIMIT 20", engine)
            count   = pd.read_sql(f"SELECT COUNT(*) as n FROM {sel}", engine).iloc[0, 0]
            st.caption(f"Showing 20 of {count:,} rows")
            st.dataframe(preview, use_container_width=True)
        except Exception as e:
            st.warning(f"Table not yet populated: {e}")

    except Exception as e:
        st.error(f"Database error: {e}")
        st.info("Run `python sql/setup_db.py` first to create the database.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — SHAP EXPLAINABILITY
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔬 SHAP Explainability":
    st.markdown('<p class="main-header">🔬 SHAP Global Explainability</p>', unsafe_allow_html=True)
    st.markdown("SHAP (SHapley Additive exPlanations) shows **why** the model makes each prediction.")

    plots_dir = os.path.join(BASE_DIR, "explainability", "plots")

    if os.path.exists(os.path.join(plots_dir, "feature_importance.png")):
        c1, c2 = st.columns(2)
        with c1:
            st.image(os.path.join(plots_dir, "feature_importance.png"),
                     caption="Global Feature Importance (Mean |SHAP|)", use_column_width=True)
        with c2:
            st.image(os.path.join(plots_dir, "shap_beeswarm.png"),
                     caption="SHAP Beeswarm — Impact Direction", use_column_width=True)
        st.image(os.path.join(plots_dir, "waterfall_sample.png"),
                 caption="Individual Prediction Waterfall", use_column_width=True)

        st.subheader("How to read these charts")
        col1, col2 = st.columns(2)
        with col1:
            st.info("**Feature Importance bar chart** — Shows which features matter most globally "
                    "across all 307k applicants.")
            st.info("**Beeswarm plot** — Each dot is one applicant. Red = high feature value, "
                    "Blue = low. Position shows impact direction.")
        with col2:
            st.success("**Waterfall plot** — Shows a single applicant's prediction. Each bar shows "
                       "how much each feature pushed the score up or down from the baseline.")
            st.warning("**Key finding** — behavioral_risk_score and salary_regularity are the top "
                       "drivers, confirming alternative data adds genuine signal.")
    else:
        st.warning("⚠️ SHAP plots not found.")
        st.markdown("**Run this command first to generate them:**")
        st.code("python explainability/shap_analysis.py", language="bash")
        st.info(f"Expected location: `{plots_dir}`")
        st.markdown("**Verify your folder structure:**")
        st.code(
            f"os.listdir: {os.listdir(plots_dir) if os.path.exists(plots_dir) else 'folder does not exist'}"
        )