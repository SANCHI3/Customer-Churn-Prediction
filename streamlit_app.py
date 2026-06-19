import streamlit as st
import pandas as pd
import joblib

# ─────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Churn Predictor | ML Demo",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────
# CUSTOM CSS  – clean SaaS look
# ─────────────────────────────────────────
st.markdown("""
<style>
  /* Page background */
  .stApp { background-color: #F8F9FB; }

  /* Section card */
  .card {
    background: white;
    border-radius: 12px;
    padding: 1.5rem 2rem;
    margin-bottom: 1.2rem;
    border: 1px solid #E8ECF0;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
  }

  /* Section heading inside card */
  .card-title {
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #6B7280;
    margin-bottom: 1rem;
  }

  /* Result probability number */
  .prob-number {
    font-size: 3.5rem;
    font-weight: 700;
    line-height: 1;
  }

  /* Risk badge */
  .badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 600;
    margin-top: 0.5rem;
  }
  .badge-red    { background:#FEE2E2; color:#B91C1C; }
  .badge-amber  { background:#FEF3C7; color:#92400E; }
  .badge-green  { background:#D1FAE5; color:#065F46; }

  /* Action box */
  .action-box {
    border-radius: 8px;
    padding: 0.9rem 1.2rem;
    font-size: 0.9rem;
    font-weight: 500;
    margin-top: 1rem;
  }
  .action-red   { background:#FEF2F2; border-left: 4px solid #EF4444; color:#7F1D1D; }
  .action-amber { background:#FFFBEB; border-left: 4px solid #F59E0B; color:#78350F; }
  .action-green { background:#F0FDF4; border-left: 4px solid #22C55E; color:#14532D; }

  /* Progress bar override */
  .stProgress > div > div { border-radius: 10px; }

  /* Hide Streamlit default header */
  header[data-testid="stHeader"] { background: transparent; }

  /* Metric cards row */
  .metric-row {
    display: flex;
    gap: 12px;
    margin-top: 0.5rem;
  }
  .mini-metric {
    flex: 1;
    background: #F3F4F6;
    border-radius: 8px;
    padding: 0.75rem 1rem;
    text-align: center;
  }
  .mini-metric .label { font-size:0.7rem; color:#6B7280; text-transform:uppercase; }
  .mini-metric .value { font-size:1.1rem; font-weight:700; color:#111827; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────
@st.cache_resource
def load_pipeline():
    return joblib.load("churn_pipeline.pkl")

pipeline = load_pipeline()
model         = pipeline["model"]
scaler        = pipeline["scaler"]
feature_names = pipeline["feature_names"]

# ─────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────
st.markdown("""
<div style="padding:1.5rem 0 0.5rem 0;">
  <h1 style="font-size:1.8rem; font-weight:700; color:#111827; margin:0;">
    📊 Customer Churn Predictor
  </h1>
  <p style="color:#6B7280; margin:0.3rem 0 0 0; font-size:0.95rem;">
    Real-time churn risk scoring powered by XGBoost · Telco dataset · Accuracy ~82%
  </p>
</div>
<hr style="border:none; border-top:1px solid #E5E7EB; margin:1rem 0 1.5rem 0;">
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# TWO-COLUMN LAYOUT
# ─────────────────────────────────────────
col_form, col_result = st.columns([1.1, 0.9], gap="large")

# ══════════════════════════════════════════
# LEFT — INPUT FORM
# ══════════════════════════════════════════
with col_form:

    # — Account Info —
    st.markdown('<div class="card"><div class="card-title">Account Information</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    tenure   = c1.number_input("Tenure (months)", 0, 72, 12)
    contract = c2.selectbox("Contract Type",
                            ["Month-to-month", "One year", "Two year"])
    c3, c4 = st.columns(2)
    monthly = c3.number_input("Monthly Charges ($)", 0.0, 200.0, 70.0)
    total   = c4.number_input("Total Charges ($)", 0.0, 10000.0, 500.0)
    payment = st.selectbox("Payment Method",
                           ["Bank transfer (automatic)",
                            "Credit card (automatic)",
                            "Electronic check",
                            "Mailed check"])
    st.markdown('</div>', unsafe_allow_html=True)

    # — Demographics —
    st.markdown('<div class="card"><div class="card-title">Demographics</div>', unsafe_allow_html=True)
    c5, c6, c7, c8 = st.columns(4)
    gender     = c5.selectbox("Gender",         ["Female", "Male"])
    senior     = c6.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x else "No")
    partner    = c7.selectbox("Partner",        ["No", "Yes"])
    dependents = c8.selectbox("Dependents",     ["No", "Yes"])
    st.markdown('</div>', unsafe_allow_html=True)

    # — Services —
    st.markdown('<div class="card"><div class="card-title">Services Subscribed</div>', unsafe_allow_html=True)
    c9, c10 = st.columns(2)
    phone    = c9.selectbox("Phone Service",   ["No", "Yes"])
    multiple = c10.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    c11, c12 = st.columns(2)
    internet  = c11.selectbox("Internet Service",  ["DSL", "Fiber optic", "No"])
    paperless = c12.selectbox("Paperless Billing", ["No", "Yes"])

    c13, c14, c15 = st.columns(3)
    online_security   = c13.selectbox("Online Security",   ["No", "Yes", "No internet service"])
    online_backup     = c14.selectbox("Online Backup",     ["No", "Yes", "No internet service"])
    device_protection = c15.selectbox("Device Protection", ["No", "Yes", "No internet service"])

    c16, c17, c18 = st.columns(3)
    tech_support    = c16.selectbox("Tech Support",     ["No", "Yes", "No internet service"])
    streaming_tv    = c17.selectbox("Streaming TV",     ["No", "Yes", "No internet service"])
    streaming_movies = c18.selectbox("Streaming Movies",["No", "Yes", "No internet service"])
    st.markdown('</div>', unsafe_allow_html=True)

    # — Predict Button —
    predict_btn = st.button("🔍 Predict Churn Risk", use_container_width=True, type="primary")

# ══════════════════════════════════════════
# RIGHT — RESULTS PANEL
# ══════════════════════════════════════════
with col_result:

    if not predict_btn:
        # Placeholder state
        st.markdown("""
        <div class="card" style="text-align:center; padding:3rem 2rem; color:#9CA3AF;">
          <div style="font-size:3rem;">🎯</div>
          <div style="font-size:1rem; margin-top:0.5rem; font-weight:500;">
            Fill in customer details and click<br><strong>Predict Churn Risk</strong>
          </div>
          <div style="font-size:0.8rem; margin-top:1rem;">
            Results will appear here
          </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        # ── BUILD FEATURE ROW ──────────────────
        row = {col: 0 for col in feature_names}

        row["gender"]          = 1 if gender == "Male" else 0
        row["SeniorCitizen"]   = senior
        row["Partner"]         = 1 if partner == "Yes" else 0
        row["Dependents"]      = 1 if dependents == "Yes" else 0
        row["PhoneService"]    = 1 if phone == "Yes" else 0
        row["PaperlessBilling"]= 1 if paperless == "Yes" else 0
        row["tenure"]          = tenure
        row["MonthlyCharges"]  = monthly
        row["TotalCharges"]    = total
        row["avg_monthly_spend"] = total / (tenure + 1)
        row["num_services"]    = sum([
            online_security   == "Yes",
            online_backup     == "Yes",
            device_protection == "Yes",
            tech_support      == "Yes",
            streaming_tv      == "Yes",
            streaming_movies  == "Yes"
        ])

        # One-hot encoding helpers
        def ohe(prefix, value, base_values):
            for v in base_values:
                key = f"{prefix}_{v}"
                if key in row:
                    row[key] = 1 if value == v else 0

        ohe("MultipleLines",   multiple,          ["Yes", "No phone service"])
        ohe("InternetService", internet,           ["Fiber optic", "No"])
        ohe("OnlineSecurity",  online_security,    ["Yes", "No internet service"])
        ohe("OnlineBackup",    online_backup,      ["Yes", "No internet service"])
        ohe("DeviceProtection",device_protection,  ["Yes", "No internet service"])
        ohe("TechSupport",     tech_support,       ["Yes", "No internet service"])
        ohe("StreamingTV",     streaming_tv,       ["Yes", "No internet service"])
        ohe("StreamingMovies", streaming_movies,   ["Yes", "No internet service"])
        ohe("Contract",        contract,            ["One year", "Two year"])
        ohe("PaymentMethod",   payment,
            ["Credit card (automatic)", "Electronic check", "Mailed check"])

        # Tenure group
        tg = ("0-12" if tenure <= 12 else
              "13-24" if tenure <= 24 else
              "25-48" if tenure <= 48 else
              "49-60" if tenure <= 60 else "61-72")
        for g in ["13-24", "25-48", "49-60", "61-72"]:
            key = f"tenure_group_{g}"
            if key in row:
                row[key] = 1 if tg == g else 0

        # ── RUN MODEL ──────────────────────────
        try:
            X = pd.DataFrame([row])
            prob = model.predict_proba(X)[0][1]
            pct  = prob * 100

            # Risk tier
            if prob >= 0.7:
                tier        = "High Risk"
                badge_cls   = "badge-red"
                action_cls  = "action-red"
                action_icon = "🚨"
                action_text = "Urgent: Offer a contract upgrade discount immediately. This customer is very likely to leave."
                bar_color   = "#EF4444"
                prob_color  = "#B91C1C"
            elif prob >= 0.4:
                tier        = "Medium Risk"
                badge_cls   = "badge-amber"
                action_cls  = "action-amber"
                action_icon = "⚠️"
                action_text = "Send a personalised re-engagement offer and schedule a check-in call within 7 days."
                bar_color   = "#F59E0B"
                prob_color  = "#92400E"
            else:
                tier        = "Low Risk"
                badge_cls   = "badge-green"
                action_cls  = "action-green"
                action_icon = "✅"
                action_text = "Customer is stable. No immediate action needed. Continue standard engagement."
                bar_color   = "#22C55E"
                prob_color  = "#065F46"

            priority_score = round(prob * monthly, 1)

            # ── RESULT CARD ──────────────────────
            st.markdown(f"""
            <div class="card">
              <div class="card-title">Prediction Result</div>
              <div class="prob-number" style="color:{prob_color};">{pct:.1f}%</div>
              <div style="color:#6B7280; font-size:0.85rem; margin-top:2px;">Churn Probability</div>
              <span class="badge {badge_cls}">{tier}</span>
            </div>
            """, unsafe_allow_html=True)

            # Progress bar
            st.progress(int(pct))

            # Mini metrics
            st.markdown(f"""
            <div class="metric-row">
              <div class="mini-metric">
                <div class="label">Priority Score</div>
                <div class="value">{priority_score}</div>
              </div>
              <div class="mini-metric">
                <div class="label">Tenure</div>
                <div class="value">{tenure} mo</div>
              </div>
              <div class="mini-metric">
                <div class="label">Services</div>
                <div class="value">{row['num_services']}/6</div>
              </div>
              <div class="mini-metric">
                <div class="label">Monthly $</div>
                <div class="value">${monthly:.0f}</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            # Recommended action
            st.markdown(f"""
            <div class="action-box {action_cls}">
              {action_icon} <strong>Recommended Action</strong><br>
              <span style="font-weight:400;">{action_text}</span>
            </div>
            """, unsafe_allow_html=True)

            # Key risk drivers (rule-based, matches SHAP findings)
            drivers = []
            if contract == "Month-to-month":
                drivers.append("Month-to-month contract (highest churn driver)")
            if tenure <= 12:
                drivers.append("Low tenure — early churn window (0-12 months)")
            if internet == "Fiber optic" and tech_support == "No":
                drivers.append("Fiber optic with no tech support")
            if monthly > 70:
                drivers.append(f"High monthly charges (${monthly:.0f})")
            if payment == "Electronic check":
                drivers.append("Electronic check payment (correlates with disengagement)")
            if row["num_services"] <= 1:
                drivers.append("Low service adoption — low switching cost")

            if drivers:
                st.markdown('<div style="margin-top:1rem;">', unsafe_allow_html=True)
                st.markdown("**🔍 Key Risk Factors Detected:**")
                for d in drivers:
                    st.markdown(f"- {d}")
                st.markdown('</div>', unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Prediction error: {e}")
            st.info("Make sure `churn_pipeline.pkl` is in the same folder as `app.py`.")

# ─────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────
st.markdown("""
<hr style="border:none; border-top:1px solid #E5E7EB; margin:2rem 0 1rem 0;">
<div style="text-align:center; color:#9CA3AF; font-size:0.8rem;">
  Built with Python · XGBoost · SHAP · Streamlit &nbsp;|&nbsp;
  Telco Customer Churn Dataset (Kaggle) &nbsp;|&nbsp;
  <a href="https://github.com/YOUR_USERNAME/churn-predictor"
     style="color:#6B7280;" target="_blank">GitHub ↗</a>
</div>
""", unsafe_allow_html=True)
