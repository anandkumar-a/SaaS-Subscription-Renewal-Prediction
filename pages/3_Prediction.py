import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Customer Renewal Prediction", layout="wide")

# ---------- GLASS UI STYLE ----------
st.markdown("""
<style>

.main-title {
    font-size:30px;
    font-weight:700;
}

.glass-card {
    background: rgba(255,255,255,0.06);
    padding:20px;
    border-radius:15px;
    backdrop-filter: blur(10px);
    margin-bottom:15px;
}

.result-box {
    padding:20px;
    border-radius:12px;
    text-align:center;
    font-size:22px;
    font-weight:bold;
}

</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🔮 Customer Renewal Prediction</div>', unsafe_allow_html=True)
st.caption("AI-powered SaaS subscription renewal prediction")

# -------- LOAD DATA + TRAIN MODEL --------

@st.cache_resource
def load_model():

    BASE_DIR = os.path.dirname(os.path.dirname(__file__))

    accounts = pd.read_csv(os.path.join(BASE_DIR, "ravenstack_accounts.csv"))
    subscriptions = pd.read_csv(os.path.join(BASE_DIR, "ravenstack_subscriptions.csv"))
    usage = pd.read_csv(os.path.join(BASE_DIR, "ravenstack_feature_usage.csv"))
    tickets = pd.read_csv(os.path.join(BASE_DIR, "ravenstack_support_tickets.csv"))
    churn = pd.read_csv(os.path.join(BASE_DIR, "ravenstack_churn_events.csv"))

    usage_features = usage.groupby("subscription_id").agg({
        "usage_count": "sum",
        "usage_duration_secs": "sum",
        "error_count": "sum"
    }).reset_index()

    ticket_features = tickets.groupby("account_id").agg({
        "resolution_time_hours": "mean",
        "first_response_time_minutes": "mean",
        "escalation_flag": "sum"
    }).reset_index()

    data = subscriptions.merge(accounts, on="account_id", how="left")
    data = data.merge(usage_features, on="subscription_id", how="left")
    data = data.merge(ticket_features, on="account_id", how="left")
    data = data.merge(churn, on="account_id", how="left")

    data["churn_flag"] = data["churn_date"].notna().astype(int)

    possible_features = [
        "mrr_amount",
        "usage_count",
        "usage_duration_secs",
        "error_count",
        "resolution_time_hours",
        "first_response_time_minutes",
        "escalation_flag"
    ]

    features = [col for col in possible_features if col in data.columns]

    X = data[features]
    y = data["churn_flag"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_scaled, y)

    return model, scaler, features

model, scaler, features = load_model()

# -------- INPUT UI ----------
st.markdown("## 🧾 Customer Input")

st.markdown('<div class="glass-card">', unsafe_allow_html=True)

inputs = {}

# 2 column layout
col1, col2 = st.columns(2)

for i, feature in enumerate(features):
    if i % 2 == 0:
        with col1:
            inputs[feature] = st.number_input(
                feature.replace("_", " ").title(),
                value=1.0
            )
    else:
        with col2:
            inputs[feature] = st.number_input(
                feature.replace("_", " ").title(),
                value=1.0
            )

st.markdown("</div>", unsafe_allow_html=True)

# -------- PREDICTION ----------
st.markdown("###")

if st.button("🚀 Predict Renewal Status", use_container_width=True):

    input_array = np.array([[inputs[f] for f in features]])
    input_scaled = scaler.transform(input_array)

    prob = model.predict_proba(input_scaled)[0][1]

    st.markdown("## 🎯 Prediction Result")

    # probability gauge
    st.progress(float(prob))

    if prob < 0.3:
        st.markdown(
            '<div class="result-box" style="background:#1f7a1f;">✅ Likely Renew</div>',
            unsafe_allow_html=True
        )

    elif prob < 0.7:
        st.markdown(
            '<div class="result-box" style="background:#c48f00;">⚠️ Customer At Risk</div>',
            unsafe_allow_html=True
        )

    else:
        st.markdown(
            '<div class="result-box" style="background:#b30000;">❌ Likely Churn</div>',
            unsafe_allow_html=True
        )

    st.write("### Renewal Probability:", round(prob, 2))

    # business recommendation
    st.markdown("### 💡 Suggested Action")

    if prob < 0.3:
        st.success("Customer is stable. Consider upsell opportunities.")
    elif prob < 0.7:
        st.warning("Offer retention discount or proactive support.")
    else:
        st.error("Immediate intervention required — high churn risk.")

