

import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import os
import time

st.set_page_config(page_title="Dynamic Dataset Analysis", layout="wide", page_icon="⚡")

# -----------------------
# Page Header
# -----------------------
st.markdown("""
# ⚡ Dynamic Dataset Analysis
Upload a new dataset and automatically generate churn predictions.
""")

# -----------------------
# File uploader
# -----------------------
uploaded_file = st.file_uploader("Upload your CSV dataset", type=["csv"])
show_raw = st.checkbox("Show Raw Data", False)

# -----------------------
# Features used in trained model
# -----------------------
features = [
    "mrr_amount",
    "usage_count",
    "usage_duration_secs",
    "error_count",
    "resolution_time_hours",
    "first_response_time_minutes",
    "escalation_flag"
]

# -----------------------
# Column mapping for different dataset names
# -----------------------
column_mapping = {
    "seat_count": "seats",
    "mrr": "mrr_amount",
    "errors": "error_count",
    "first_response_time": "first_response_time_minutes",
    "resolution_time": "resolution_time_hours"
}

st.info("📌 Expected columns: " + ", ".join(features))

# -----------------------
# Load pretrained model & scaler
# -----------------------
@st.cache_data
def load_model_objects():
    if not os.path.exists("rf_model.joblib") or not os.path.exists("scaler.joblib"):
        st.error("⚠ Model files not found! Train and save model first.")
        return None, None

    model = joblib.load("rf_model.joblib")
    scaler = joblib.load("scaler.joblib")
    return model, scaler


model, scaler = load_model_objects()
if model is None or scaler is None:
    st.stop()

# -----------------------
# DATASET VALIDATION
# -----------------------
def validate_dataset(data):
    """
    Check whether uploaded dataset is relevant for churn prediction.
    """

    # Apply mapping temporarily
    mapped_cols = [column_mapping.get(col, col) for col in data.columns]

    # Check matching features
    matched_features = [f for f in features if f in mapped_cols]

    # No useful columns
    if len(matched_features) == 0:
        return False, "❌ Dataset is irrelevant. No required feature columns found."

    # Too few useful columns
    if len(matched_features) < 3:
        return False, f"⚠ Dataset has very limited useful columns: {matched_features}"

    # Check numeric data
    numeric_count = data.select_dtypes(include=["number"]).shape[1]
    if numeric_count == 0:
        return False, "❌ Dataset contains no numeric values required for prediction."

    return True, f"✅ Dataset valid. Using {len(matched_features)} feature(s)."

# -----------------------
# Prediction function
# -----------------------
def predict_new_data(new_data):

    # Rename columns based on mapping
    new_data = new_data.rename(columns=column_mapping)

    # Add missing columns with default 0
    for col in features:
        if col not in new_data.columns:
            new_data[col] = 0

    # Keep required features in correct order
    X_new = new_data[features].copy()

    # Fill missing values
    X_new = X_new.fillna(0)

    # Scale features
    X_new_scaled = scaler.transform(X_new)

    # Predict probabilities
    new_data["renewal_probability"] = model.predict_proba(X_new_scaled)[:, 1]

    # Segment classification
    def segment(p):
        if p < 0.3:
            return "Likely Renew"
        elif p < 0.7:
            return "At Risk"
        else:
            return "Likely Churn"

    new_data["segment"] = new_data["renewal_probability"].apply(segment)

    return new_data

# -----------------------
# MAIN FLOW
# -----------------------
if uploaded_file:

    new_data = pd.read_csv(uploaded_file)

    # Validate dataset
    is_valid, message = validate_dataset(new_data)

    if not is_valid:
        st.error(message)
        st.warning("⚠ Please upload a dataset with relevant churn features.")
        st.stop()
    else:
        st.success(message)

    predictions = predict_new_data(new_data)
    st.success("✅ Predictions generated successfully!")

    # -----------------------
    # Animated KPI Cards
    # -----------------------
    st.subheader("📊 Key Metrics")

    col1, col2, col3 = st.columns(3)

    total_customers = len(predictions)
    churn_customers = len(predictions[predictions.segment == "Likely Churn"])
    risk_customers = len(predictions[predictions.segment == "At Risk"])

    def animate_number(col, value, label):
        placeholder = col.empty()
        step = max(1, value // 40)
        for i in range(0, value + 1, step):
            placeholder.metric(label, i)
            time.sleep(0.01)
        placeholder.metric(label, value)

    animate_number(col1, total_customers, "Total Customers")
    animate_number(col2, churn_customers, "Likely Churn")
    animate_number(col3, risk_customers, "At Risk")

    # -----------------------
    # Segment Distribution
    # -----------------------
    st.subheader("👥 Customer Segment Distribution")

    segment_counts = predictions["segment"].value_counts().reset_index()
    segment_counts.columns = ["segment", "count"]

    fig_seg = px.bar(
        segment_counts,
        x="segment",
        y="count",
        color="segment",
        text="count",
        color_discrete_sequence=px.colors.qualitative.Set2,
        template="plotly_dark"
    )

    fig_seg.update_traces(textposition='outside')
    st.plotly_chart(fig_seg, use_container_width=True)

    # -----------------------
    # Probability Distribution
    # -----------------------
    st.subheader("📊 Renewal Probability Distribution")

    fig_prob = px.histogram(
        predictions,
        x="renewal_probability",
        nbins=30,
        color="segment",
        color_discrete_sequence=px.colors.qualitative.Set2,
        template="plotly_dark"
    )

    st.plotly_chart(fig_prob, use_container_width=True)

    # -----------------------
    # Download Predictions
    # -----------------------
    st.subheader("⬇ Export Results")

    csv = predictions.to_csv(index=False)
    st.download_button("Download Predictions CSV", csv, "predictions.csv")

    # -----------------------
    # Show Raw Data
    # -----------------------
    if show_raw:
        st.subheader("📄 Dataset with Predictions")
        st.dataframe(predictions)

else:
    st.info("📌 Upload a CSV file to perform dynamic analysis.")