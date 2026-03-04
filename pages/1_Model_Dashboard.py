import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

st.title("📊 SaaS Churn Analytics")

# ----------------------------
# SIDEBAR CONTROLS
# ----------------------------

st.sidebar.header("Dashboard Controls")

show_raw = st.sidebar.toggle("Show Raw Data", False)

# ----------------------------
# LOAD DATA
# ----------------------------

@st.cache_data
def load_data():
    accounts = pd.read_csv("ravenstack_accounts.csv")
    subscriptions = pd.read_csv("ravenstack_subscriptions.csv")
    usage = pd.read_csv("ravenstack_feature_usage.csv")
    tickets = pd.read_csv("ravenstack_support_tickets.csv")
    churn = pd.read_csv("ravenstack_churn_events.csv")
    return accounts, subscriptions, usage, tickets, churn

accounts, subscriptions, usage, tickets, churn = load_data()

# ----------------------------
# FEATURE ENGINEERING
# ----------------------------

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

data = subscriptions.merge(accounts,on="account_id",how="left")
data = data.merge(usage_features,on="subscription_id",how="left")
data = data.merge(ticket_features,on="account_id",how="left")
data = data.merge(churn,on="account_id",how="left")

data["churn_flag"] = data["churn_date"].notna().astype(int)

# ----------------------------
# FEATURE SELECTION
# ----------------------------

possible_features = [
    "seats","mrr_amount","usage_count","usage_duration_secs",
    "error_count","resolution_time_hours",
    "first_response_time_minutes","escalation_flag"
]

features = [col for col in possible_features if col in data.columns]

X = data[features].fillna(0)
y = data["churn_flag"]

# ----------------------------
# MODEL
# ----------------------------

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train,X_test,y_train,y_test = train_test_split(
    X_scaled,y,test_size=0.2,random_state=42
)

model = RandomForestClassifier(n_estimators=200,random_state=42)
model.fit(X_train,y_train)

y_pred = model.predict(X_test)

# ----------------------------
# KPI METRICS
# ----------------------------

st.subheader("📈 Model Performance")

c1,c2,c3,c4 = st.columns(4)
c1.metric("Accuracy", round(accuracy_score(y_test,y_pred),3))
c2.metric("Precision", round(precision_score(y_test,y_pred),3))
c3.metric("Recall", round(recall_score(y_test,y_pred),3))
c4.metric("F1 Score", round(f1_score(y_test,y_pred),3))

st.divider()

# ----------------------------
# FEATURE IMPORTANCE (INTERACTIVE)
# ----------------------------

importance = model.feature_importances_

importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": importance
}).sort_values("Importance",ascending=True)

st.subheader("🎯 Feature Importance")

fig_imp = px.bar(
    importance_df,
    x="Importance",
    y="Feature",
    orientation="h",
    color="Importance",
    color_continuous_scale="viridis"
)

st.plotly_chart(fig_imp,use_container_width=True)

# ----------------------------
# RENEWAL PROBABILITY
# ----------------------------

data["renewal_probability"] = model.predict_proba(X_scaled)[:,1]

def segment(p):
    if p < 0.3:
        return "Likely Renew"
    elif p < 0.7:
        return "At Risk"
    else:
        return "Likely Churn"

data["segment"] = data["renewal_probability"].apply(segment)

# ----------------------------
# SIDEBAR FILTERS
# ----------------------------

st.sidebar.subheader("Filter Customers")

segment_filter = st.sidebar.multiselect(
    "Select Segment",
    data["segment"].unique(),
    default=data["segment"].unique()
)

prob_range = st.sidebar.slider(
    "Renewal Probability Range",
    0.0,1.0,(0.0,1.0)
)

filtered_data = data[
    (data["segment"].isin(segment_filter)) &
    (data["renewal_probability"].between(prob_range[0],prob_range[1]))
]

# ----------------------------
# SEGMENT DISTRIBUTION
# ----------------------------

st.subheader("👥 Customer Segments")

fig_seg = px.bar(
    filtered_data["segment"].value_counts().reset_index(),
    x="segment",
    y="count",
    color="segment",
    title="Segment Distribution"
)

st.plotly_chart(fig_seg,use_container_width=True)

# ----------------------------
# PROBABILITY DISTRIBUTION
# ----------------------------

st.subheader("📊 Renewal Probability Distribution")

fig_prob = px.histogram(
    filtered_data,
    x="renewal_probability",
    nbins=30,
    color="segment"
)

st.plotly_chart(fig_prob,use_container_width=True)

# ----------------------------
# CONFUSION MATRIX (INTERACTIVE)
# ----------------------------

st.subheader("Confusion Matrix")

cm = confusion_matrix(y_test,y_pred)

fig_cm = px.imshow(
    cm,
    text_auto=True,
    color_continuous_scale="Blues"
)

st.plotly_chart(fig_cm)

# ----------------------------
# DOWNLOAD RESULTS
# ----------------------------

st.subheader("Download Predictions")

csv = filtered_data.to_csv(index=False)

st.download_button(
    label="Download CSV",
    data=csv,
    file_name="churn_predictions.csv",
    mime="text/csv"
)

# ----------------------------
# RAW DATA VIEW
# ----------------------------

if show_raw:
    st.subheader("Filtered Dataset")
    st.dataframe(filtered_data)