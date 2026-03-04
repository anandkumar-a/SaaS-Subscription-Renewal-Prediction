import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# -------------------------
# Load your CSVs
# -------------------------
accounts = pd.read_csv("ravenstack_accounts.csv")
subscriptions = pd.read_csv("ravenstack_subscriptions.csv")
usage = pd.read_csv("ravenstack_feature_usage.csv")
tickets = pd.read_csv("ravenstack_support_tickets.csv")
churn = pd.read_csv("ravenstack_churn_events.csv")

# -------------------------
# Feature Engineering
# -------------------------
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
    "seats","mrr_amount","usage_count","usage_duration_secs",
    "error_count","resolution_time_hours",
    "first_response_time_minutes","escalation_flag"
]
features = [col for col in possible_features if col in data.columns]

X = data[features].fillna(0)
y = data["churn_flag"]

# -------------------------
# Scale features
# -------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------
# Split data and train model
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# -------------------------
# Save model & scaler
# -------------------------
joblib.dump(model, "rf_model.joblib")
joblib.dump(scaler, "scaler.joblib")

print("✅ Model and scaler saved successfully!")