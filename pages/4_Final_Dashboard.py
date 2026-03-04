import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="SaaS Business Dashboard", layout="wide")

# ---------- GLASS UI ----------
st.markdown("""
<style>
.big-title {
    font-size:28px;
    font-weight:700;
}
.glass {
    background: rgba(255,255,255,0.1);
    padding:15px;
    border-radius:12px;
    backdrop-filter: blur(10px);
    margin-bottom:10px;
}
.metric-card {
    background: rgba(255,255,255,0.08);
    padding:20px;
    border-radius:10px;
    text-align:center;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="big-title">🚀 SaaS Platform Business Dashboard</div>', unsafe_allow_html=True)
st.caption("Subscription Renewal Prediction System Insights")

# ---------- LOAD DATA ----------
try:
    accounts = pd.read_csv("ravenstack_accounts.csv")
except:
    accounts = pd.DataFrame()
    st.warning("accounts.csv not found. Using empty dataset.")

try:
    subscriptions = pd.read_csv("ravenstack_subscriptions.csv")
except:
    subscriptions = pd.DataFrame()
    st.warning("subscriptions.csv not found. Using empty dataset.")

# ---------- KEY METRICS ----------
st.markdown("## 🚀 Key Business Metrics")

col1, col2, col3, col4 = st.columns(4)

total_customers = accounts.shape[0] if not accounts.empty else 0
active_subs = subscriptions.shape[0] if not subscriptions.empty else 0

avg_seats = 0
if not accounts.empty and "seats" in accounts.columns:
    avg_seats = int(accounts["seats"].mean())

revenue = avg_seats * active_subs * 20  # simple estimation

col1.metric("Total Customers", total_customers)
col2.metric("Active Subscriptions", active_subs)
col3.metric("Average Seats", avg_seats)
col4.metric("Revenue", revenue)

st.divider()

# ---------- SMALL CHARTS ----------
st.markdown("## 📊 Business Insights")

left, right = st.columns(2)

# ----- PLAN DISTRIBUTION -----
st.subheader("📈 Plan Distribution")

try:
    if 'plan_tier' in accounts.columns:

        plan_counts = accounts['plan_tier'].value_counts()

        # 👇 create blank space around chart
        left, center, right = st.columns([2, 1, 2])

        with left:  # chart only in small middle space
            fig, ax = plt.subplots(figsize=(3, 2))  # very small chart

            plan_counts.plot(kind='bar', ax=ax)

            ax.set_xlabel("Plan Tier")
            ax.set_ylabel("Count")

            # 👇 IMPORTANT → prevents stretching
            st.pyplot(fig, use_container_width=False)

    else:
        st.warning("plan_tier column not available")

except Exception as e:
    st.error(f"Error loading accounts: {e}")

# ----- CUSTOMER SEGMENTATION -----
with right:
    st.markdown("### 👥 Customer Segmentation")

    if not accounts.empty and "plan_tier" in accounts.columns:
        fig2, ax2 = plt.subplots(figsize=(2.5, 2.5))  # SMALL PIE
        accounts["plan_tier"].value_counts().plot(kind="pie", ax=ax2, autopct="%1.0f%%")
        ax2.set_ylabel("")
        st.pyplot(fig2, use_container_width=False)
    else:
        st.info("plan_tier column not available")

st.divider()

# ---------- DATA PREVIEW ----------
st.markdown("## 📋 Customer Data Preview")

if not accounts.empty:
    st.dataframe(accounts.head(), height=200)  # fixed small height
else:
    st.info("No account data available")