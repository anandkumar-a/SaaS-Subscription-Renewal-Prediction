import streamlit as st

st.set_page_config(
    page_title="SaaS Analytics Platform",
    page_icon="📊",
    layout="wide"
)

# -----------------------------
# GLASSMORPHISM UI + ANIMATION
# -----------------------------

st.markdown("""
<style>

[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg,#0f2027,#203a43,#2c5364);
}

/* Glass card */
.glass {
    background: rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 25px;
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    border: 1px solid rgba(255,255,255,0.18);
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
}

/* Animated KPI */
.kpi {
    animation: float 3s ease-in-out infinite;
}

@keyframes float {
    0% { transform: translateY(0px);}
    50% { transform: translateY(-8px);}
    100% { transform: translateY(0px);}
}

</style>
""", unsafe_allow_html=True)

st.title("🚀 SaaS Customer Analytics Platform")

st.markdown("""
<div class="glass">
<h2>Welcome to SmartChurn</h2>
<p>Make smarter retention decisions with predictive analytics and interactive dashboards.</p>

### Navigate Your Insights:
📊 Predictive Model Dashboard  
💡 Customer Behavior Insights  
🗂️ Data Explorer

</div>
""", unsafe_allow_html=True)

st.sidebar.success("Select a page above.")

