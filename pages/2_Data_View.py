import streamlit as st
import pandas as pd

st.title("📂 Data Explorer")

accounts = pd.read_csv("ravenstack_accounts.csv")

st.markdown('<div class="glass">Raw Dataset</div>', unsafe_allow_html=True)

st.dataframe(accounts)