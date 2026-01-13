import streamlit as st
import numpy as np
import joblib
import os

# =========================
# Page Config
# =========================
st.set_page_config(
    page_title="Fraud Detection System",
    layout="wide"
)

# =========================
# Load Model & Scaler
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))

TOTAL_FEATURES = 30  # model was trained on 30 features

# =========================
# Sidebar Controls
# =========================
st.sidebar.header("🛠 Transaction Options")

transaction_type = st.sidebar.radio(
    "Choose transaction example:",
    ["Normal Transaction (Safe)", "Fraud Transaction (Risky)"]
)

sensitivity = st.sidebar.slider(
    "Detection Sensitivity (Higher = more strict)",
    min_value=-0.5,
    max_value=0.5,
    value=0.0,
    step=0.01
)

# =========================
# Sample Transactions
# =========================
if transaction_type == "Normal Transaction (Safe)":
    visible_features = [
        -1.22, 0.61, -0.49, 1.07, 2.61,
        -0.80, 0.24, -1.14, 0.06, -1.63
    ]
else:
    visible_features = [
        4.2, -6.5, 7.1, -8.3, 5.9,
        -7.4, 6.2, -9.1, 8.0, -6.8
    ]

# =========================
# Fill Remaining Features
# =========================
full_features = visible_features + [0.0] * (TOTAL_FEATURES - len(visible_features))
full_features = np.array(full_features).reshape(1, -1)

# Scale features
scaled_features = scaler.transform(full_features)

# =========================
# Prediction
# =========================
anomaly_score = model.decision_function(scaled_features)[0]
is_fraud = anomaly_score < sensitivity

# =========================
# UI Layout
# =========================
st.title("💳 Fraud Detection System (Anomaly Detection)")
st.write(
    "This system checks whether a transaction looks **normal** or **suspicious** "
    "by comparing it with past transaction patterns."
)

st.subheader("📄 Transaction Details")

cols = st.columns(10)
for i, value in enumerate(visible_features):
    cols[i].metric(f"Pattern {i+1}", round(value, 4))

st.subheader("📊 Prediction Result")
st.metric("Anomaly Score", round(anomaly_score, 4))

if is_fraud:
    st.error("🚨 **Fraudulent Transaction Detected**")
else:
    st.success("✅ **Transaction is Normal**")
