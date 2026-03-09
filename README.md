# 💳 Credit Card Fraud Anomaly Detector

A machine learning-powered web application that detects potentially fraudulent credit card transactions using **anomaly detection techniques**. The system analyzes transaction patterns and flags suspicious activity in real time through an interactive **Streamlit dashboard**.

---

## 🚀 Project Overview

Credit card fraud is rare but highly damaging. Traditional classification models struggle with such **extremely imbalanced datasets**, where fraudulent transactions represent less than **0.2% of all transactions**.

This project uses **Isolation Forest**, an unsupervised anomaly detection algorithm, to identify transactions that deviate significantly from normal behavior.

The application provides an interactive interface where users can simulate transactions and determine whether they are **normal or potentially fraudulent**.

---

## 🧠 Machine Learning Approach

The model is trained using **Isolation Forest**, an anomaly detection algorithm designed to isolate unusual observations.

### Key Concept

Normal transactions require **many splits** to isolate.

Fraudulent transactions are **isolated quickly** because they behave differently from the majority.

Model output:

| Prediction | Meaning |
|-----------|---------|
| `1` | Normal Transaction |
| `-1` | Fraudulent / Suspicious Transaction |

This makes Isolation Forest well suited for detecting rare anomalies such as fraud.

---

## 📊 Dataset

Dataset used: **Credit Card Fraud Detection Dataset**

Dataset characteristics:

- **284,807 transactions**
- **492 fraudulent transactions**
- Fraud ratio: **0.17%**
- Highly imbalanced dataset

Features include:

- **Time** – Seconds elapsed between transactions
- **Amount** – Transaction amount
- **V1–V28** – PCA-transformed anonymized features
- **Class** – Fraud label (used only for evaluation)

---

## ⚙️ Tech Stack

| Component | Technology |
|--------|-------------|
| Programming Language | Python |
| Machine Learning | Scikit-learn |
| Model | Isolation Forest |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib |
| Web Interface | Streamlit |
| Model Storage | Joblib |

---

## ✨ Features

- Detect anomalous credit card transactions
- Interactive Streamlit dashboard
- Adjustable fraud **sensitivity threshold**
- Real-time transaction prediction
- User-friendly interface for testing transactions
- Lightweight machine learning deployment

