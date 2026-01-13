import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Load data
df = pd.read_csv("data/creditcard.csv")

# Separate features and labels
X = df.drop("Class", axis=1)
y = df["Class"]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train Isolation Forest
model = IsolationForest(
    n_estimators=100,
    contamination=0.0017,  # approx fraud ratio
    random_state=42
)

model.fit(X_scaled)

# Predict anomalies
y_pred = model.predict(X_scaled)

# Convert predictions: -1 = fraud, 1 = normal
y_pred = np.where(y_pred == -1, 1, 0)

# Evaluation
print("Confusion Matrix:")
print(confusion_matrix(y, y_pred))

print("\nClassification Report:")
print(classification_report(y, y_pred))

# Save model & scaler
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\nModel and scaler saved successfully.")
