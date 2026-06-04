# breast_cancer_model.py

import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# -----------------------------
# Load Dataset
# -----------------------------
DATA_PATH = "data/breast-cancer.csv"

df = pd.read_csv(DATA_PATH)

print("Dataset Shape:", df.shape)
print(df.head())

# -----------------------------
# Data Preprocessing
# -----------------------------

# Remove unnecessary columns if present
if "id" in df.columns:
    df.drop("id", axis=1, inplace=True)

if "Unnamed: 32" in df.columns:
    df.drop("Unnamed: 32", axis=1, inplace=True)

# Encode target column
# M = Malignant -> 1
# B = Benign -> 0

if df["diagnosis"].dtype == object:
    df["diagnosis"] = df["diagnosis"].map({
        "M": 1,
        "B": 0
    })

# Features and Target
X = df.drop("diagnosis", axis=1)
y = df["diagnosis"]

print("\nFeatures:", X.shape)
print("Target:", y.shape)

# -----------------------------
# Train Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Samples:", len(X_train))
print("Testing Samples:", len(X_test))

# -----------------------------
# Model Training
# -----------------------------
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

model.fit(X_train, y_train)

# -----------------------------
# Predictions
# -----------------------------
y_pred = model.predict(X_test)

# -----------------------------
# Evaluation
# -----------------------------
accuracy = accuracy_score(y_test, y_pred)

print("\n" + "=" * 50)
print("MODEL PERFORMANCE")
print("=" * 50)

print(f"\nAccuracy: {accuracy:.4f}")

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

# -----------------------------
# Save Model
# -----------------------------
os.makedirs("models", exist_ok=True)

MODEL_PATH = "models/breast_cancer_model.pkl"

joblib.dump(model, MODEL_PATH)

print(f"\nModel saved successfully:")
print(MODEL_PATH)

# -----------------------------
# Feature Importance
# -----------------------------
feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop 10 Important Features:")
print(feature_importance.head(10))
