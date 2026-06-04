# pages/02_Model_Performance.py

import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)

# ----------------------------------------
# PAGE CONFIG
# ----------------------------------------
st.set_page_config(
    page_title="Model Performance",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Breast Cancer Model Performance")
st.markdown("---")

# ----------------------------------------
# LOAD DATA
# ----------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data/breast-cancer.csv")

    if "Unnamed: 32" in df.columns:
        df.drop("Unnamed: 32", axis=1, inplace=True)

    if "id" in df.columns:
        df.drop("id", axis=1, inplace=True)

    return df


df = load_data()

# ----------------------------------------
# PREPROCESS
# ----------------------------------------
df["diagnosis"] = df["diagnosis"].map({
    "M": 1,
    "B": 0
})

X = df.drop("diagnosis", axis=1)
y = df["diagnosis"]

# ----------------------------------------
# TRAIN TEST SPLIT
# ----------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ----------------------------------------
# LOAD MODEL
# ----------------------------------------
try:
    model = joblib.load(
        "models/breast_cancer_model.pkl"
    )
except Exception as e:
    st.error(
        "Model file not found. Train model first."
    )
    st.stop()

# ----------------------------------------
# PREDICTIONS
# ----------------------------------------
y_pred = model.predict(X_test)

if hasattr(model, "predict_proba"):
    y_prob = model.predict_proba(X_test)[:, 1]
else:
    y_prob = None

# ----------------------------------------
# METRICS
# ----------------------------------------
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# ----------------------------------------
# PERFORMANCE METRICS
# ----------------------------------------
st.subheader("📈 Performance Metrics")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Accuracy",
        f"{accuracy:.4f}"
    )

with col2:
    st.metric(
        "Precision",
        f"{precision:.4f}"
    )

with col3:
    st.metric(
        "Recall",
        f"{recall:.4f}"
    )

with col4:
    st.metric(
        "F1 Score",
        f"{f1:.4f}"
    )

# ----------------------------------------
# CONFUSION MATRIX
# ----------------------------------------
st.markdown("---")
st.subheader("🔍 Confusion Matrix")

cm = confusion_matrix(
    y_test,
    y_pred
)

fig, ax = plt.subplots(figsize=(6, 4))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Benign", "Malignant"],
    yticklabels=["Benign", "Malignant"]
)

ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix")

st.pyplot(fig)

# ----------------------------------------
# CLASSIFICATION REPORT
# ----------------------------------------
st.markdown("---")
st.subheader("📋 Classification Report")

report = classification_report(
    y_test,
    y_pred,
    output_dict=True
)

report_df = pd.DataFrame(
    report
).transpose()

st.dataframe(
    report_df,
    use_container_width=True
)

# ----------------------------------------
# ROC CURVE
# ----------------------------------------
if y_prob is not None:

    st.markdown("---")
    st.subheader("📉 ROC Curve")

    fpr, tpr, _ = roc_curve(
        y_test,
        y_prob
    )

    roc_auc = auc(
        fpr,
        tpr
    )

    fig, ax = plt.subplots(
        figsize=(7, 5)
    )

    ax.plot(
        fpr,
        tpr,
        label=f"AUC = {roc_auc:.4f}"
    )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--"
    )

    ax.set_xlabel(
        "False Positive Rate"
    )

    ax.set_ylabel(
        "True Positive Rate"
    )

    ax.set_title(
        "ROC Curve"
    )

    ax.legend()

    st.pyplot(fig)

# ----------------------------------------
# FEATURE IMPORTANCE
# ----------------------------------------
st.markdown("---")
st.subheader("⭐ Feature Importance")

if hasattr(model, "feature_importances_"):

    importance_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance": model.feature_importances_
    })

    importance_df = (
        importance_df
        .sort_values(
            by="Importance",
            ascending=False
        )
    )

    st.dataframe(
        importance_df.head(20),
        use_container_width=True
    )

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    sns.barplot(
        data=importance_df.head(15),
        x="Importance",
        y="Feature",
        ax=ax
    )

    ax.set_title(
        "Top 15 Important Features"
    )

    st.pyplot(fig)

# ----------------------------------------
# TEST DATA PREVIEW
# ----------------------------------------
st.markdown("---")
st.subheader("🧪 Test Dataset Preview")

preview_df = X_test.copy()

preview_df["Actual"] = y_test.values
preview_df["Predicted"] = y_pred

st.dataframe(
    preview_df.head(20),
    use_container_width=True
)

# ----------------------------------------
# DOWNLOAD REPORT
# ----------------------------------------
st.markdown("---")
st.subheader("📥 Download Metrics")

metrics_df = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ],
    "Value": [
        accuracy,
        precision,
        recall,
        f1
    ]
})

csv = metrics_df.to_csv(
    index=False
)

st.download_button(
    label="Download Metrics CSV",
    data=csv,
    file_name="model_metrics.csv",
    mime="text/csv"
)

# ----------------------------------------
# FOOTER
# ----------------------------------------
st.markdown("---")
st.success(
    "Model Performance Dashboard Loaded Successfully"
)
