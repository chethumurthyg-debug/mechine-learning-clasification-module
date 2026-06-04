# pages/03_Predict.py

import streamlit as st
import pandas as pd
import joblib
from datetime import datetime

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="Breast Cancer Prediction",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 Breast Cancer Prediction")
st.markdown("Enter the patient measurements and click **Predict**.")

# -------------------------------------------------
# LOAD MODEL
# -------------------------------------------------
try:
    model = joblib.load("models/breast_cancer_model.pkl")
except Exception:
    st.error("Model file not found: models/breast_cancer_model.pkl")
    st.stop()

# -------------------------------------------------
# LOAD DATASET FOR FEATURE NAMES
# -------------------------------------------------
try:
    df = pd.read_csv("data/breast-cancer.csv")

    if "id" in df.columns:
        df = df.drop("id", axis=1)

    if "Unnamed: 32" in df.columns:
        df = df.drop("Unnamed: 32", axis=1)

    feature_columns = [col for col in df.columns if col != "diagnosis"]

except Exception:
    st.error("Dataset file not found.")
    st.stop()

# -------------------------------------------------
# INPUT FORM
# -------------------------------------------------
st.subheader("Patient Measurements")

input_values = {}

cols = st.columns(3)

for idx, feature in enumerate(feature_columns):

    mean_value = float(df[feature].mean())
    min_value = float(df[feature].min())
    max_value = float(df[feature].max())

    with cols[idx % 3]:
        input_values[feature] = st.number_input(
            feature.replace("_", " ").title(),
            min_value=min_value,
            max_value=max_value,
            value=mean_value,
            format="%.4f"
        )

# -------------------------------------------------
# PREDICTION BUTTON
# -------------------------------------------------
if st.button("🔍 Predict", use_container_width=True):

    input_df = pd.DataFrame([input_values])

    prediction = model.predict(input_df)[0]

    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(input_df)[0]
    else:
        probability = None

    st.markdown("---")
    st.subheader("Prediction Result")

    if prediction == 1:

        st.error("⚠️ Prediction: Malignant")

        if probability is not None:
            st.metric(
                "Malignant Probability",
                f"{probability[1]*100:.2f}%"
            )

    else:

        st.success("✅ Prediction: Benign")

        if probability is not None:
            st.metric(
                "Benign Probability",
                f"{probability[0]*100:.2f}%"
            )

    # -----------------------------------------
    # Probability Breakdown
    # -----------------------------------------
    if probability is not None:

        st.subheader("Probability Breakdown")

        prob_df = pd.DataFrame({
            "Class": ["Benign", "Malignant"],
            "Probability (%)": [
                probability[0] * 100,
                probability[1] * 100
            ]
        })

        st.dataframe(
            prob_df,
            use_container_width=True
        )

        st.bar_chart(
            prob_df.set_index("Class")
        )

    # -----------------------------------------
    # Patient Input Summary
    # -----------------------------------------
    st.subheader("Input Summary")

    st.dataframe(
        input_df,
        use_container_width=True
    )

    # -----------------------------------------
    # Download Prediction Report
    # -----------------------------------------
    result_label = (
        "Malignant"
        if prediction == 1
        else "Benign"
    )

    report_df = pd.DataFrame({
        "Timestamp": [
            datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        ],
        "Prediction": [result_label]
    })

    if probability is not None:
        report_df["Benign Probability (%)"] = [
            round(probability[0] * 100, 2)
        ]
        report_df["Malignant Probability (%)"] = [
            round(probability[1] * 100, 2)
        ]

    csv = report_df.to_csv(
        index=False
    )

    st.download_button(
        label="📥 Download Prediction Report",
        data=csv,
        file_name="prediction_report.csv",
        mime="text/csv"
    )

# -------------------------------------------------
# INFORMATION SECTION
# -------------------------------------------------
st.markdown("---")

with st.expander("ℹ️ About This Prediction Tool"):

    st.write("""
    This tool uses a trained Random Forest machine learning model
    to classify breast cancer tumors as:

    - **Benign (B)** → Non-cancerous
    - **Malignant (M)** → Cancerous

    The prediction is based on tumor measurements from the
    Breast Cancer Wisconsin dataset.
    """)

st.success("Prediction Module Ready")
