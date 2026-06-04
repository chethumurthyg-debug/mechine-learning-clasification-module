# app.py

import streamlit as st
import pandas as pd

# --------------------------------------
# PAGE CONFIGURATION
# --------------------------------------
st.set_page_config(
    page_title="Breast Cancer Analytics",
    page_icon="🎗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------
# LOAD DATA
# --------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data/breast-cancer.csv")

    if "Unnamed: 32" in df.columns:
        df.drop("Unnamed: 32", axis=1, inplace=True)

    return df


df = load_data()

# --------------------------------------
# SIDEBAR
# --------------------------------------
st.sidebar.title("🎗️ Navigation")

st.sidebar.info(
    """
    Breast Cancer Analytics &
    Classification System

    Use the pages menu above
    to navigate.
    """
)

# --------------------------------------
# MAIN PAGE
# --------------------------------------
st.title("🎗️ Breast Cancer Analytics Dashboard")

st.markdown("""
Welcome to the **Breast Cancer Analytics & Classification System**.

This application helps:
- Explore breast cancer datasets
- Analyze feature distributions
- Evaluate machine learning performance
- Predict tumor diagnosis
""")

st.markdown("---")

# --------------------------------------
# KPI METRICS
# --------------------------------------
total_records = len(df)

if "diagnosis" in df.columns:

    malignant_count = len(
        df[df["diagnosis"] == "M"]
    )

    benign_count = len(
        df[df["diagnosis"] == "B"]
    )

else:
    malignant_count = 0
    benign_count = 0

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Total Records",
        total_records
    )

with col2:
    st.metric(
        "Malignant Cases",
        malignant_count
    )

with col3:
    st.metric(
        "Benign Cases",
        benign_count
    )

# --------------------------------------
# DATASET PREVIEW
# --------------------------------------
st.markdown("---")
st.subheader("📊 Dataset Preview")

st.dataframe(
    df.head(10),
    use_container_width=True
)

# --------------------------------------
# DATASET INFORMATION
# --------------------------------------
st.markdown("---")
st.subheader("📋 Dataset Information")

info_df = pd.DataFrame({
    "Column": df.columns,
    "Data Type": df.dtypes.astype(str)
})

st.dataframe(
    info_df,
    use_container_width=True
)

# --------------------------------------
# STATISTICAL SUMMARY
# --------------------------------------
st.markdown("---")
st.subheader("📈 Statistical Summary")

st.dataframe(
    df.describe(),
    use_container_width=True
)

# --------------------------------------
# MISSING VALUES
# --------------------------------------
st.markdown("---")
st.subheader("🔍 Missing Values")

missing_values = pd.DataFrame({
    "Column": df.columns,
    "Missing Values": df.isnull().sum().values
})

st.dataframe(
    missing_values,
    use_container_width=True
)

# --------------------------------------
# QUICK INSIGHTS
# --------------------------------------
st.markdown("---")
st.subheader("💡 Quick Insights")

st.success(
    f"""
    Dataset contains **{total_records} samples**
    with **{df.shape[1]} features**.
    """
)

if malignant_count > 0 and benign_count > 0:

    malignant_percentage = (
        malignant_count / total_records
    ) * 100

    benign_percentage = (
        benign_count / total_records
    ) * 100

    st.info(
        f"""
        Benign Cases: {benign_percentage:.2f}% |
        Malignant Cases: {malignant_percentage:.2f}%
        """
    )

# --------------------------------------
# DOWNLOAD DATASET
# --------------------------------------
st.markdown("---")
st.subheader("📥 Download Dataset")

csv = df.to_csv(index=False)

st.download_button(
    label="Download CSV",
    data=csv,
    file_name="breast_cancer_dataset.csv",
    mime="text/csv"
)

# --------------------------------------
# FOOTER
# --------------------------------------
st.markdown("---")

st.caption(
    "Breast Cancer Analytics & Classification System | Streamlit + Scikit-learn"
)
