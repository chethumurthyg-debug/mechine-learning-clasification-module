# pages/01_Analytics.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# -----------------------------------
# Page Configuration
# -----------------------------------
st.set_page_config(
    page_title="Breast Cancer Analytics",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Breast Cancer Data Analytics")
st.markdown("---")

# -----------------------------------
# Load Dataset
# -----------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data/breast-cancer.csv")

    if "Unnamed: 32" in df.columns:
        df.drop("Unnamed: 32", axis=1, inplace=True)

    return df

df = load_data()

# -----------------------------------
# Dataset Overview
# -----------------------------------
st.subheader("Dataset Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Rows", df.shape[0])

with col2:
    st.metric("Columns", df.shape[1])

with col3:
    st.metric("Missing Values", int(df.isnull().sum().sum()))

st.dataframe(df.head())

# -----------------------------------
# Dataset Information
# -----------------------------------
st.subheader("Data Types")

dtype_df = pd.DataFrame({
    "Column": df.columns,
    "Datatype": df.dtypes.astype(str)
})

st.dataframe(dtype_df)

# -----------------------------------
# Target Distribution
# -----------------------------------
st.subheader("Diagnosis Distribution")

diagnosis_count = df["diagnosis"].value_counts()

fig = px.pie(
    values=diagnosis_count.values,
    names=diagnosis_count.index,
    title="Benign vs Malignant"
)

st.plotly_chart(fig, use_container_width=True)

# -----------------------------------
# Count Plot
# -----------------------------------
fig, ax = plt.subplots(figsize=(6, 4))

sns.countplot(
    data=df,
    x="diagnosis",
    ax=ax
)

ax.set_title("Diagnosis Count")

st.pyplot(fig)

# -----------------------------------
# Statistical Summary
# -----------------------------------
st.subheader("Statistical Summary")

st.dataframe(df.describe())

# -----------------------------------
# Missing Values
# -----------------------------------
st.subheader("Missing Values")

missing_df = pd.DataFrame(
    df.isnull().sum(),
    columns=["Missing Values"]
)

st.dataframe(missing_df)

# -----------------------------------
# Correlation Heatmap
# -----------------------------------
st.subheader("Correlation Heatmap")

numeric_df = df.select_dtypes(include=["float64", "int64"])

corr_matrix = numeric_df.corr()

fig, ax = plt.subplots(figsize=(14, 10))

sns.heatmap(
    corr_matrix,
    cmap="coolwarm",
    linewidths=0.5
)

st.pyplot(fig)

# -----------------------------------
# Feature Distribution
# -----------------------------------
st.subheader("Feature Distribution")

feature = st.selectbox(
    "Select Feature",
    numeric_df.columns
)

fig = px.histogram(
    df,
    x=feature,
    color="diagnosis",
    marginal="box",
    nbins=30,
    title=f"Distribution of {feature}"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# -----------------------------------
# Box Plot Analysis
# -----------------------------------
st.subheader("Box Plot Analysis")

box_feature = st.selectbox(
    "Choose Feature for Box Plot",
    numeric_df.columns,
    key="boxplot"
)

fig = px.box(
    df,
    x="diagnosis",
    y=box_feature,
    color="diagnosis",
    title=f"{box_feature} by Diagnosis"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# -----------------------------------
# Scatter Plot Analysis
# -----------------------------------
st.subheader("Feature Relationship")

col1, col2 = st.columns(2)

with col1:
    x_feature = st.selectbox(
        "X Axis",
        numeric_df.columns,
        index=0
    )

with col2:
    y_feature = st.selectbox(
        "Y Axis",
        numeric_df.columns,
        index=1
    )

fig = px.scatter(
    df,
    x=x_feature,
    y=y_feature,
    color="diagnosis",
    title=f"{x_feature} vs {y_feature}"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# -----------------------------------
# Top Correlated Features
# -----------------------------------
st.subheader("Top Correlated Features")

if "diagnosis" in df.columns:

    temp_df = df.copy()

    temp_df["diagnosis"] = temp_df["diagnosis"].map({
        "M": 1,
        "B": 0
    })

    correlation = temp_df.corr(
        numeric_only=True
    )["diagnosis"].sort_values(
        ascending=False
    )

    st.dataframe(
        correlation.reset_index().rename(
            columns={
                "index": "Feature",
                "diagnosis": "Correlation"
            }
        )
    )

# -----------------------------------
# Download Dataset
# -----------------------------------
st.subheader("Download Dataset")

csv = df.to_csv(index=False)

st.download_button(
    label="📥 Download Dataset",
    data=csv,
    file_name="breast_cancer_dataset.csv",
    mime="text/csv"
)

# -----------------------------------
# Footer
# -----------------------------------
st.markdown("---")
st.success("Analytics Dashboard Loaded Successfully")
