# utils/visualizations.py

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc
)


class BreastCancerVisualizer:

    def __init__(self):
        sns.set_style("whitegrid")

    # --------------------------------------------------
    # Correlation Heatmap
    # --------------------------------------------------
    def correlation_heatmap(self, df):

        numeric_df = df.select_dtypes(
            include=["int64", "float64"]
        )

        corr_matrix = numeric_df.corr()

        fig, ax = plt.subplots(
            figsize=(14, 10)
        )

        sns.heatmap(
            corr_matrix,
            cmap="coolwarm",
            linewidths=0.5,
            ax=ax
        )

        ax.set_title(
            "Correlation Heatmap"
        )

        return fig

    # --------------------------------------------------
    # Target Distribution
    # --------------------------------------------------
    def target_distribution(
        self,
        df,
        target_column="diagnosis"
    ):

        fig, ax = plt.subplots(
            figsize=(7, 5)
        )

        sns.countplot(
            x=target_column,
            data=df,
            ax=ax
        )

        ax.set_title(
            "Target Distribution"
        )

        return fig

    # --------------------------------------------------
    # Feature Histogram
    # --------------------------------------------------
    def feature_distribution(
        self,
        df,
        feature
    ):

        fig, ax = plt.subplots(
            figsize=(8, 5)
        )

        sns.histplot(
            data=df,
            x=feature,
            kde=True,
            ax=ax
        )

        ax.set_title(
            f"{feature} Distribution"
        )

        return fig

    # --------------------------------------------------
    # Box Plot
    # --------------------------------------------------
    def box_plot(
        self,
        df,
        feature,
        target="diagnosis"
    ):

        fig, ax = plt.subplots(
            figsize=(8, 5)
        )

        sns.boxplot(
            data=df,
            x=target,
            y=feature,
            ax=ax
        )

        ax.set_title(
            f"{feature} vs {target}"
        )

        return fig

    # --------------------------------------------------
    # Scatter Plot
    # --------------------------------------------------
    def scatter_plot(
        self,
        df,
        x_feature,
        y_feature,
        target="diagnosis"
    ):

        fig, ax = plt.subplots(
            figsize=(8, 5)
        )

        sns.scatterplot(
            data=df,
            x=x_feature,
            y=y_feature,
            hue=target,
            ax=ax
        )

        ax.set_title(
            f"{x_feature} vs {y_feature}"
        )

        return fig

    # --------------------------------------------------
    # Confusion Matrix
    # --------------------------------------------------
    def confusion_matrix_plot(
        self,
        y_true,
        y_pred
    ):

        cm = confusion_matrix(
            y_true,
            y_pred
        )

        fig, ax = plt.subplots(
            figsize=(6, 4)
        )

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=[
                "Benign",
                "Malignant"
            ],
            yticklabels=[
                "Benign",
                "Malignant"
            ],
            ax=ax
        )

        ax.set_title(
            "Confusion Matrix"
        )

        ax.set_xlabel(
            "Predicted"
        )

        ax.set_ylabel(
            "Actual"
        )

        return fig

    # --------------------------------------------------
    # ROC Curve
    # --------------------------------------------------
    def roc_curve_plot(
        self,
        y_true,
        y_prob
    ):

        fpr, tpr, _ = roc_curve(
            y_true,
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

        return fig

    # --------------------------------------------------
    # Feature Importance
    # --------------------------------------------------
    def feature_importance_plot(
        self,
        model,
        feature_names,
        top_n=15
    ):

        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance":
            model.feature_importances_
        })

        importance_df = (
            importance_df
            .sort_values(
                by="Importance",
                ascending=False
            )
            .head(top_n)
        )

        fig, ax = plt.subplots(
            figsize=(10, 6)
        )

        sns.barplot(
            data=importance_df,
            x="Importance",
            y="Feature",
            ax=ax
        )

        ax.set_title(
            f"Top {top_n} Feature Importance"
        )

        return fig

    # --------------------------------------------------
    # Missing Values Plot
    # --------------------------------------------------
    def missing_values_plot(
        self,
        df
    ):

        missing = (
            df.isnull()
            .sum()
            .sort_values(
                ascending=False
            )
        )

        fig, ax = plt.subplots(
            figsize=(10, 5)
        )

        missing.plot(
            kind="bar",
            ax=ax
        )

        ax.set_title(
            "Missing Values"
        )

        ax.set_ylabel(
            "Count"
        )

        return fig

    # --------------------------------------------------
    # Pair Plot
    # --------------------------------------------------
    def pair_plot(
        self,
        df,
        target="diagnosis"
    ):

        selected_cols = list(
            df.columns[:5]
        )

        if target not in selected_cols:
            selected_cols.append(
                target
            )

        pair = sns.pairplot(
            df[selected_cols],
            hue=target
        )

        return pair.fig


# --------------------------------------------------
# Testing Module
# --------------------------------------------------
if __name__ == "__main__":

    df = pd.read_csv(
        "data/breast-cancer.csv"
    )

    visualizer = (
        BreastCancerVisualizer()
    )

    fig = (
        visualizer
        .correlation_heatmap(df)
    )

    plt.show()
