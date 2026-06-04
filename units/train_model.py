# utils/train_model.py

import os
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from utils.preprocessing import BreastCancerPreprocessor


# --------------------------------------------------
# Train Model Class
# --------------------------------------------------
class BreastCancerModelTrainer:

    def __init__(self, data_path):

        self.data_path = data_path
        self.model = None
        self.scaler = None

    # --------------------------------------------------
    # Load & Preprocess Data
    # --------------------------------------------------
    def load_data(self):

        processor = BreastCancerPreprocessor(
            self.data_path
        )

        data = processor.preprocess()

        self.scaler = data["scaler"]

        return data

    # --------------------------------------------------
    # Train Model
    # --------------------------------------------------
    def train(self):

        data = self.load_data()

        X_train = data["X_train"]
        X_test = data["X_test"]

        y_train = data["y_train"]
        y_test = data["y_test"]

        feature_names = data["feature_names"]

        print("=" * 60)
        print("TRAINING RANDOM FOREST MODEL")
        print("=" * 60)

        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42
        )

        self.model.fit(
            X_train,
            y_train
        )

        # Predictions
        y_pred = self.model.predict(X_test)

        # Metrics
        accuracy = accuracy_score(
            y_test,
            y_pred
        )

        precision = precision_score(
            y_test,
            y_pred
        )

        recall = recall_score(
            y_test,
            y_pred
        )

        f1 = f1_score(
            y_test,
            y_pred
        )

        print(f"\nAccuracy : {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1 Score : {f1:.4f}")

        print("\nConfusion Matrix")
        print(
            confusion_matrix(
                y_test,
                y_pred
            )
        )

        print("\nClassification Report")
        print(
            classification_report(
                y_test,
                y_pred
            )
        )

        # Save models
        self.save_model()

        self.save_scaler()

        self.save_feature_importance(
            feature_names
        )

        print("\nTraining Completed Successfully")

    # --------------------------------------------------
    # Save Model
    # --------------------------------------------------
    def save_model(self):

        os.makedirs(
            "models",
            exist_ok=True
        )

        model_path = (
            "models/breast_cancer_model.pkl"
        )

        joblib.dump(
            self.model,
            model_path
        )

        print(
            f"\nModel Saved: {model_path}"
        )

    # --------------------------------------------------
    # Save Scaler
    # --------------------------------------------------
    def save_scaler(self):

        scaler_path = (
            "models/scaler.pkl"
        )

        joblib.dump(
            self.scaler,
            scaler_path
        )

        print(
            f"Scaler Saved: {scaler_path}"
        )

    # --------------------------------------------------
    # Save Feature Importance
    # --------------------------------------------------
    def save_feature_importance(
        self,
        feature_names
    ):

        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance":
            self.model.feature_importances_
        })

        importance_df = (
            importance_df
            .sort_values(
                by="Importance",
                ascending=False
            )
        )

        file_path = (
            "models/feature_importance.csv"
        )

        importance_df.to_csv(
            file_path,
            index=False
        )

        print(
            f"Feature Importance Saved: {file_path}"
        )

    # --------------------------------------------------
    # Predict New Data
    # --------------------------------------------------
    def predict(self, sample):

        if self.model is None:
            raise ValueError(
                "Model not trained."
            )

        prediction = (
            self.model.predict(sample)
        )

        probability = (
            self.model.predict_proba(sample)
        )

        return prediction, probability


# --------------------------------------------------
# Run Training
# --------------------------------------------------
if __name__ == "__main__":

    trainer = BreastCancerModelTrainer(
        "data/breast-cancer.csv"
    )

    trainer.train()
