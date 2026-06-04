# utils/preprocessing.py

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class BreastCancerPreprocessor:

    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.scaler = StandardScaler()

    # ----------------------------------
    # Load Dataset
    # ----------------------------------
    def load_data(self):
        """
        Load dataset from CSV
        """

        self.df = pd.read_csv(self.data_path)

        return self.df

    # ----------------------------------
    # Clean Dataset
    # ----------------------------------
    def clean_data(self):
        """
        Remove unwanted columns
        """

        if self.df is None:
            raise ValueError("Dataset not loaded.")

        # Drop ID column if present
        if "id" in self.df.columns:
            self.df.drop(
                columns=["id"],
                inplace=True
            )

        # Drop empty column if present
        if "Unnamed: 32" in self.df.columns:
            self.df.drop(
                columns=["Unnamed: 32"],
                inplace=True
            )

        return self.df

    # ----------------------------------
    # Encode Target
    # ----------------------------------
    def encode_target(self):
        """
        Convert diagnosis:
        M -> 1
        B -> 0
        """

        if self.df is None:
            raise ValueError("Dataset not loaded.")

        if "diagnosis" in self.df.columns:

            self.df["diagnosis"] = (
                self.df["diagnosis"]
                .map({
                    "M": 1,
                    "B": 0
                })
            )

        return self.df

    # ----------------------------------
    # Handle Missing Values
    # ----------------------------------
    def handle_missing_values(self):
        """
        Fill missing values
        """

        if self.df is None:
            raise ValueError("Dataset not loaded.")

        numeric_cols = self.df.select_dtypes(
            include=["int64", "float64"]
        ).columns

        self.df[numeric_cols] = (
            self.df[numeric_cols]
            .fillna(
                self.df[numeric_cols].median()
            )
        )

        return self.df

    # ----------------------------------
    # Feature & Target Split
    # ----------------------------------
    def split_features_target(self):

        if self.df is None:
            raise ValueError("Dataset not loaded.")

        X = self.df.drop(
            columns=["diagnosis"]
        )

        y = self.df["diagnosis"]

        return X, y

    # ----------------------------------
    # Train-Test Split
    # ----------------------------------
    def train_test_split_data(
        self,
        X,
        y,
        test_size=0.2,
        random_state=42
    ):

        X_train, X_test, y_train, y_test = (
            train_test_split(
                X,
                y,
                test_size=test_size,
                random_state=random_state,
                stratify=y
            )
        )

        return (
            X_train,
            X_test,
            y_train,
            y_test
        )

    # ----------------------------------
    # Feature Scaling
    # ----------------------------------
    def scale_features(
        self,
        X_train,
        X_test
    ):
        """
        Standardize features
        """

        X_train_scaled = (
            self.scaler.fit_transform(
                X_train
            )
        )

        X_test_scaled = (
            self.scaler.transform(
                X_test
            )
        )

        return (
            X_train_scaled,
            X_test_scaled
        )

    # ----------------------------------
    # Full Pipeline
    # ----------------------------------
    def preprocess(self):

        self.load_data()

        self.clean_data()

        self.handle_missing_values()

        self.encode_target()

        X, y = self.split_features_target()

        (
            X_train,
            X_test,
            y_train,
            y_test
        ) = self.train_test_split_data(
            X,
            y
        )

        (
            X_train_scaled,
            X_test_scaled
        ) = self.scale_features(
            X_train,
            X_test
        )

        return {
            "X_train": X_train_scaled,
            "X_test": X_test_scaled,
            "y_train": y_train,
            "y_test": y_test,
            "feature_names": X.columns.tolist(),
            "scaler": self.scaler
        }


# --------------------------------------------------
# Example Usage
# --------------------------------------------------
if __name__ == "__main__":

    processor = BreastCancerPreprocessor(
        "data/breast-cancer.csv"
    )

    data = processor.preprocess()

    print("Preprocessing Complete")

    print(
        "X_train shape:",
        data["X_train"].shape
    )

    print(
        "X_test shape:",
        data["X_test"].shape
    )

    print(
        "Features:",
        len(data["feature_names"])
    )
