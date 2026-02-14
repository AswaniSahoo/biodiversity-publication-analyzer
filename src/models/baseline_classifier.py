"""
Baseline Classifiers for Biodiversity Publication Classification.

Implements TF-IDF vectorization + classical ML classifiers:
- Logistic Regression
- Support Vector Machine (LinearSVC)
- Random Forest
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


# Available model constructors
MODEL_REGISTRY = {
    "logistic_regression": lambda: LogisticRegression(
        max_iter=1000, C=1.0, solver="lbfgs", random_state=42
    ),
    "svm": lambda: CalibratedClassifierCV(
        LinearSVC(max_iter=2000, C=1.0, random_state=42),
        cv=3,
    ),
    "random_forest": lambda: RandomForestClassifier(
        n_estimators=200, max_depth=None, random_state=42, n_jobs=-1
    ),
}


class BaselineClassifier:
    """TF-IDF + Classical ML classifier pipeline."""

    def __init__(
        self,
        model_name: str = "logistic_regression",
        max_features: int = 10000,
        ngram_range: tuple = (1, 2),
    ):
        """
        Initialize the baseline classifier.

        Args:
            model_name: One of 'logistic_regression', 'svm', 'random_forest'.
            max_features: Maximum number of TF-IDF features.
            ngram_range: N-gram range for TF-IDF vectorizer.
        """
        if model_name not in MODEL_REGISTRY:
            raise ValueError(
                f"Unknown model: '{model_name}'. "
                f"Available: {list(MODEL_REGISTRY.keys())}"
            )

        self.model_name = model_name
        self.max_features = max_features
        self.ngram_range = ngram_range

        # Build pipeline
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words="english",
            sublinear_tf=True,
            dtype=np.float32,
        )

        self.classifier = MODEL_REGISTRY[model_name]()

        self.pipeline = Pipeline([
            ("tfidf", self.vectorizer),
            ("clf", self.classifier),
        ])

        self.is_fitted = False
        self.results: dict = {}

    def fit(self, texts: list[str], labels: list[int]) -> "BaselineClassifier":
        """
        Train the classifier on text data.

        Args:
            texts: List of text strings.
            labels: List of integer labels.

        Returns:
            Self (for chaining).
        """
        logger.info(f"Training {self.model_name} on {len(texts)} samples...")
        self.pipeline.fit(texts, labels)
        self.is_fitted = True
        logger.info(f"Training complete.")
        return self

    def predict(self, texts: list[str]) -> np.ndarray:
        """
        Predict labels for text data.

        Args:
            texts: List of text strings.

        Returns:
            Array of predicted labels.
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return self.pipeline.predict(texts)

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            texts: List of text strings.

        Returns:
            Array of shape (n_samples, 2) with probabilities.
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return self.pipeline.predict_proba(texts)

    def evaluate(
        self, texts: list[str], labels: list[int], split_name: str = "test",
    ) -> dict:
        """
        Evaluate the classifier and return metrics.

        Args:
            texts: List of text strings.
            labels: True labels.
            split_name: Name of the split (for logging).

        Returns:
            Dictionary with evaluation metrics.
        """
        predictions = self.predict(texts)
        probabilities = self.predict_proba(texts)[:, 1]

        results = {
            "model": self.model_name,
            "split": split_name,
            "accuracy": float(accuracy_score(labels, predictions)),
            "precision": float(precision_score(labels, predictions, zero_division=0)),
            "recall": float(recall_score(labels, predictions, zero_division=0)),
            "f1": float(f1_score(labels, predictions, zero_division=0)),
            "auc_roc": float(roc_auc_score(labels, probabilities)),
            "confusion_matrix": confusion_matrix(labels, predictions).tolist(),
            "classification_report": classification_report(
                labels, predictions, output_dict=True
            ),
        }

        self.results = results
        return results

    def get_top_features(self, n: int = 20) -> dict[str, list[tuple[str, float]]]:
        """
        Get top TF-IDF features for each class.

        Only works with logistic_regression and svm.

        Args:
            n: Number of top features to return.

        Returns:
            Dictionary with 'positive' and 'negative' feature lists.
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted.")

        feature_names = self.vectorizer.get_feature_names_out()

        if self.model_name == "logistic_regression":
            coefficients = self.classifier.coef_[0]
        elif self.model_name == "svm":
            # CalibratedClassifierCV wraps LinearSVC
            base = self.classifier.calibrated_classifiers_[0].estimator
            coefficients = base.coef_[0]
        else:
            # Random Forest doesn't have coefficients
            importances = self.classifier.feature_importances_
            top_idx = np.argsort(importances)[-n:][::-1]
            return {
                "important": [
                    (feature_names[i], float(importances[i])) for i in top_idx
                ]
            }

        # Top positive features (predict class 1)
        top_pos_idx = np.argsort(coefficients)[-n:][::-1]
        # Top negative features (predict class 0)
        top_neg_idx = np.argsort(coefficients)[:n]

        return {
            "positive": [
                (feature_names[i], float(coefficients[i])) for i in top_pos_idx
            ],
            "negative": [
                (feature_names[i], float(coefficients[i])) for i in top_neg_idx
            ],
        }

    def save_model(self, path: str) -> str:
        """
        Save the trained model to disk.

        Args:
            path: File path for the saved model.

        Returns:
            Path where model was saved.
        """
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, "wb") as f:
            pickle.dump({
                "pipeline": self.pipeline,
                "model_name": self.model_name,
                "max_features": self.max_features,
                "ngram_range": self.ngram_range,
                "is_fitted": self.is_fitted,
                "results": self.results,
            }, f)

        logger.info(f"Model saved to {save_path}")
        return str(save_path)

    @classmethod
    def load_model(cls, path: str) -> "BaselineClassifier":
        """
        Load a previously saved model.

        Args:
            path: Path to the saved model file.

        Returns:
            Loaded BaselineClassifier instance.
        """
        with open(path, "rb") as f:
            data = pickle.load(f)

        instance = cls.__new__(cls)
        instance.pipeline = data["pipeline"]
        instance.model_name = data["model_name"]
        instance.max_features = data["max_features"]
        instance.ngram_range = data["ngram_range"]
        instance.is_fitted = data["is_fitted"]
        instance.results = data.get("results", {})
        instance.vectorizer = instance.pipeline.named_steps["tfidf"]
        instance.classifier = instance.pipeline.named_steps["clf"]

        logger.info(f"Model loaded from {path}")
        return instance

    def print_results(self) -> None:
        """Print formatted evaluation results."""
        if not self.results:
            print("No results available. Run evaluate() first.")
            return

        r = self.results
        print(f"\n{'='*50}")
        print(f"  Model: {r['model']}")
        print(f"  Split: {r['split']}")
        print(f"{'='*50}")
        print(f"  Accuracy:  {r['accuracy']:.4f}")
        print(f"  Precision: {r['precision']:.4f}")
        print(f"  Recall:    {r['recall']:.4f}")
        print(f"  F1 Score:  {r['f1']:.4f}")
        print(f"  AUC-ROC:   {r['auc_roc']:.4f}")
        print(f"{'='*50}\n")

    def __repr__(self) -> str:
        status = "fitted" if self.is_fitted else "not fitted"
        return (
            f"BaselineClassifier(model='{self.model_name}', "
            f"features={self.max_features}, status={status})"
        )