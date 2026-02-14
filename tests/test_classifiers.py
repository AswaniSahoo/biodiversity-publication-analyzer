"""Tests for baseline classifiers."""

import pytest
import numpy as np

from src.models.baseline_classifier import BaselineClassifier


@pytest.fixture
def sample_data():
    """Create simple training data."""
    texts = [
        "genome assembly of butterfly using PacBio HiFi Darwin Tree of Life",
        "reference genome sequencing of insect species biodiversity genomics",
        "chromosome-level assembly scaffold N50 BUSCO completeness genome",
        "Earth BioGenome Project vertebrate genome assembly annotation",
        "Hi-C scaffolding genome curation PacBio long-read sequencing",
        "genome note wellcome open research lepidoptera insecta",
        "clinical trial drug treatment randomized placebo diabetes",
        "epidemiology cohort study cardiovascular disease risk factors",
        "psychology behavior cognitive therapy mental health intervention",
        "cell culture in vitro experiment protein expression western blot",
        "randomized controlled trial aspirin heart disease prevention",
        "clinical drug study patients hospital treatment medication",
    ]
    labels = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
    return texts, labels


class TestBaselineClassifier:
    """Tests for the BaselineClassifier class."""

    def test_initialization(self):
        clf = BaselineClassifier(model_name="logistic_regression")
        assert clf.model_name == "logistic_regression"
        assert clf.is_fitted is False

    def test_invalid_model_name(self):
        with pytest.raises(ValueError):
            BaselineClassifier(model_name="invalid_model")

    def test_fit_logistic_regression(self, sample_data):
        texts, labels = sample_data
        clf = BaselineClassifier(model_name="logistic_regression", max_features=100)
        clf.fit(texts, labels)
        assert clf.is_fitted is True

    def test_fit_svm(self, sample_data):
        texts, labels = sample_data
        clf = BaselineClassifier(model_name="svm", max_features=100)
        clf.fit(texts, labels)
        assert clf.is_fitted is True

    def test_fit_random_forest(self, sample_data):
        texts, labels = sample_data
        clf = BaselineClassifier(model_name="random_forest", max_features=100)
        clf.fit(texts, labels)
        assert clf.is_fitted is True

    def test_predict_returns_labels(self, sample_data):
        texts, labels = sample_data
        clf = BaselineClassifier(model_name="logistic_regression", max_features=100)
        clf.fit(texts, labels)
        preds = clf.predict(texts)
        assert len(preds) == len(texts)
        assert set(preds).issubset({0, 1})

    def test_predict_proba_shape(self, sample_data):
        texts, labels = sample_data
        clf = BaselineClassifier(model_name="logistic_regression", max_features=100)
        clf.fit(texts, labels)
        proba = clf.predict_proba(texts)
        assert proba.shape == (len(texts), 2)
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_predict_before_fit_raises(self):
        clf = BaselineClassifier()
        with pytest.raises(RuntimeError):
            clf.predict(["test text"])

    def test_evaluate_returns_metrics(self, sample_data):
        texts, labels = sample_data
        clf = BaselineClassifier(model_name="logistic_regression", max_features=100)
        clf.fit(texts, labels)
        results = clf.evaluate(texts, labels)
        assert "accuracy" in results
        assert "precision" in results
        assert "recall" in results
        assert "f1" in results
        assert "auc_roc" in results
        assert 0 <= results["accuracy"] <= 1
        assert 0 <= results["auc_roc"] <= 1

    def test_get_top_features(self, sample_data):
        texts, labels = sample_data
        clf = BaselineClassifier(model_name="logistic_regression", max_features=100)
        clf.fit(texts, labels)
        features = clf.get_top_features(n=5)
        assert "positive" in features
        assert "negative" in features
        assert len(features["positive"]) <= 5

    def test_save_and_load(self, sample_data, tmp_path):
        texts, labels = sample_data
        clf = BaselineClassifier(model_name="logistic_regression", max_features=100)
        clf.fit(texts, labels)
        clf.evaluate(texts, labels)

        path = str(tmp_path / "model.pkl")
        clf.save_model(path)

        loaded = BaselineClassifier.load_model(path)
        assert loaded.is_fitted is True
        assert loaded.model_name == "logistic_regression"

        # Predictions should match
        orig_preds = clf.predict(texts)
        loaded_preds = loaded.predict(texts)
        np.testing.assert_array_equal(orig_preds, loaded_preds)

    def test_repr(self):
        clf = BaselineClassifier(model_name="svm")
        repr_str = repr(clf)
        assert "BaselineClassifier" in repr_str
        assert "svm" in repr_str
        assert "not fitted" in repr_str