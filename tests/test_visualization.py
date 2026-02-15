"""Tests for visualization functions."""

import os
import tempfile
import pytest
import numpy as np

from src.visualization.plot_trends import (
    plot_publications_timeline,
    plot_cumulative_growth,
    plot_journal_distribution,
    plot_model_comparison,
    plot_confusion_matrix,
)


@pytest.fixture
def yearly_counts():
    return {2020: 10, 2021: 15, 2022: 25, 2023: 40, 2024: 55}


@pytest.fixture
def top_journals():
    return [
        {"journal": "Nature", "count": 50, "percentage": 25.0},
        {"journal": "Science", "count": 30, "percentage": 15.0},
        {"journal": "PLOS ONE", "count": 20, "percentage": 10.0},
    ]


@pytest.fixture
def comparison_rows():
    return [
        {"model": "LogReg", "type": "baseline",
         "accuracy": 0.95, "precision": 0.96, "recall": 0.94, "f1": 0.95, "auc_roc": 0.98},
        {"model": "SciBERT", "type": "transformer",
         "accuracy": 0.97, "precision": 0.98, "recall": 0.96, "f1": 0.97, "auc_roc": 0.99},
    ]


class TestPlotPublicationsTimeline:
    def test_creates_file(self, yearly_counts, tmp_path):
        path = str(tmp_path / "timeline.png")
        plot_publications_timeline(yearly_counts, save_path=path)
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0

    def test_no_save(self, yearly_counts):
        """Should not raise when save_path is None."""
        plot_publications_timeline(yearly_counts, save_path=None)


class TestPlotCumulativeGrowth:
    def test_creates_file(self, yearly_counts, tmp_path):
        path = str(tmp_path / "cumulative.png")
        plot_cumulative_growth(yearly_counts, save_path=path)
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0


class TestPlotJournalDistribution:
    def test_creates_file(self, top_journals, tmp_path):
        path = str(tmp_path / "journals.png")
        plot_journal_distribution(top_journals, save_path=path)
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0


class TestPlotModelComparison:
    def test_creates_file(self, comparison_rows, tmp_path):
        path = str(tmp_path / "comparison.png")
        plot_model_comparison(comparison_rows, save_path=path)
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0


class TestPlotConfusionMatrix:
    def test_creates_file(self, tmp_path):
        cm = [[95, 5], [3, 97]]
        path = str(tmp_path / "cm.png")
        plot_confusion_matrix(cm, model_name="TestModel", save_path=path)
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0

    def test_custom_title(self, tmp_path):
        cm = [[100, 0], [0, 100]]
        path = str(tmp_path / "cm_perfect.png")
        plot_confusion_matrix(cm, model_name="PerfectModel", save_path=path)
        assert os.path.exists(path)
