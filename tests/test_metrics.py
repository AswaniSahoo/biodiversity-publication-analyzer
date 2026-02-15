"""Tests for impact metrics and analysis functions."""

import pytest
import pandas as pd

from src.analysis.impact_metrics import (
    compute_citation_metrics,
    compute_journal_spread,
    compute_open_access_rate,
    compute_yearly_stats,
)
from src.analysis.trend_analysis import (
    publications_per_year,
    compute_growth_rate,
    compute_trend_summary,
)
from src.analysis.keyword_extraction import extract_tfidf_keywords


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "title": ["Article A", "Article B", "Article C", "Article D", "Article E"],
        "abstract": ["genome assembly", "clinical trial", "biodiversity", "drug study", "genome note"],
        "text": [
            "genome assembly PacBio butterfly",
            "clinical trial drug randomized",
            "biodiversity conservation genomics",
            "drug study placebo treatment",
            "genome note wellcome lepidoptera",
        ],
        "journal": ["Nature", "Lancet", "Nature", "BMJ", "Wellcome Open Res"],
        "year": [2020, 2021, 2022, 2022, 2023],
        "cited_by_count": [10, 50, 5, 100, 0],
        "is_open_access": [True, False, True, False, True],
        "label": [1, 0, 1, 0, 1],
    })


class TestCitationMetrics:
    def test_basic_stats(self, sample_df):
        metrics = compute_citation_metrics(sample_df)
        assert metrics["total_citations"] == 165
        assert metrics["max_citations"] == 100
        assert metrics["min_citations"] == 0
        assert metrics["articles_with_citations"] == 4
        assert metrics["articles_without_citations"] == 1

    def test_highly_cited(self, sample_df):
        metrics = compute_citation_metrics(sample_df)
        assert metrics["highly_cited_10plus"] == 3
        assert metrics["highly_cited_50plus"] == 2


class TestJournalSpread:
    def test_unique_journals(self, sample_df):
        result = compute_journal_spread(sample_df)
        assert result["total_journals"] == 4

    def test_top_journals(self, sample_df):
        result = compute_journal_spread(sample_df, top_n=3)
        assert len(result["top_journals"]) <= 3
        assert result["top_journals"][0]["journal"] == "Nature"
        assert result["top_journals"][0]["count"] == 2


class TestOpenAccess:
    def test_oa_rate(self, sample_df):
        result = compute_open_access_rate(sample_df)
        assert result["open_access_count"] == 3
        assert result["closed_access_count"] == 2
        assert result["open_access_rate"] == 60.0


class TestYearlyStats:
    def test_yearly_counts(self, sample_df):
        stats = compute_yearly_stats(sample_df)
        assert len(stats) >= 3
        years = [s["year"] for s in stats]
        assert 2022 in years

    def test_yearly_citations(self, sample_df):
        stats = compute_yearly_stats(sample_df)
        y2022 = [s for s in stats if s["year"] == 2022][0]
        assert y2022["count"] == 2
        assert y2022["total_citations"] == 105


class TestTrendAnalysis:
    def test_publications_per_year(self, sample_df):
        yearly = publications_per_year(sample_df)
        assert yearly.sum() == 5

    def test_trend_summary(self, sample_df):
        summary = compute_trend_summary(sample_df)
        assert "total_publications" in summary
        assert "peak_year" in summary
        assert "average_growth_rate" in summary
        assert summary["total_publications"] == 5


class TestKeywordExtraction:
    def test_tfidf_keywords(self, sample_df):
        texts = sample_df["text"].tolist()
        keywords = extract_tfidf_keywords(texts, n=5)
        assert len(keywords) <= 5
        assert all(isinstance(k, tuple) and len(k) == 2 for k in keywords)
        assert all(isinstance(k[1], float) for k in keywords)