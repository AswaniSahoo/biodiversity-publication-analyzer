"""Tests for data preprocessing and dataset classes."""

import pytest
import pandas as pd

from src.data.preprocessing import (
    clean_text,
    combine_fields,
    preprocess_dataframe,
    create_splits,
)
from src.data.dataset import ArticleDataset


class TestCleanText:
    """Tests for text cleaning functions."""

    def test_removes_html_tags(self):
        text = "<p>Hello <b>world</b></p>"
        assert clean_text(text) == "Hello world"

    def test_removes_html_entities(self):
        text = "temperature &gt; 100&deg;C"
        assert "&gt;" not in clean_text(text)
        assert "&deg;" not in clean_text(text)

    def test_normalizes_whitespace(self):
        text = "hello   world\n\tnew  line"
        assert clean_text(text) == "hello world new line"

    def test_handles_empty_string(self):
        assert clean_text("") == ""

    def test_handles_none(self):
        assert clean_text(None) == ""

    def test_handles_numeric(self):
        assert clean_text(123) == ""

    def test_strips_whitespace(self):
        assert clean_text("  hello  ") == "hello"


class TestCombineFields:
    """Tests for field combination."""

    def test_combines_title_and_abstract(self):
        result = combine_fields("My Title", "My abstract text.")
        assert "TITLE: My Title" in result
        assert "ABSTRACT: My abstract text." in result

    def test_includes_journal(self):
        result = combine_fields("Title", "Abstract", journal="Nature")
        assert "JOURNAL: Nature" in result

    def test_empty_optional_fields(self):
        result = combine_fields("Title", "Abstract", journal="", keywords="")
        assert "JOURNAL" not in result
        assert "KEYWORDS" not in result

    def test_cleans_html_in_fields(self):
        result = combine_fields("<b>Title</b>", "<p>Abstract</p>")
        assert "<b>" not in result
        assert "<p>" not in result


class TestPreprocessDataframe:
    """Tests for DataFrame preprocessing."""

    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({
            "title": [
                "Genome assembly of butterfly",
                "<b>Clinical trial</b> for drug",
                "Short",
            ],
            "abstract": [
                "We present the genome assembly using PacBio HiFi reads.",
                "A randomized controlled trial of aspirin for heart disease.",
                "X",
            ],
            "journal": ["Nature", "Lancet", ""],
            "year": ["2023", "2024", "2022"],
            "label": [1, 0, 1],
            "cited_by_count": [10, 5, 0],
            "keywords": ["['genome']", "['trial']", "[]"],
        })

    def test_creates_text_column(self, sample_df):
        result = preprocess_dataframe(sample_df)
        assert "text" in result.columns
        assert all(len(t) > 0 for t in result["text"])

    def test_creates_clean_columns(self, sample_df):
        result = preprocess_dataframe(sample_df)
        assert "title_clean" in result.columns
        assert "abstract_clean" in result.columns

    def test_removes_html(self, sample_df):
        result = preprocess_dataframe(sample_df)
        for text in result["title_clean"]:
            assert "<b>" not in text

    def test_adds_length_features(self, sample_df):
        result = preprocess_dataframe(sample_df)
        assert "title_length" in result.columns
        assert "abstract_length" in result.columns
        assert "text_length" in result.columns

    def test_drops_short_texts(self, sample_df):
        result = preprocess_dataframe(sample_df)
        # "Short" + "X" should be dropped (text_length < 50)
        assert len(result) <= len(sample_df)

    def test_label_is_integer(self, sample_df):
        result = preprocess_dataframe(sample_df)
        assert result["label"].dtype == int


class TestCreateSplits:
    """Tests for dataset splitting."""

    @pytest.fixture
    def preprocessed_df(self):
        # Create enough data for splitting
        data = []
        for i in range(100):
            data.append({
                "text": f"This is article {i} with enough text to pass the length filter easily.",
                "label": i % 2,  # Balanced: 50 positive, 50 negative
                "title_clean": f"Title {i}",
                "abstract_clean": f"Abstract {i}",
            })
        return pd.DataFrame(data)

    def test_split_ratios(self, preprocessed_df):
        splits = create_splits(preprocessed_df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
        total = sum(len(s) for s in splits.values())
        assert total == len(preprocessed_df)

    def test_split_keys(self, preprocessed_df):
        splits = create_splits(preprocessed_df)
        assert "train" in splits
        assert "val" in splits
        assert "test" in splits

    def test_stratification(self, preprocessed_df):
        splits = create_splits(preprocessed_df, stratify=True)
        # Each split should have roughly equal class distribution
        for name, df in splits.items():
            if len(df) >= 4:
                pos_ratio = df["label"].mean()
                assert 0.3 < pos_ratio < 0.7, f"{name} not well stratified"

    def test_no_overlap(self, preprocessed_df):
        splits = create_splits(preprocessed_df)
        train_idx = set(splits["train"].index)
        val_idx = set(splits["val"].index)
        test_idx = set(splits["test"].index)
        # After reset_index, indices will be 0-based for each split
        # Check via text content instead
        train_texts = set(splits["train"]["text"])
        val_texts = set(splits["val"]["text"])
        test_texts = set(splits["test"]["text"])
        assert len(train_texts & val_texts) == 0
        assert len(train_texts & test_texts) == 0
        assert len(val_texts & test_texts) == 0


class TestArticleDataset:
    """Tests for the ArticleDataset class."""

    def test_dataset_length(self):
        ds = ArticleDataset(["text1", "text2", "text3"], [1, 0, 1])
        assert len(ds) == 3

    def test_dataset_getitem(self):
        ds = ArticleDataset(["hello world", "test"], [1, 0])
        text, label = ds[0]
        assert text == "hello world"
        assert label == 1

    def test_from_dataframe(self):
        df = pd.DataFrame({
            "text": ["article one", "article two"],
            "label": [1, 0],
        })
        ds = ArticleDataset.from_dataframe(df)
        assert len(ds) == 2
        text, label = ds[1]
        assert text == "article two"
        assert label == 0