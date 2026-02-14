"""
Data Preprocessing for Biodiversity Publication Analyzer.

Handles text cleaning, feature combination, train/val/test splitting,
and augmentation with dictionary match features.
"""

import re
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def clean_text(text: str) -> str:
    """
    Clean raw text from Europe PMC articles.

    - Remove HTML tags
    - Normalize whitespace
    - Strip leading/trailing whitespace

    Args:
        text: Raw text string.

    Returns:
        Cleaned text.
    """
    if not isinstance(text, str):
        return ""

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Remove HTML entities
    text = re.sub(r"&[a-zA-Z]+;", " ", text)
    text = re.sub(r"&#\d+;", " ", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def combine_fields(
    title: str,
    abstract: str,
    journal: str = "",
    keywords: str = "",
) -> str:
    """
    Combine article fields into a single text string for classification.

    Args:
        title: Article title.
        abstract: Article abstract.
        journal: Journal name (optional).
        keywords: Keywords (optional).

    Returns:
        Combined text string.
    """
    parts = []

    if title:
        parts.append(f"TITLE: {clean_text(title)}")
    if abstract:
        parts.append(f"ABSTRACT: {clean_text(abstract)}")
    if journal:
        parts.append(f"JOURNAL: {clean_text(journal)}")
    if keywords and keywords != "[]":
        parts.append(f"KEYWORDS: {clean_text(str(keywords))}")

    return " ".join(parts)


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess a raw articles DataFrame.

    - Clean title and abstract
    - Create combined text field
    - Convert types
    - Drop rows with missing essential fields

    Args:
        df: Raw DataFrame from article collection.

    Returns:
        Preprocessed DataFrame.
    """
    df = df.copy()

    # Clean text fields
    df["title_clean"] = df["title"].apply(clean_text)
    df["abstract_clean"] = df["abstract"].apply(clean_text)

    # Create combined text
    df["text"] = df.apply(
        lambda row: combine_fields(
            title=row.get("title", ""),
            abstract=row.get("abstract", ""),
            journal=row.get("journal", ""),
            keywords=str(row.get("keywords", "")),
        ),
        axis=1,
    )

    # Ensure label is integer
    df["label"] = df["label"].astype(int)

    # Convert year to integer
    df["year"] = pd.to_numeric(df["year"], errors="coerce").fillna(0).astype(int)

    # Convert cited_by_count
    df["cited_by_count"] = pd.to_numeric(
        df["cited_by_count"], errors="coerce"
    ).fillna(0).astype(int)

    # Text length features
    df["title_length"] = df["title_clean"].str.len()
    df["abstract_length"] = df["abstract_clean"].str.len()
    df["text_length"] = df["text"].str.len()

    # Drop rows with empty text
    df = df[df["text_length"] > 50].reset_index(drop=True)

    logger.info(f"Preprocessed {len(df)} articles")
    return df


def augment_with_dictionary(
    df: pd.DataFrame,
    matcher,
) -> pd.DataFrame:
    """
    Add dictionary match features to the DataFrame.

    Args:
        df: Preprocessed DataFrame with 'text' column.
        matcher: DictionaryMatcher instance.

    Returns:
        DataFrame with added dictionary feature columns.
    """
    df = df.copy()

    features_list = []
    for text in df["text"]:
        features = matcher.compute_feature_vector(text)
        features_list.append(features)

    features_df = pd.DataFrame(features_list)
    df = pd.concat([df, features_df], axis=1)

    # Add relevance score
    df["relevance_score"] = df["text"].apply(matcher.compute_relevance_score)

    logger.info(f"Added {len(features_df.columns)} dictionary features")
    return df


def create_splits(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
    stratify: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    Split the dataset into train, validation, and test sets.

    Args:
        df: Preprocessed DataFrame.
        train_ratio: Fraction for training.
        val_ratio: Fraction for validation.
        test_ratio: Fraction for testing.
        random_state: Random seed for reproducibility.
        stratify: Whether to stratify by label.

    Returns:
        Dictionary with 'train', 'val', 'test' DataFrames.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, (
        f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}"
    )

    from sklearn.model_selection import train_test_split

    stratify_col = df["label"] if stratify else None

    # First split: train vs (val + test)
    train_df, temp_df = train_test_split(
        df,
        test_size=(val_ratio + test_ratio),
        random_state=random_state,
        stratify=stratify_col,
    )

    # Second split: val vs test
    relative_test = test_ratio / (val_ratio + test_ratio)
    stratify_temp = temp_df["label"] if stratify else None

    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test,
        random_state=random_state,
        stratify=stratify_temp,
    )

    splits = {
        "train": train_df.reset_index(drop=True),
        "val": val_df.reset_index(drop=True),
        "test": test_df.reset_index(drop=True),
    }

    logger.info(
        f"Splits — Train: {len(splits['train'])}, "
        f"Val: {len(splits['val'])}, Test: {len(splits['test'])}"
    )

    return splits


def save_splits(
    splits: dict[str, pd.DataFrame],
    output_dir: str = "data/processed",
) -> None:
    """
    Save train/val/test splits to CSV files.

    Args:
        splits: Dictionary with 'train', 'val', 'test' DataFrames.
        output_dir: Directory to save the files.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for name, df in splits.items():
        path = output_path / f"{name}.csv"
        df.to_csv(path, index=False)
        logger.info(f"Saved {name}: {len(df)} rows → {path}")

    # Save split statistics
    stats = {
        name: {
            "size": len(df),
            "positive": int(df["label"].sum()),
            "negative": int((df["label"] == 0).sum()),
        }
        for name, df in splits.items()
    }

    import json
    stats_path = output_path / "split_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    logger.info(f"Split stats saved to {stats_path}")


def load_splits(
    input_dir: str = "data/processed",
) -> dict[str, pd.DataFrame]:
    """
    Load previously saved splits.

    Args:
        input_dir: Directory containing the split CSV files.

    Returns:
        Dictionary with 'train', 'val', 'test' DataFrames.
    """
    input_path = Path(input_dir)
    splits = {}

    for name in ["train", "val", "test"]:
        path = input_path / f"{name}.csv"
        if path.exists():
            splits[name] = pd.read_csv(path)
            logger.info(f"Loaded {name}: {len(splits[name])} rows")
        else:
            logger.warning(f"Split file not found: {path}")

    return splits