"""
Impact Metrics for Biodiversity Genomics Publications.

Computes citation analysis, journal spread, open access rates,
and other impact indicators.
"""

import logging
from collections import Counter
from typing import Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def compute_citation_metrics(df: pd.DataFrame) -> dict:
    """
    Compute citation statistics for articles.

    Args:
        df: DataFrame with 'cited_by_count' column.

    Returns:
        Dictionary with citation statistics.
    """
    citations = pd.to_numeric(df["cited_by_count"], errors="coerce").fillna(0)

    return {
        "total_citations": int(citations.sum()),
        "mean_citations": float(citations.mean()),
        "median_citations": float(citations.median()),
        "max_citations": int(citations.max()),
        "min_citations": int(citations.min()),
        "std_citations": float(citations.std()),
        "articles_with_citations": int((citations > 0).sum()),
        "articles_without_citations": int((citations == 0).sum()),
        "highly_cited_10plus": int((citations >= 10).sum()),
        "highly_cited_50plus": int((citations >= 50).sum()),
    }


def compute_journal_spread(df: pd.DataFrame, top_n: int = 15) -> dict:
    """
    Analyze journal distribution.

    Args:
        df: DataFrame with 'journal' column.
        top_n: Number of top journals to return.

    Returns:
        Dictionary with journal statistics.
    """
    journals = df["journal"].dropna().astype(str).str.strip()
    journals = journals[(journals != "") & (journals != "nan")]

    journal_counts = Counter(journals)
    total = len(journals)

    top_journals = journal_counts.most_common(top_n)

    return {
        "total_journals": len(journal_counts),
        "total_articles_with_journal": total,
        "top_journals": [
            {"journal": name, "count": count, "percentage": round(count / max(total, 1) * 100, 1)}
            for name, count in top_journals
        ],
    }


def compute_open_access_rate(df: pd.DataFrame) -> dict:
    """
    Compute open access statistics.

    Args:
        df: DataFrame with 'is_open_access' column.

    Returns:
        Dictionary with OA statistics.
    """
    oa = df["is_open_access"]

    # Handle different representations
    if oa.dtype == bool:
        oa_count = oa.sum()
    else:
        oa_count = (oa.astype(str).str.lower().isin(["true", "y", "yes", "1"])).sum()

    total = len(oa)

    return {
        "total_articles": total,
        "open_access_count": int(oa_count),
        "closed_access_count": int(total - oa_count),
        "open_access_rate": round(oa_count / max(total, 1) * 100, 1),
    }


def compute_yearly_stats(df: pd.DataFrame) -> list[dict]:
    """
    Compute per-year publication statistics.

    Args:
        df: DataFrame with 'year' and 'cited_by_count' columns.

    Returns:
        List of per-year stat dictionaries.
    """
    df = df.copy()
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["year"])
    df["year"] = df["year"].astype(int)
    df["cited_by_count"] = pd.to_numeric(df["cited_by_count"], errors="coerce").fillna(0)

    yearly = []
    for year in sorted(df["year"].unique()):
        year_df = df[df["year"] == year]
        yearly.append({
            "year": int(year),
            "count": len(year_df),
            "mean_citations": float(year_df["cited_by_count"].mean()),
            "total_citations": int(year_df["cited_by_count"].sum()),
        })

    return yearly


def compute_all_impact_metrics(
    df: pd.DataFrame,
    positive_only: bool = True,
) -> dict:
    """
    Compute all impact metrics for a dataset.

    Args:
        df: Full dataset DataFrame.
        positive_only: If True, only analyze biodiversity articles (label=1).

    Returns:
        Dictionary with all impact metrics.
    """
    if positive_only and "label" in df.columns:
        analysis_df = df[df["label"] == 1].copy()
        logger.info(f"Analyzing {len(analysis_df)} positive articles")
    else:
        analysis_df = df.copy()
        logger.info(f"Analyzing {len(analysis_df)} total articles")

    return {
        "dataset_size": len(analysis_df),
        "citations": compute_citation_metrics(analysis_df),
        "journals": compute_journal_spread(analysis_df),
        "open_access": compute_open_access_rate(analysis_df),
        "yearly_stats": compute_yearly_stats(analysis_df),
    }