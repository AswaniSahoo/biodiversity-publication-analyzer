"""
Trend Analysis for Biodiversity Genomics Publications.

Analyzes publication trends, growth rates, and temporal patterns.
"""

import logging
from typing import Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def publications_per_year(df: pd.DataFrame) -> pd.Series:
    """
    Count publications per year.

    Args:
        df: DataFrame with 'year' column.

    Returns:
        Series with year as index and count as values.
    """
    years = pd.to_numeric(df["year"], errors="coerce").dropna().astype(int)
    return years.value_counts().sort_index()


def compute_growth_rate(yearly_counts: pd.Series) -> pd.Series:
    """
    Compute year-over-year growth rate.

    Args:
        yearly_counts: Series with year counts.

    Returns:
        Series with growth rates (percentage).
    """
    return yearly_counts.pct_change() * 100


def cumulative_publications(df: pd.DataFrame) -> pd.Series:
    """
    Compute cumulative publication count over time.

    Args:
        df: DataFrame with 'year' column.

    Returns:
        Series with cumulative counts per year.
    """
    yearly = publications_per_year(df)
    return yearly.cumsum()


def compute_trend_summary(df: pd.DataFrame) -> dict:
    """
    Compute a summary of publication trends.

    Args:
        df: DataFrame with 'year' column.

    Returns:
        Dictionary with trend summary statistics.
    """
    yearly = publications_per_year(df)
    growth = compute_growth_rate(yearly)
    cumulative = cumulative_publications(df)

    return {
        "yearly_counts": yearly.to_dict(),
        "growth_rates": {
            int(k): round(v, 1) for k, v in growth.dropna().items()
        },
        "cumulative": cumulative.to_dict(),
        "total_publications": int(yearly.sum()),
        "year_range": f"{int(yearly.index.min())}–{int(yearly.index.max())}",
        "peak_year": int(yearly.idxmax()),
        "peak_count": int(yearly.max()),
        "average_per_year": round(float(yearly.mean()), 1),
        "average_growth_rate": round(float(growth.dropna().mean()), 1),
    }