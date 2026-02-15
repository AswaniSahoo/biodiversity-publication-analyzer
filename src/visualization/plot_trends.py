"""
Visualization functions for publication trends and analysis.

Generates:
- Publication timeline charts
- Growth curves
- Classification result plots (confusion matrix, ROC, PR)
- Word clouds
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

# Set style
sns.set_theme(style="whitegrid")


def plot_publications_timeline(
    yearly_counts: dict,
    title: str = "Biodiversity Genomics Publications by Year",
    save_path: Optional[str] = None,
) -> None:
    """Plot yearly publication counts as a bar chart."""
    years = sorted(yearly_counts.keys())
    counts = [yearly_counts[y] for y in years]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(years, counts, color="#2196F3", edgecolor="white", width=0.7)

    # Add value labels
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                str(count), ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Number of Publications", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xticks(years)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_cumulative_growth(
    yearly_counts: dict,
    save_path: Optional[str] = None,
) -> None:
    """Plot cumulative publication growth."""
    years = sorted(yearly_counts.keys())
    counts = [yearly_counts[y] for y in years]
    cumulative = np.cumsum(counts)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.fill_between(years, cumulative, alpha=0.3, color="#4CAF50")
    ax.plot(years, cumulative, "o-", color="#4CAF50", linewidth=2, markersize=6)

    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Cumulative Publications", fontsize=12)
    ax.set_title("Cumulative Growth of Biodiversity Genomics Publications",
                 fontsize=14, fontweight="bold")
    ax.set_xticks(years)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_journal_distribution(
    top_journals: list[dict],
    save_path: Optional[str] = None,
) -> None:
    """Plot top journals as horizontal bar chart."""
    names = [j["journal"][:40] for j in reversed(top_journals)]
    counts = [j["count"] for j in reversed(top_journals)]

    fig, ax = plt.subplots(figsize=(10, max(5, len(names) * 0.4)))
    bars = ax.barh(names, counts, color="#FF9800", edgecolor="white")

    ax.set_xlabel("Number of Articles", fontsize=12)
    ax.set_title("Top Journals — Biodiversity Genomics", fontsize=14, fontweight="bold")

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_model_comparison(
    comparison_rows: list[dict],
    save_path: Optional[str] = None,
) -> None:
    """Plot model comparison as grouped bar chart."""
    models = [r["model"] for r in comparison_rows]
    metrics = ["accuracy", "precision", "recall", "f1", "auc_roc"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1", "AUC-ROC"]

    x = np.arange(len(models))
    width = 0.15
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#F44336", "#9C27B0"]

    fig, ax = plt.subplots(figsize=(12, 5))

    for i, (metric, label, color) in enumerate(zip(metrics, metric_labels, colors)):
        values = [r[metric] for r in comparison_rows]
        offset = (i - len(metrics) / 2 + 0.5) * width
        ax.bar(x + offset, values, width, label=label, color=color, edgecolor="white")

    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Model Comparison — Test Set", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylim(0.95, 1.005)
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_confusion_matrix(
    cm: list[list[int]],
    model_name: str = "Model",
    save_path: Optional[str] = None,
) -> None:
    """Plot confusion matrix as heatmap."""
    cm_array = np.array(cm)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm_array, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Non-Biodiv", "Biodiv"],
        yticklabels=["Non-Biodiv", "Biodiv"],
        ax=ax,
    )
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=13, fontweight="bold")

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_wordcloud_from_keywords(
    keywords: list[tuple[str, float]],
    title: str = "Top Keywords",
    save_path: Optional[str] = None,
) -> None:
    """Plot a word cloud from keyword-score pairs."""
    try:
        from wordcloud import WordCloud
    except ImportError:
        logger.warning("wordcloud not installed. Skipping word cloud.")
        return

    freq_dict = {word: score for word, score in keywords}

    wc = WordCloud(
        width=800, height=400,
        background_color="white",
        colormap="viridis",
        max_words=100,
    ).generate_from_frequencies(freq_dict)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(title, fontsize=14, fontweight="bold")

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()