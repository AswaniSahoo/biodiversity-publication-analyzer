"""
Model utility functions.

Comparison tables, results aggregation, and helper functions
for working across multiple trained models.
"""

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def load_all_results(reports_dir: str = "results/reports") -> dict:
    """
    Load all saved model results from the reports directory.

    Args:
        reports_dir: Path to results/reports directory.

    Returns:
        Dictionary with baseline and transformer results.
    """
    results = {}
    reports_path = Path(reports_dir)

    # Load baseline results
    baseline_path = reports_path / "baseline_comparison.json"
    if baseline_path.exists():
        with open(baseline_path) as f:
            results["baselines"] = json.load(f)

    # Load transformer results
    transformer_path = reports_path / "transformer_results.json"
    if transformer_path.exists():
        with open(transformer_path) as f:
            results["transformer"] = json.load(f)

    return results


def build_comparison_table(results: dict) -> list[dict]:
    """
    Build a comparison table from all model results.

    Args:
        results: Dictionary from load_all_results().

    Returns:
        List of dictionaries, one per model, with metrics.
    """
    rows = []

    # Baseline models
    if "baselines" in results:
        for model_name, data in results["baselines"].items():
            test = data.get("test", {})
            rows.append({
                "model": f"TF-IDF + {model_name}",
                "type": "baseline",
                "accuracy": test.get("accuracy", 0),
                "precision": test.get("precision", 0),
                "recall": test.get("recall", 0),
                "f1": test.get("f1", 0),
                "auc_roc": test.get("auc_roc", 0),
            })

    # Transformer
    if "transformer" in results:
        test = results["transformer"].get("test_results", {})
        model_name = results["transformer"].get("model", "SciBERT")
        rows.append({
            "model": model_name.split("/")[-1],
            "type": "transformer",
            "accuracy": test.get("accuracy", 0),
            "precision": test.get("precision", 0),
            "recall": test.get("recall", 0),
            "f1": test.get("f1", 0),
            "auc_roc": test.get("auc_roc", 0),
        })

    return rows


def print_comparison_table(rows: list[dict]) -> None:
    """Print a formatted comparison table."""
    print(f"\n{'='*80}")
    print("📊 ALL MODELS — TEST SET COMPARISON")
    print(f"{'='*80}")
    print(
        f"{'Model':<35} {'Accuracy':>10} {'Precision':>10} "
        f"{'Recall':>10} {'F1':>10} {'AUC':>10}"
    )
    print(f"{'─'*85}")

    for row in rows:
        print(
            f"  {row['model']:<33} "
            f"{row['accuracy']:>9.4f} "
            f"{row['precision']:>9.4f} "
            f"{row['recall']:>9.4f} "
            f"{row['f1']:>9.4f} "
            f"{row['auc_roc']:>9.4f}"
        )

    print(f"{'─'*85}")

    # Find best model
    best = max(rows, key=lambda r: r["f1"])
    print(f"\n  🏆 Best model: {best['model']} (F1 = {best['f1']:.4f})")