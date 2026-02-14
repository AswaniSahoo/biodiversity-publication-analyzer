"""
CLI script to train baseline classifiers.

Usage:
    python -m scripts.train_baseline
    python -m scripts.train_baseline --config configs/default.yaml
"""

import argparse
import json
import yaml
import logging
from pathlib import Path

from src.data.preprocessing import load_splits
from src.models.baseline_classifier import BaselineClassifier

logging.basicConfig(level=logging.INFO, format="%(message)s")


def main():
    parser = argparse.ArgumentParser(
        description="Train baseline TF-IDF classifiers"
    )
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--data-dir", type=str, default=None,
        help="Override data directory (default: data/processed)",
    )
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    baseline_config = config.get("baseline", {})
    paths_config = config.get("paths", {})

    data_dir = args.data_dir or paths_config.get("processed_data", "data/processed")
    models_dir = paths_config.get("models", "results/models")
    reports_dir = paths_config.get("reports", "results/reports")

    # Load splits
    print("📂 Loading data splits...")
    splits = load_splits(data_dir)

    if not splits:
        print("❌ No data splits found. Run preprocessing first.")
        return

    train_texts = splits["train"]["text"].tolist()
    train_labels = splits["train"]["label"].tolist()
    val_texts = splits["val"]["text"].tolist()
    val_labels = splits["val"]["label"].tolist()
    test_texts = splits["test"]["text"].tolist()
    test_labels = splits["test"]["label"].tolist()

    print(f"   Train: {len(train_texts)}, Val: {len(val_texts)}, Test: {len(test_texts)}")

    # Model settings
    max_features = baseline_config.get("max_features", 10000)
    ngram_range = tuple(baseline_config.get("ngram_range", [1, 2]))
    model_names = baseline_config.get("models", [
        "logistic_regression", "svm", "random_forest"
    ])

    print(f"\n🚀 Training {len(model_names)} baseline models...")
    print(f"   TF-IDF: max_features={max_features}, ngram_range={ngram_range}")
    print()

    all_results = {}

    for model_name in model_names:
        print(f"{'─'*50}")
        print(f"🔧 Training: {model_name}")

        clf = BaselineClassifier(
            model_name=model_name,
            max_features=max_features,
            ngram_range=ngram_range,
        )

        # Train
        clf.fit(train_texts, train_labels)

        # Evaluate on validation
        val_results = clf.evaluate(val_texts, val_labels, split_name="val")
        clf.print_results()

        # Evaluate on test
        test_results = clf.evaluate(test_texts, test_labels, split_name="test")
        print(f"  Test — Acc: {test_results['accuracy']:.4f}, "
              f"F1: {test_results['f1']:.4f}, "
              f"AUC: {test_results['auc_roc']:.4f}")

        # Save model
        model_path = f"{models_dir}/baseline_{model_name}.pkl"
        clf.save_model(model_path)

        # Get top features
        top_features = clf.get_top_features(n=15)

        all_results[model_name] = {
            "val": val_results,
            "test": test_results,
            "top_features": {
                k: [(term, round(score, 4)) for term, score in v]
                for k, v in top_features.items()
            },
        }

    # Print comparison table
    print(f"\n{'='*70}")
    print("📊 MODEL COMPARISON (Test Set)")
    print(f"{'='*70}")
    print(f"{'Model':<25} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'AUC':>10}")
    print(f"{'─'*75}")

    for model_name, results in all_results.items():
        r = results["test"]
        print(
            f"  {model_name:<23} "
            f"{r['accuracy']:>9.4f} "
            f"{r['precision']:>9.4f} "
            f"{r['recall']:>9.4f} "
            f"{r['f1']:>9.4f} "
            f"{r['auc_roc']:>9.4f}"
        )

    print(f"{'─'*75}")

    # Save comparison report
    Path(reports_dir).mkdir(parents=True, exist_ok=True)
    report_path = f"{reports_dir}/baseline_comparison.json"
    with open(report_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n✅ All models trained and saved!")
    print(f"   Models: {models_dir}/")
    print(f"   Report: {report_path}")


if __name__ == "__main__":
    main()