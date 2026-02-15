"""
CLI script to fine-tune a transformer classifier (SciBERT/BioBERT).

Usage:
    python -m scripts.train_transformer
    python -m scripts.train_transformer --config configs/default.yaml
    python -m scripts.train_transformer --device cuda --epochs 3
"""

import argparse
import json
import yaml
import logging
from pathlib import Path

from src.data.preprocessing import load_splits
from src.models.transformer_classifier import TransformerClassifier

logging.basicConfig(level=logging.INFO, format="%(message)s")


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune SciBERT for article classification"
    )
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device: 'cuda' or 'cpu' (auto-detect if not set)",
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Override number of epochs",
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="Override batch size",
    )
    parser.add_argument(
        "--data-dir", type=str, default=None,
        help="Override data directory",
    )
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    transformer_config = config.get("transformer", {})
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
    model_name = transformer_config.get("model_name", "allenai/scibert_scivocab_uncased")
    max_length = transformer_config.get("max_length", 512)
    batch_size = args.batch_size or transformer_config.get("batch_size", 16)
    learning_rate = transformer_config.get("learning_rate", 2e-5)
    epochs = args.epochs or transformer_config.get("epochs", 5)
    warmup_ratio = transformer_config.get("warmup_ratio", 0.1)
    weight_decay = transformer_config.get("weight_decay", 0.01)

    # Initialize classifier
    clf = TransformerClassifier(
        model_name=model_name,
        max_length=max_length,
        batch_size=batch_size,
        learning_rate=learning_rate,
        epochs=epochs,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        device=args.device,
    )

    # Train
    save_dir = f"{models_dir}/transformer"
    history = clf.train(
        train_texts=train_texts,
        train_labels=train_labels,
        val_texts=val_texts,
        val_labels=val_labels,
        save_dir=save_dir,
    )

    # Load best model and evaluate on test
    clf.load_best_model(save_dir)
    test_results = clf.evaluate(test_texts, test_labels, split_name="test")
    clf.print_results()

    # Save test results
    Path(reports_dir).mkdir(parents=True, exist_ok=True)
    report = {
        "model": model_name,
        "config": {
            "max_length": max_length,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "epochs": epochs,
            "warmup_ratio": warmup_ratio,
            "weight_decay": weight_decay,
        },
        "training_history": history,
        "test_results": {
            k: v for k, v in test_results.items()
            if k not in ["predictions", "probabilities", "true_labels"]
        },
    }

    report_path = f"{reports_dir}/transformer_results.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\n✅ Results saved to: {report_path}")
    print(f"   Model saved to: {save_dir}/best_model/")


if __name__ == "__main__":
    main()