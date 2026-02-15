"""
Transformer-based Classifier for Biodiversity Publication Classification.

Fine-tunes SciBERT (or BioBERT) for binary classification of
scientific articles as biodiversity-genomics-related or not.
"""

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from tqdm import tqdm

from src.data.dataset import TransformerDataset

logger = logging.getLogger(__name__)


class TransformerClassifier:
    """Fine-tuned transformer classifier for article classification."""

    def __init__(
        self,
        model_name: str = "allenai/scibert_scivocab_uncased",
        num_labels: int = 2,
        max_length: int = 512,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
        epochs: int = 5,
        warmup_ratio: float = 0.1,
        weight_decay: float = 0.01,
        device: Optional[str] = None,
    ):
        """
        Initialize the transformer classifier.

        Args:
            model_name: HuggingFace model name/path.
            num_labels: Number of classification labels.
            max_length: Maximum token sequence length.
            batch_size: Training batch size.
            learning_rate: Peak learning rate.
            epochs: Number of training epochs.
            warmup_ratio: Fraction of steps for LR warmup.
            weight_decay: Weight decay for AdamW.
            device: 'cuda', 'cpu', or None (auto-detect).
        """
        self.model_name = model_name
        self.num_labels = num_labels
        self.max_length = max_length
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.warmup_ratio = warmup_ratio
        self.weight_decay = weight_decay

        # Device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Load tokenizer and model
        print(f"Loading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels
        )
        self.model.to(self.device)

        self.training_history: list[dict] = []
        self.best_val_f1: float = 0.0
        self.results: dict = {}

    def _create_dataset(self, texts: list[str], labels: list[int]) -> TransformerDataset:
        """Create a TransformerDataset from texts and labels."""
        return TransformerDataset(
            texts=texts,
            labels=labels,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
        )

    def train(
        self,
        train_texts: list[str],
        train_labels: list[int],
        val_texts: list[str],
        val_labels: list[int],
        save_dir: str = "results/models/transformer",
    ) -> list[dict]:
        """
        Fine-tune the transformer model.

        Args:
            train_texts: Training texts.
            train_labels: Training labels.
            val_texts: Validation texts.
            val_labels: Validation labels.
            save_dir: Directory to save the best model.

        Returns:
            Training history (list of per-epoch metrics).
        """
        # Create datasets and dataloaders
        train_dataset = self._create_dataset(train_texts, train_labels)
        val_dataset = self._create_dataset(val_texts, val_labels)

        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False
        )

        # Optimizer
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        # Scheduler
        total_steps = len(train_loader) * self.epochs
        warmup_steps = int(total_steps * self.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        # Training loop
        print(f"\n🚀 Training {self.model_name} for {self.epochs} epochs")
        print(f"   Device: {self.device}")
        print(f"   Train samples: {len(train_dataset)}")
        print(f"   Val samples: {len(val_dataset)}")
        print(f"   Batch size: {self.batch_size}")
        print(f"   Learning rate: {self.learning_rate}")
        print()

        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        self.training_history = []
        self.best_val_f1 = 0.0

        for epoch in range(1, self.epochs + 1):
            # --- Training ---
            self.model.train()
            total_train_loss = 0
            train_preds = []
            train_true = []

            progress = tqdm(
                train_loader,
                desc=f"Epoch {epoch}/{self.epochs} [Train]",
                leave=False,
            )

            for batch in progress:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                optimizer.zero_grad()
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )

                loss = outputs.loss
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                total_train_loss += loss.item()
                preds = torch.argmax(outputs.logits, dim=1)
                train_preds.extend(preds.cpu().numpy())
                train_true.extend(labels.cpu().numpy())

                progress.set_postfix(loss=f"{loss.item():.4f}")

            avg_train_loss = total_train_loss / len(train_loader)
            train_acc = accuracy_score(train_true, train_preds)

            # --- Validation ---
            val_metrics = self._evaluate_loader(val_loader)
            val_loss = val_metrics["loss"]
            val_acc = val_metrics["accuracy"]
            val_f1 = val_metrics["f1"]

            # Track history
            epoch_data = {
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "val_f1": val_f1,
                "val_precision": val_metrics["precision"],
                "val_recall": val_metrics["recall"],
                "val_auc_roc": val_metrics["auc_roc"],
                "lr": scheduler.get_last_lr()[0],
            }
            self.training_history.append(epoch_data)

            # Save best model
            star = ""
            if val_f1 > self.best_val_f1:
                self.best_val_f1 = val_f1
                self.model.save_pretrained(save_path / "best_model")
                self.tokenizer.save_pretrained(save_path / "best_model")
                star = " ⭐"

            print(
                f"Epoch {epoch:2d}/{self.epochs} | "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val F1: {val_f1:.4f} | "
                f"Val Acc: {val_acc:.4f}{star}"
            )

        # Save training history
        history_path = save_path / "training_history.json"
        with open(history_path, "w") as f:
            json.dump(self.training_history, f, indent=2)

        print(f"\n✅ Training complete! Best Val F1: {self.best_val_f1:.4f}")
        print(f"   Best model saved to: {save_path / 'best_model'}")

        return self.training_history

    def _evaluate_loader(self, dataloader: DataLoader) -> dict:
        """
        Evaluate the model on a DataLoader.

        Args:
            dataloader: DataLoader to evaluate on.

        Returns:
            Dictionary with evaluation metrics.
        """
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )

                total_loss += outputs.loss.item()

                probs = torch.softmax(outputs.logits, dim=1)
                preds = torch.argmax(probs, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())

        avg_loss = total_loss / len(dataloader)

        return {
            "loss": avg_loss,
            "accuracy": float(accuracy_score(all_labels, all_preds)),
            "precision": float(precision_score(all_labels, all_preds, zero_division=0)),
            "recall": float(recall_score(all_labels, all_preds, zero_division=0)),
            "f1": float(f1_score(all_labels, all_preds, zero_division=0)),
            "auc_roc": float(roc_auc_score(all_labels, all_probs)),
            "confusion_matrix": confusion_matrix(all_labels, all_preds).tolist(),
            "predictions": all_preds,
            "probabilities": all_probs,
            "true_labels": all_labels,
        }

    def evaluate(
        self, texts: list[str], labels: list[int], split_name: str = "test",
    ) -> dict:
        """
        Evaluate the model on text data.

        Args:
            texts: List of text strings.
            labels: True labels.
            split_name: Name of the split.

        Returns:
            Dictionary with evaluation metrics.
        """
        dataset = self._create_dataset(texts, labels)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        results = self._evaluate_loader(loader)
        results["model"] = self.model_name
        results["split"] = split_name

        self.results = results
        return results

    def predict(self, texts: list[str]) -> list[int]:
        """
        Predict labels for texts.

        Args:
            texts: List of text strings.

        Returns:
            List of predicted labels.
        """
        dataset = self._create_dataset(texts, [0] * len(texts))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        self.model.eval()
        all_preds = []

        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids, attention_mask=attention_mask
                )
                preds = torch.argmax(outputs.logits, dim=1)
                all_preds.extend(preds.cpu().numpy().tolist())

        return all_preds

    def predict_proba(self, texts: list[str]) -> list[float]:
        """
        Predict probability of positive class.

        Args:
            texts: List of text strings.

        Returns:
            List of probabilities for class 1.
        """
        dataset = self._create_dataset(texts, [0] * len(texts))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        self.model.eval()
        all_probs = []

        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids, attention_mask=attention_mask
                )
                probs = torch.softmax(outputs.logits, dim=1)[:, 1]
                all_probs.extend(probs.cpu().numpy().tolist())

        return all_probs

    def load_best_model(self, save_dir: str = "results/models/transformer") -> None:
        """
        Load the best saved model from training.

        Args:
            save_dir: Directory where the best model was saved.
        """
        model_path = Path(save_dir) / "best_model"
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path, num_labels=self.num_labels
        )
        self.model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        print(f"✅ Loaded best model from {model_path}")

    def print_results(self) -> None:
        """Print formatted evaluation results."""
        if not self.results:
            print("No results. Run evaluate() first.")
            return

        r = self.results
        print(f"\n{'='*50}")
        print(f"  Model: {r.get('model', self.model_name)}")
        print(f"  Split: {r.get('split', 'unknown')}")
        print(f"{'='*50}")
        print(f"  Accuracy:  {r['accuracy']:.4f}")
        print(f"  Precision: {r['precision']:.4f}")
        print(f"  Recall:    {r['recall']:.4f}")
        print(f"  F1 Score:  {r['f1']:.4f}")
        print(f"  AUC-ROC:   {r['auc_roc']:.4f}")
        print(f"{'='*50}\n")

    def __repr__(self) -> str:
        return (
            f"TransformerClassifier(model='{self.model_name}', "
            f"device={self.device}, epochs={self.epochs})"
        )