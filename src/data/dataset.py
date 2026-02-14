"""
PyTorch and HuggingFace Dataset classes for article classification.

Provides:
- ArticleDataset: PyTorch Dataset for baseline models (TF-IDF)
- TransformerDataset: PyTorch Dataset for transformer models (tokenized)
- Helper functions for creating DataLoaders
"""

import logging
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

logger = logging.getLogger(__name__)


class ArticleDataset(Dataset):
    """
    Simple PyTorch Dataset for article text and labels.

    Used with baseline models (TF-IDF + classical ML).
    """

    def __init__(self, texts: list[str], labels: list[int]):
        """
        Initialize the dataset.

        Args:
            texts: List of combined text strings.
            labels: List of integer labels (0 or 1).
        """
        assert len(texts) == len(labels), "texts and labels must have same length"
        self.texts = texts
        self.labels = labels

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> tuple[str, int]:
        return self.texts[idx], self.labels[idx]

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> "ArticleDataset":
        """
        Create dataset from a preprocessed DataFrame.

        Args:
            df: DataFrame with 'text' and 'label' columns.

        Returns:
            ArticleDataset instance.
        """
        texts = df["text"].tolist()
        labels = df["label"].astype(int).tolist()
        return cls(texts, labels)


class TransformerDataset(Dataset):
    """
    PyTorch Dataset for transformer models.

    Tokenizes text using a HuggingFace tokenizer and returns
    input_ids, attention_mask, and labels.
    """

    def __init__(
        self,
        texts: list[str],
        labels: list[int],
        tokenizer,
        max_length: int = 512,
    ):
        """
        Initialize the transformer dataset.

        Args:
            texts: List of text strings.
            labels: List of integer labels.
            tokenizer: HuggingFace tokenizer instance.
            max_length: Maximum token length.
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        tokenizer,
        max_length: int = 512,
    ) -> "TransformerDataset":
        """
        Create dataset from a preprocessed DataFrame.

        Args:
            df: DataFrame with 'text' and 'label' columns.
            tokenizer: HuggingFace tokenizer.
            max_length: Maximum token length.

        Returns:
            TransformerDataset instance.
        """
        texts = df["text"].tolist()
        labels = df["label"].astype(int).tolist()
        return cls(texts, labels, tokenizer, max_length)


def create_dataloaders(
    splits: dict[str, pd.DataFrame],
    batch_size: int = 16,
    tokenizer=None,
    max_length: int = 512,
    num_workers: int = 0,
) -> dict[str, DataLoader]:
    """
    Create DataLoaders for all splits.

    If tokenizer is provided, creates TransformerDataset loaders.
    Otherwise, creates ArticleDataset loaders.

    Args:
        splits: Dictionary with 'train', 'val', 'test' DataFrames.
        batch_size: Batch size.
        tokenizer: HuggingFace tokenizer (None for baseline).
        max_length: Max token length (for transformer).
        num_workers: DataLoader workers.

    Returns:
        Dictionary of DataLoaders.
    """
    loaders = {}

    for name, df in splits.items():
        if tokenizer is not None:
            dataset = TransformerDataset.from_dataframe(
                df, tokenizer, max_length
            )
        else:
            dataset = ArticleDataset.from_dataframe(df)

        shuffle = name == "train"
        loaders[name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
        )

        logger.info(
            f"DataLoader '{name}': {len(dataset)} samples, "
            f"batch_size={batch_size}, shuffle={shuffle}"
        )

    return loaders