"""
Dictionary Builder for Biodiversity Genomics Terms.

Builds, saves, and loads a structured dictionary of domain-specific
terms used for matching against scientific publication text.
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

from src.dictionary.term_collector import collect_all_terms

logger = logging.getLogger(__name__)


class DictionaryBuilder:
    """Builds and manages a biodiversity genomics term dictionary."""

    def __init__(self, output_path: str = "data/dictionaries/biodiversity_terms.json"):
        """
        Initialize the dictionary builder.

        Args:
            output_path: Path where the dictionary JSON will be saved.
        """
        self.output_path = Path(output_path)
        self.dictionary: dict[str, list[str]] = {}
        self.metadata: dict = {}

    def build(self, sources: Optional[list[str]] = None) -> dict[str, list[str]]:
        """
        Build the dictionary from specified sources.

        Args:
            sources: List of source category names to include.
                     If None, includes all categories.

        Returns:
            The built dictionary mapping categories to term lists.
        """
        logger.info("Building biodiversity genomics dictionary...")

        self.dictionary = collect_all_terms(sources)

        # Deduplicate within each category (case-insensitive, preserve original)
        for category in self.dictionary:
            seen = set()
            unique_terms = []
            for term in self.dictionary[category]:
                lower = term.lower()
                if lower not in seen:
                    seen.add(lower)
                    unique_terms.append(term)
            self.dictionary[category] = unique_terms

        # Build metadata
        self.metadata = {
            "build_date": datetime.now().isoformat(),
            "sources": list(self.dictionary.keys()),
            "total_terms": sum(len(t) for t in self.dictionary.values()),
            "terms_per_category": {
                cat: len(terms) for cat, terms in self.dictionary.items()
            },
        }

        logger.info(
            f"Dictionary built: {self.metadata['total_terms']} terms "
            f"across {len(self.dictionary)} categories"
        )

        return self.dictionary

    def save(self, path: Optional[str] = None) -> str:
        """
        Save the dictionary to a JSON file.

        Args:
            path: Custom path. Uses self.output_path if None.

        Returns:
            Path where the dictionary was saved.
        """
        save_path = Path(path) if path else self.output_path
        save_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "metadata": self.metadata,
            "dictionary": self.dictionary,
        }

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"Dictionary saved to {save_path}")
        return str(save_path)

    @classmethod
    def load(cls, path: str) -> "DictionaryBuilder":
        """
        Load a previously saved dictionary.

        Args:
            path: Path to the dictionary JSON file.

        Returns:
            DictionaryBuilder instance with loaded data.
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        builder = cls(output_path=path)
        builder.dictionary = data.get("dictionary", {})
        builder.metadata = data.get("metadata", {})

        logger.info(
            f"Dictionary loaded from {path}: "
            f"{builder.metadata.get('total_terms', 0)} terms"
        )
        return builder

    def get_all_terms_flat(self) -> list[str]:
        """
        Get all terms as a flat list (across all categories).

        Returns:
            Flat list of all terms.
        """
        all_terms = []
        for terms in self.dictionary.values():
            all_terms.extend(terms)
        return all_terms

    def print_summary(self) -> None:
        """Print a formatted summary of the dictionary."""
        print("=" * 60)
        print("BIODIVERSITY GENOMICS DICTIONARY SUMMARY")
        print("=" * 60)
        print(f"Build date: {self.metadata.get('build_date', 'N/A')}")
        print(f"Total terms: {self.metadata.get('total_terms', 0)}")
        print()
        print(f"{'Category':<25} {'Terms':>8}")
        print("-" * 35)
        for cat, count in self.metadata.get("terms_per_category", {}).items():
            print(f"  {cat:<23} {count:>6}")
        print("-" * 35)
        print()

        # Show sample terms per category
        for cat, terms in self.dictionary.items():
            samples = terms[:5]
            print(f"  {cat}: {', '.join(samples)}, ...")
        print()

    def __repr__(self) -> str:
        total = self.metadata.get("total_terms", 0)
        cats = len(self.dictionary)
        return f"DictionaryBuilder(terms={total}, categories={cats})"
