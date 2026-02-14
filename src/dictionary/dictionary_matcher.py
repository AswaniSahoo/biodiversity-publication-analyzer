"""
Dictionary Matcher for Biodiversity Genomics Publications.

Matches article text (title + abstract) against the biodiversity
dictionary to compute relevance scores and feature vectors.
"""

import re
import math
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class DictionaryMatcher:
    """Matches text against a biodiversity genomics dictionary."""

    def __init__(self, dictionary: dict[str, list[str]]):
        """
        Initialize the matcher with a dictionary.

        Args:
            dictionary: Dictionary mapping categories to term lists.
        """
        self.dictionary = dictionary
        # Pre-compile regex patterns for each term (case-insensitive)
        self._patterns = {}
        for category, terms in dictionary.items():
            self._patterns[category] = []
            for term in terms:
                # Escape special regex characters, match as whole word when possible
                escaped = re.escape(term)
                # For short terms (<=4 chars), require word boundaries
                if len(term) <= 4:
                    pattern = re.compile(r"\b" + escaped + r"\b", re.IGNORECASE)
                else:
                    pattern = re.compile(escaped, re.IGNORECASE)
                self._patterns[category].append((term, pattern))

    def match_text(self, text: str) -> dict[str, list[str]]:
        """
        Find all dictionary terms present in the text.

        Args:
            text: Input text (e.g., title + abstract).

        Returns:
            Dictionary mapping categories to lists of matched terms.
        """
        matches = {}
        for category, patterns in self._patterns.items():
            found = []
            for term, pattern in patterns:
                if pattern.search(text):
                    found.append(term)
            if found:
                matches[category] = found
        return matches

    def match_article(self, article: dict) -> dict[str, list[str]]:
        """
        Match dictionary terms against an article's title and abstract.

        Args:
            article: Article dictionary with 'title' and 'abstract' fields.

        Returns:
            Dictionary mapping categories to matched terms.
        """
        title = article.get("title", "")
        abstract = article.get("abstract", "")
        combined = f"{title} {abstract}"
        return self.match_text(combined)

    def compute_match_counts(self, text: str) -> dict[str, int]:
        """
        Compute the number of matched terms per category.

        Args:
            text: Input text.

        Returns:
            Dictionary mapping categories to match counts.
        """
        matches = self.match_text(text)
        return {cat: len(terms) for cat, terms in matches.items()}

    def compute_feature_vector(self, text: str) -> dict[str, float]:
        """
        Compute a feature vector for the text based on dictionary matches.

        Features include:
        - Per-category match counts
        - Total match count
        - Match score (0-1, normalized by total dictionary size)

        Args:
            text: Input text.

        Returns:
            Feature dictionary.
        """
        counts = self.compute_match_counts(text)
        total_matches = sum(counts.values())
        total_terms = sum(len(terms) for terms in self.dictionary.values())

        features = {}
        for category in self.dictionary:
            features[f"match_{category}"] = counts.get(category, 0)

        features["match_total"] = total_matches
        features["match_score"] = total_matches / max(total_terms, 1)
        features["categories_matched"] = sum(1 for c in counts.values() if c > 0)

        return features

    def compute_relevance_score(self, text: str) -> float:
        """
        Compute a simple relevance score (0 to 1).

        Higher score = more biodiversity genomics terms found.
        Score is weighted: more categories matched = higher score.

        Args:
            text: Input text.

        Returns:
            Relevance score between 0 and 1.
        """
        counts = self.compute_match_counts(text)
        if not counts:
            return 0.0

        total_categories = len(self.dictionary)
        categories_matched = sum(1 for c in counts.values() if c > 0)
        total_matches = sum(counts.values())

        # Category coverage (0 to 1)
        coverage = categories_matched / max(total_categories, 1)

        # Match density (log-scaled, capped at 1)
        density = min(1.0, math.log1p(total_matches) / math.log1p(20))

        # Weighted combination
        score = 0.4 * coverage + 0.6 * density

        return round(score, 4)

    def __repr__(self) -> str:
        total = sum(len(terms) for terms in self.dictionary.values())
        return f"DictionaryMatcher(terms={total}, categories={len(self.dictionary)})"
