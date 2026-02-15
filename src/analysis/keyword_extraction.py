"""
Keyword Extraction for Biodiversity Genomics Publications.

Extracts important terms using TF-IDF weighting and
dictionary-based frequency analysis.
"""

import logging
from collections import Counter
from typing import Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)


def extract_tfidf_keywords(
    texts: list[str],
    n: int = 50,
    max_features: int = 5000,
    ngram_range: tuple = (1, 2),
) -> list[tuple[str, float]]:
    """
    Extract top keywords using TF-IDF weighting.

    Args:
        texts: List of text strings.
        n: Number of top keywords to return.
        max_features: Max features for TF-IDF.
        ngram_range: N-gram range.

    Returns:
        List of (keyword, score) tuples sorted by score.
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        stop_words="english",
        sublinear_tf=True,
    )

    tfidf_matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()

    # Average TF-IDF score across all documents
    avg_scores = np.asarray(tfidf_matrix.mean(axis=0)).flatten()

    # Get top n
    top_indices = avg_scores.argsort()[-n:][::-1]

    return [
        (feature_names[i], float(avg_scores[i]))
        for i in top_indices
    ]


def extract_dictionary_keywords(
    texts: list[str],
    matcher,
    n: int = 50,
) -> list[tuple[str, int]]:
    """
    Extract top keywords by matching against the biodiversity dictionary.

    Args:
        texts: List of text strings.
        matcher: DictionaryMatcher instance.
        n: Number of top keywords.

    Returns:
        List of (term, count) tuples sorted by frequency.
    """
    all_matches = Counter()

    for text in texts:
        matches = matcher.match_text(text)
        for terms in matches.values():
            for term in terms:
                all_matches[term] += 1

    return all_matches.most_common(n)


def compare_positive_negative_keywords(
    positive_texts: list[str],
    negative_texts: list[str],
    n: int = 30,
) -> dict:
    """
    Compare top keywords between positive and negative articles.

    Args:
        positive_texts: Biodiversity genomics texts.
        negative_texts: Non-biodiversity texts.
        n: Number of top keywords per class.

    Returns:
        Dictionary with positive-only, negative-only, and shared keywords.
    """
    pos_keywords = dict(extract_tfidf_keywords(positive_texts, n=n * 2))
    neg_keywords = dict(extract_tfidf_keywords(negative_texts, n=n * 2))

    pos_set = set(list(pos_keywords.keys())[:n])
    neg_set = set(list(neg_keywords.keys())[:n])

    return {
        "positive_distinctive": [
            (k, pos_keywords[k]) for k in sorted(pos_set - neg_set,
            key=lambda x: pos_keywords[x], reverse=True)
        ][:n],
        "negative_distinctive": [
            (k, neg_keywords[k]) for k in sorted(neg_set - pos_set,
            key=lambda x: neg_keywords[x], reverse=True)
        ][:n],
        "shared": list(pos_set & neg_set),
    }