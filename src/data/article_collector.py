"""
Article Collector for Biodiversity Publication Analyzer.

Collects positive (biodiversity genomics) and negative (non-biodiversity)
articles from Europe PMC, labels them, deduplicates, and saves as CSV.
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
from tqdm import tqdm

from src.data.europepmc_client import EuropePMCClient

logger = logging.getLogger(__name__)


class ArticleCollector:
    """Collects and labels articles from Europe PMC."""

    def __init__(
        self,
        client: Optional[EuropePMCClient] = None,
        max_articles_per_query: int = 500,
        min_year: int = 2018,
        max_year: int = 2025,
    ):
        """
        Initialize the article collector.

        Args:
            client: EuropePMCClient instance. Creates one if None.
            max_articles_per_query: Max articles to collect per query.
            min_year: Earliest publication year to include.
            max_year: Latest publication year to include.
        """
        self.client = client or EuropePMCClient()
        self.max_articles_per_query = max_articles_per_query
        self.min_year = min_year
        self.max_year = max_year

    def _add_year_filter(self, query: str) -> str:
        """Add year range filter to a query."""
        return (
            f"{query} AND (FIRST_PDATE:"
            f"[{self.min_year}-01-01 TO {self.max_year}-12-31])"
        )

    def collect_articles_for_query(
        self, query: str, label: int, max_results: Optional[int] = None,
    ) -> list[dict]:
        """
        Collect articles for a single query and assign a label.

        Args:
            query: Europe PMC search query.
            label: Class label (1 = positive, 0 = negative).
            max_results: Override max articles for this query.

        Returns:
            List of article metadata dictionaries with label.
        """
        max_results = max_results or self.max_articles_per_query
        filtered_query = self._add_year_filter(query)

        raw_articles = self.client.search_articles(
            filtered_query, max_results=max_results
        )

        articles = []
        for raw in raw_articles:
            meta = self.client.extract_article_metadata(raw)
            # Skip articles without title or abstract
            if not meta["title"] or not meta["abstract"]:
                continue
            meta["label"] = label
            meta["query"] = query
            articles.append(meta)

        return articles

    def collect_positive_articles(
        self, queries: list[str],
    ) -> list[dict]:
        """
        Collect biodiversity genomics articles (label = 1).

        Args:
            queries: List of positive search queries.

        Returns:
            List of labeled article dictionaries.
        """
        all_articles = []
        print(f"\n📗 Collecting POSITIVE articles ({len(queries)} queries)...")

        for query in tqdm(queries, desc="Positive queries"):
            articles = self.collect_articles_for_query(query, label=1)
            all_articles.extend(articles)
            logger.info(f"  [{query[:50]}...] → {len(articles)} articles")

        print(f"  Total positive (raw): {len(all_articles)}")
        return all_articles

    def collect_negative_articles(
        self, queries: list[str],
    ) -> list[dict]:
        """
        Collect non-biodiversity articles (label = 0).

        Args:
            queries: List of negative search queries.

        Returns:
            List of labeled article dictionaries.
        """
        all_articles = []
        print(f"\n📕 Collecting NEGATIVE articles ({len(queries)} queries)...")

        for query in tqdm(queries, desc="Negative queries"):
            articles = self.collect_articles_for_query(query, label=0)
            all_articles.extend(articles)
            logger.info(f"  [{query[:50]}...] → {len(articles)} articles")

        print(f"  Total negative (raw): {len(all_articles)}")
        return all_articles

    @staticmethod
    def deduplicate(articles: list[dict]) -> list[dict]:
        """
        Remove duplicate articles based on PMID, PMCID, or DOI.

        Priority: PMID > DOI > PMCID > title hash.

        Args:
            articles: List of article dictionaries.

        Returns:
            Deduplicated list.
        """
        seen_ids = set()
        unique = []

        for article in articles:
            # Build a unique identifier
            uid = None
            if article.get("pmid"):
                uid = f"pmid:{article['pmid']}"
            elif article.get("doi"):
                uid = f"doi:{article['doi']}"
            elif article.get("pmcid"):
                uid = f"pmcid:{article['pmcid']}"
            else:
                # Fallback: use title hash
                uid = f"title:{hash(article.get('title', '').lower().strip())}"

            if uid not in seen_ids:
                seen_ids.add(uid)
                unique.append(article)

        removed = len(articles) - len(unique)
        if removed > 0:
            logger.info(f"Deduplication: removed {removed} duplicates")

        return unique

    @staticmethod
    def balance_dataset(
        positive: list[dict], negative: list[dict],
    ) -> tuple[list[dict], list[dict]]:
        """
        Balance positive and negative sets by downsampling the larger class.

        Args:
            positive: Positive articles.
            negative: Negative articles.

        Returns:
            Tuple of (balanced_positive, balanced_negative).
        """
        import random

        min_size = min(len(positive), len(negative))

        if len(positive) > min_size:
            positive = random.sample(positive, min_size)
        if len(negative) > min_size:
            negative = random.sample(negative, min_size)

        return positive, negative

    def collect_and_save(
        self,
        positive_queries: list[str],
        negative_queries: list[str],
        output_path: str = "data/raw/articles.csv",
        balance: bool = True,
    ) -> pd.DataFrame:
        """
        Full collection pipeline: collect, deduplicate, balance, save.

        Args:
            positive_queries: List of positive search queries.
            negative_queries: List of negative search queries.
            output_path: Path to save the CSV file.
            balance: Whether to balance classes.

        Returns:
            DataFrame with all collected articles.
        """
        print("🧬 Article Collection Pipeline")
        print("=" * 50)

        # Collect
        positive = self.collect_positive_articles(positive_queries)
        negative = self.collect_negative_articles(negative_queries)

        # Deduplicate within each class
        positive = self.deduplicate(positive)
        negative = self.deduplicate(negative)
        print(f"\n  After dedup — Positive: {len(positive)}, Negative: {len(negative)}")

        # Balance
        if balance:
            positive, negative = self.balance_dataset(positive, negative)
            print(f"  After balance — Positive: {len(positive)}, Negative: {len(negative)}")

        # Combine
        all_articles = positive + negative
        df = pd.DataFrame(all_articles)

        # Save
        save_path = Path(output_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path, index=False)

        print(f"\n📊 Dataset Summary:")
        print(f"  Total articles: {len(df)}")
        print(f"  Positive (1):   {len(df[df['label'] == 1])}")
        print(f"  Negative (0):   {len(df[df['label'] == 0])}")
        print(f"  Columns:        {list(df.columns)}")
        print(f"\n✅ Saved to: {save_path}")

        return df

    def __repr__(self) -> str:
        return (
            f"ArticleCollector(max_per_query={self.max_articles_per_query}, "
            f"years={self.min_year}-{self.max_year})"
        )