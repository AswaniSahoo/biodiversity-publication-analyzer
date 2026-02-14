"""
CLI script to collect articles from Europe PMC.

Usage:
    python -m scripts.collect_articles
    python -m scripts.collect_articles --config configs/default.yaml
    python -m scripts.collect_articles --max-per-query 100
"""

import argparse
import yaml
import logging

from src.data.europepmc_client import EuropePMCClient
from src.data.article_collector import ArticleCollector

logging.basicConfig(level=logging.INFO, format="%(message)s")


def main():
    parser = argparse.ArgumentParser(
        description="Collect biodiversity and non-biodiversity articles from Europe PMC"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--max-per-query",
        type=int,
        default=None,
        help="Override max articles per query (for quick testing, e.g. 50)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Override output CSV path",
    )
    parser.add_argument(
        "--no-balance",
        action="store_true",
        help="Don't balance positive/negative classes",
    )
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    data_config = config.get("data", {})
    api_config = config.get("api", {})

    # Setup client
    client = EuropePMCClient(
        base_url=api_config.get("base_url", "https://www.ebi.ac.uk/europepmc/webservices/rest"),
        page_size=api_config.get("page_size", 100),
        rate_limit_delay=api_config.get("rate_limit_delay", 0.5),
        max_retries=api_config.get("max_retries", 3),
    )

    # Setup collector
    max_per_query = args.max_per_query or data_config.get("max_articles_per_query", 500)
    collector = ArticleCollector(
        client=client,
        max_articles_per_query=max_per_query,
        min_year=data_config.get("min_year", 2018),
        max_year=data_config.get("max_year", 2025),
    )

    # Queries
    positive_queries = data_config.get("positive_queries", [])
    negative_queries = data_config.get("negative_queries", [])

    # Output path
    output_path = args.output or "data/raw/articles.csv"

    print(f"⚙️  Config: {args.config}")
    print(f"   Max per query: {max_per_query}")
    print(f"   Positive queries: {len(positive_queries)}")
    print(f"   Negative queries: {len(negative_queries)}")
    print(f"   Year range: {collector.min_year}-{collector.max_year}")
    print(f"   Output: {output_path}")
    print()

    # Run collection
    df = collector.collect_and_save(
        positive_queries=positive_queries,
        negative_queries=negative_queries,
        output_path=output_path,
        balance=not args.no_balance,
    )

    print(f"\n🎉 Collection complete! {len(df)} articles saved.")


if __name__ == "__main__":
    main()