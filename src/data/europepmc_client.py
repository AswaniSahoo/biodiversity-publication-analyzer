"""
Europe PMC REST API Client.

Provides methods to search articles, retrieve metadata, citations,
references, and text-mined annotations from Europe PMC.

API Docs: https://europepmc.org/RestfulWebService
"""

import time
import logging
from typing import Optional

import requests

logger = logging.getLogger(__name__)


class EuropePMCClient:
    """Client for the Europe PMC REST API."""

    def __init__(
        self,
        base_url: str = "https://www.ebi.ac.uk/europepmc/webservices/rest",
        page_size: int = 100,
        rate_limit_delay: float = 0.5,
        max_retries: int = 3,
    ):
        """
        Initialize the Europe PMC API client.

        Args:
            base_url: Base URL for the Europe PMC REST API.
            page_size: Number of results per page (max 1000).
            rate_limit_delay: Seconds to wait between requests.
            max_retries: Maximum number of retry attempts on failure.
        """
        self.base_url = base_url.rstrip("/")
        self.page_size = min(page_size, 1000)
        self.rate_limit_delay = rate_limit_delay
        self.max_retries = max_retries
        self.session = requests.Session()

    def _make_request(self, url: str, params: dict) -> dict:
        """
        Make an API request with retry logic and rate limiting.

        Args:
            url: The full URL to request.
            params: Query parameters.

        Returns:
            JSON response as a dictionary.

        Raises:
            requests.exceptions.RequestException: If all retries fail.
        """
        for attempt in range(1, self.max_retries + 1):
            try:
                time.sleep(self.rate_limit_delay)
                response = self.session.get(url, params=params, timeout=30)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                logger.warning(
                    f"Request failed (attempt {attempt}/{self.max_retries}): {e}"
                )
                if attempt == self.max_retries:
                    raise
                time.sleep(2 ** attempt)  # Exponential backoff

    def search_articles(
        self,
        query: str,
        max_results: int = 500,
        result_type: str = "core",
    ) -> list[dict]:
        """
        Search Europe PMC for articles matching a query.

        Uses cursor-based pagination to retrieve all results up to max_results.

        Args:
            query: Europe PMC search query string.
            max_results: Maximum number of articles to retrieve.
            result_type: 'core' for full metadata, 'lite' for basic fields.

        Returns:
            List of article dictionaries with metadata.
        """
        articles = []
        cursor_mark = "*"
        url = f"{self.base_url}/search"

        while len(articles) < max_results:
            params = {
                "query": query,
                "format": "json",
                "pageSize": min(self.page_size, max_results - len(articles)),
                "cursorMark": cursor_mark,
                "resultType": result_type,
            }

            try:
                data = self._make_request(url, params)
            except requests.exceptions.RequestException:
                logger.error(f"Failed to search for query: {query}")
                break

            result_list = data.get("resultList", {}).get("result", [])
            if not result_list:
                break

            articles.extend(result_list)

            # Get next cursor
            next_cursor = data.get("nextCursorMark")
            if not next_cursor or next_cursor == cursor_mark:
                break
            cursor_mark = next_cursor

            hit_count = data.get("hitCount", 0)
            logger.info(
                f"Retrieved {len(articles)}/{min(hit_count, max_results)} articles"
            )

        return articles[:max_results]

    def get_article_by_id(
        self, article_id: str, source: str = "MED", result_type: str = "core"
    ) -> Optional[dict]:
        """
        Retrieve a single article by its ID (PMID, PMCID, or DOI).

        Args:
            article_id: The article identifier.
            source: Source database — 'MED' (PubMed), 'PMC', 'DOI', etc.
            result_type: 'core' for full metadata.

        Returns:
            Article dictionary or None if not found.
        """
        url = f"{self.base_url}/search"
        
        # Build query based on source
        if source == "DOI":
            query = f'DOI:"{article_id}"'
        elif source == "PMC":
            query = f'PMCID:"{article_id}"'
        else:
            query = f'EXT_ID:"{article_id}"'

        params = {
            "query": query,
            "format": "json",
            "pageSize": 1,
            "resultType": result_type,
        }

        try:
            data = self._make_request(url, params)
            results = data.get("resultList", {}).get("result", [])
            return results[0] if results else None
        except requests.exceptions.RequestException:
            logger.error(f"Failed to retrieve article: {article_id}")
            return None

    def get_citations(
        self,
        article_id: str,
        source: str = "MED",
        max_results: int = 500,
    ) -> list[dict]:
        """
        Get articles that cite the given article.

        Args:
            article_id: The article identifier (PMID or PMCID).
            source: Source database — 'MED' or 'PMC'.
            max_results: Maximum number of citations to retrieve.

        Returns:
            List of citing article dictionaries.
        """
        citations = []
        page = 1
        url = f"{self.base_url}/{source}/{article_id}/citations"

        while len(citations) < max_results:
            params = {
                "format": "json",
                "page": page,
                "pageSize": min(self.page_size, max_results - len(citations)),
            }

            try:
                data = self._make_request(url, params)
            except requests.exceptions.RequestException:
                logger.error(
                    f"Failed to get citations for {source}:{article_id}"
                )
                break

            citation_list = (
                data.get("citationList", {}).get("citation", [])
            )
            if not citation_list:
                break

            citations.extend(citation_list)
            
            hit_count = data.get("hitCount", 0)
            if len(citations) >= hit_count:
                break

            page += 1

        return citations[:max_results]

    def get_references(
        self,
        article_id: str,
        source: str = "MED",
        max_results: int = 500,
    ) -> list[dict]:
        """
        Get articles referenced by the given article.

        Args:
            article_id: The article identifier.
            source: Source database — 'MED' or 'PMC'.
            max_results: Maximum number of references to retrieve.

        Returns:
            List of referenced article dictionaries.
        """
        references = []
        page = 1
        url = f"{self.base_url}/{source}/{article_id}/references"

        while len(references) < max_results:
            params = {
                "format": "json",
                "page": page,
                "pageSize": min(self.page_size, max_results - len(references)),
            }

            try:
                data = self._make_request(url, params)
            except requests.exceptions.RequestException:
                logger.error(
                    f"Failed to get references for {source}:{article_id}"
                )
                break

            ref_list = (
                data.get("referenceList", {}).get("reference", [])
            )
            if not ref_list:
                break

            references.extend(ref_list)

            hit_count = data.get("hitCount", 0)
            if len(references) >= hit_count:
                break

            page += 1

        return references[:max_results]

    def get_annotations(
        self,
        article_ids: list[str],
        source: str = "MED",
        annotation_type: Optional[str] = None,
    ) -> list[dict]:
        """
        Get text-mined annotations for articles.

        Uses the Europe PMC Annotations API to retrieve named entities,
        gene/protein mentions, organisms, etc.

        Args:
            article_ids: List of article identifiers (PMIDs or PMCIDs).
            source: Source database — 'MED' or 'PMC'.
            annotation_type: Filter by type (e.g., 'Organisms', 'Gene_Proteins').

        Returns:
            List of annotation dictionaries.
        """
        annotations_url = (
            "https://www.ebi.ac.uk/europepmc/annotations_api"
            "/annotationsByArticleIds"
        )
        
        # Process in batches (API limit)
        batch_size = 10
        all_annotations = []

        for i in range(0, len(article_ids), batch_size):
            batch = article_ids[i : i + batch_size]
            id_string = ",".join(
                f"{source}:{aid}" for aid in batch
            )

            params = {
                "articleIds": id_string,
                "format": "json",
            }
            if annotation_type:
                params["type"] = annotation_type

            try:
                data = self._make_request(annotations_url, params)
                if isinstance(data, list):
                    all_annotations.extend(data)
                elif isinstance(data, dict):
                    all_annotations.append(data)
            except requests.exceptions.RequestException:
                logger.warning(
                    f"Failed to get annotations for batch starting at {i}"
                )
                continue

        return all_annotations

    def extract_article_metadata(self, article: dict) -> dict:
        """
        Extract key metadata fields from a raw Europe PMC article response.

        Args:
            article: Raw article dictionary from the API.

        Returns:
            Cleaned dictionary with standardized fields.
        """
        return {
            "pmid": article.get("pmid", ""),
            "pmcid": article.get("pmcid", ""),
            "doi": article.get("doi", ""),
            "title": article.get("title", ""),
            "abstract": article.get("abstractText", ""),
            "journal": article.get("journalTitle", ""),
            "year": article.get("pubYear", ""),
            "authors": article.get("authorString", ""),
            "affiliation": article.get("affiliation", ""),
            "cited_by_count": article.get("citedByCount", 0),
            "is_open_access": article.get("isOpenAccess", "N") == "Y",
            "source": article.get("source", ""),
            "pub_type": article.get("pubTypeList", {}).get("pubType", []),
            "keywords": article.get("keywordList", {}).get("keyword", []),
            "first_pub_date": article.get("firstPublicationDate", ""),
            "full_text_available": (
                article.get("hasTextMinedTerms", "N") == "Y"
            ),
        }

    def count_results(self, query: str) -> int:
        """
        Get the total number of results for a query without downloading.

        Args:
            query: Europe PMC search query string.

        Returns:
            Total hit count.
        """
        url = f"{self.base_url}/search"
        params = {
            "query": query,
            "format": "json",
            "pageSize": 1,
            "resultType": "lite",
        }

        try:
            data = self._make_request(url, params)
            return data.get("hitCount", 0)
        except requests.exceptions.RequestException:
            return 0

    def __repr__(self) -> str:
        return (
            f"EuropePMCClient(base_url='{self.base_url}', "
            f"page_size={self.page_size}, "
            f"rate_limit_delay={self.rate_limit_delay})"
        )
