"""Tests for Europe PMC API client."""

import pytest
from src.data.europepmc_client import EuropePMCClient


@pytest.fixture
def client():
    """Create a client with shorter delay for testing."""
    return EuropePMCClient(rate_limit_delay=0.3, page_size=10)


class TestEuropePMCClient:
    """Tests for the EuropePMCClient class."""

    def test_client_initialization(self, client):
        """Test client initializes with correct defaults."""
        assert client.base_url == "https://www.ebi.ac.uk/europepmc/webservices/rest"
        assert client.page_size == 10
        assert client.rate_limit_delay == 0.3
        assert client.max_retries == 3

    def test_client_repr(self, client):
        """Test string representation."""
        repr_str = repr(client)
        assert "EuropePMCClient" in repr_str
        assert "page_size=10" in repr_str

    def test_search_articles_returns_results(self, client):
        """Test that searching returns articles."""
        articles = client.search_articles('"Darwin Tree of Life"', max_results=5)
        assert len(articles) > 0
        assert len(articles) <= 5

    def test_search_articles_has_required_fields(self, client):
        """Test that articles have essential metadata fields."""
        articles = client.search_articles('"genome assembly"', max_results=3)
        assert len(articles) > 0

        article = articles[0]
        # These fields should exist in core results
        assert "title" in article
        assert "source" in article

    def test_search_articles_respects_max_results(self, client):
        """Test that max_results limit is respected."""
        articles = client.search_articles("genome", max_results=3)
        assert len(articles) <= 3

    def test_count_results(self, client):
        """Test counting results without downloading."""
        count = client.count_results('"Darwin Tree of Life"')
        assert isinstance(count, int)
        assert count > 0

    def test_extract_article_metadata(self, client):
        """Test metadata extraction from raw article."""
        articles = client.search_articles('"Darwin Tree of Life"', max_results=1)
        assert len(articles) > 0

        meta = client.extract_article_metadata(articles[0])
        assert "title" in meta
        assert "abstract" in meta
        assert "journal" in meta
        assert "year" in meta
        assert "cited_by_count" in meta
        assert "is_open_access" in meta
        assert isinstance(meta["cited_by_count"], int)
        assert isinstance(meta["is_open_access"], bool)

    def test_extract_metadata_handles_missing_fields(self, client):
        """Test metadata extraction handles empty article gracefully."""
        meta = client.extract_article_metadata({})
        assert meta["title"] == ""
        assert meta["abstract"] == ""
        assert meta["cited_by_count"] == 0
        assert meta["is_open_access"] is False

    def test_search_empty_query_returns_results(self, client):
        """Test that even a broad query works."""
        articles = client.search_articles("biodiversity", max_results=2)
        assert len(articles) > 0

    def test_get_article_by_id(self, client):
        """Test retrieving a specific article by PMID."""
        # First search to get a valid PMID
        articles = client.search_articles('"Darwin Tree of Life"', max_results=1)
        if articles and articles[0].get("pmid"):
            pmid = articles[0]["pmid"]
            article = client.get_article_by_id(pmid, source="MED")
            assert article is not None
            assert article.get("pmid") == pmid
