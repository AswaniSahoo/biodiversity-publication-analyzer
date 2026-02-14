"""Tests for the biodiversity dictionary module."""

import json
import pytest
from pathlib import Path

from src.dictionary.term_collector import (
    collect_genome_project_terms,
    collect_sequencing_terms,
    collect_tools_databases_terms,
    collect_species_terms,
    collect_all_terms,
)
from src.dictionary.dictionary_builder import DictionaryBuilder
from src.dictionary.dictionary_matcher import DictionaryMatcher


class TestTermCollector:
    """Tests for term collection functions."""

    def test_genome_projects_not_empty(self):
        terms = collect_genome_project_terms()
        assert len(terms) > 10
        assert "Darwin Tree of Life" in terms
        assert "EBP" in terms

    def test_sequencing_terms_not_empty(self):
        terms = collect_sequencing_terms()
        assert len(terms) > 20
        assert "PacBio" in terms
        assert "genome assembly" in terms
        assert "BUSCO" in terms

    def test_tools_databases_not_empty(self):
        terms = collect_tools_databases_terms()
        assert len(terms) > 10
        assert "GenBank" in terms
        assert "INSDC" in terms

    def test_species_terms_not_empty(self):
        terms = collect_species_terms()
        assert len(terms) > 10
        assert "Lepidoptera" in terms
        assert "biodiversity" in terms

    def test_collect_all_terms_returns_all_categories(self):
        all_terms = collect_all_terms()
        assert len(all_terms) == 4
        assert "genome_projects" in all_terms
        assert "sequencing_terms" in all_terms
        assert "tools_databases" in all_terms
        assert "species_terms" in all_terms

    def test_collect_specific_sources(self):
        terms = collect_all_terms(sources=["genome_projects"])
        assert len(terms) == 1
        assert "genome_projects" in terms


class TestDictionaryBuilder:
    """Tests for the DictionaryBuilder class."""

    def test_build_creates_dictionary(self):
        builder = DictionaryBuilder()
        result = builder.build()
        assert len(result) > 0
        assert builder.metadata["total_terms"] > 50

    def test_build_deduplicates(self):
        builder = DictionaryBuilder()
        builder.build()
        for terms in builder.dictionary.values():
            lower_terms = [t.lower() for t in terms]
            assert len(lower_terms) == len(set(lower_terms))

    def test_save_and_load(self, tmp_path):
        builder = DictionaryBuilder(output_path=str(tmp_path / "test.json"))
        builder.build()
        saved = builder.save()

        loaded = DictionaryBuilder.load(saved)
        assert loaded.metadata["total_terms"] == builder.metadata["total_terms"]
        assert len(loaded.dictionary) == len(builder.dictionary)

    def test_get_all_terms_flat(self):
        builder = DictionaryBuilder()
        builder.build()
        flat = builder.get_all_terms_flat()
        assert len(flat) == builder.metadata["total_terms"]

    def test_repr(self):
        builder = DictionaryBuilder()
        builder.build()
        repr_str = repr(builder)
        assert "DictionaryBuilder" in repr_str
        assert "terms=" in repr_str


class TestDictionaryMatcher:
    """Tests for the DictionaryMatcher class."""

    @pytest.fixture
    def matcher(self):
        builder = DictionaryBuilder()
        builder.build()
        return DictionaryMatcher(builder.dictionary)

    def test_match_positive_text(self, matcher):
        text = (
            "The genome assembly of the butterfly (Lepidoptera) was produced "
            "using PacBio HiFi reads and Hi-C scaffolding as part of the "
            "Darwin Tree of Life project. BUSCO completeness was 97.5%."
        )
        matches = matcher.match_text(text)
        assert len(matches) > 0
        assert "genome_projects" in matches
        assert "Darwin Tree of Life" in matches["genome_projects"]

    def test_match_negative_text(self, matcher):
        text = (
            "A randomized controlled clinical trial of aspirin for "
            "cardiovascular disease prevention in elderly patients. "
            "Results from a double-blind placebo study."
        )
        matches = matcher.match_text(text)
        # Should have very few or no matches
        total = sum(len(t) for t in matches.values())
        assert total <= 2  # Maybe "species" but not much else

    def test_compute_relevance_positive(self, matcher):
        text = (
            "Darwin Tree of Life genome assembly of Lepidoptera using "
            "PacBio and Hi-C submitted to GenBank INSDC."
        )
        score = matcher.compute_relevance_score(text)
        assert score > 0.3

    def test_compute_relevance_negative(self, matcher):
        text = "Clinical drug trial for diabetes treatment in elderly patients."
        score = matcher.compute_relevance_score(text)
        assert score < 0.15

    def test_feature_vector_keys(self, matcher):
        text = "Darwin Tree of Life genome assembly"
        features = matcher.compute_feature_vector(text)
        assert "match_total" in features
        assert "match_score" in features
        assert "categories_matched" in features
        assert "match_genome_projects" in features

    def test_match_article_dict(self, matcher):
        article = {
            "title": "Genome Note: The genome of a British butterfly",
            "abstract": "We present the genome assembly produced by the "
                        "Darwin Tree of Life project using PacBio HiFi.",
        }
        matches = matcher.match_article(article)
        assert len(matches) > 0

    def test_matcher_repr(self, matcher):
        assert "DictionaryMatcher" in repr(matcher)
