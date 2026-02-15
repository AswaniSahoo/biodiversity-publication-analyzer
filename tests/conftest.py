"""Pytest configuration and shared fixtures."""

import pytest


@pytest.fixture
def sample_article():
    """A sample biodiversity genomics article."""
    return {
        "title": "The genome sequence of the garden tiger moth, Arctia caja",
        "abstract": (
            "We present a genome assembly of the garden tiger moth, "
            "Arctia caja (Lepidoptera; Erebidae), generated using "
            "PacBio HiFi long reads and Hi-C proximity ligation data. "
            "The genome was assembled using hifiasm and scaffolded with "
            "YaHS. BUSCO completeness was 98.2%. This is part of the "
            "Darwin Tree of Life project."
        ),
        "journal": "Wellcome Open Research",
        "year": "2024",
        "cited_by_count": 0,
        "label": 1,
    }


@pytest.fixture
def sample_negative_article():
    """A sample non-biodiversity article."""
    return {
        "title": "Randomized trial of aspirin for cardiovascular prevention",
        "abstract": (
            "A double-blind, placebo-controlled, randomized clinical trial "
            "evaluated the effects of low-dose aspirin on cardiovascular "
            "events in elderly patients. Primary endpoint was a composite "
            "of myocardial infarction, stroke, and cardiovascular death."
        ),
        "journal": "The Lancet",
        "year": "2023",
        "cited_by_count": 150,
        "label": 0,
    }