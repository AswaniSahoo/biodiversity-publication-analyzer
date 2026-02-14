"""
Term Collector for Biodiversity Genomics Dictionary.

Collects domain-specific terms from multiple categories:
- Genome projects (DToL, EBP, VGP, etc.)
- Sequencing & assembly terminology
- Bioinformatics tools & databases
- Taxonomic / species terms
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def collect_genome_project_terms() -> list[str]:
    """
    Collect names and acronyms of major biodiversity genome projects.

    Returns:
        List of project-related terms.
    """
    return [
        # Darwin Tree of Life
        "Darwin Tree of Life", "DToL", "DTOL",
        # Earth BioGenome Project
        "Earth BioGenome Project", "EBP",
        # Vertebrate Genomes Project
        "Vertebrate Genomes Project", "VGP",
        # European Reference Genome Atlas
        "European Reference Genome Atlas", "ERGA",
        # i5K — insect genomes
        "i5K", "i5k",
        # Ag100Pest
        "Ag100Pest",
        # 10,000 Plant Genomes Project
        "10KP", "10000 Plant Genomes",
        # Bird 10K
        "B10K", "Bird 10K",
        # Global Invertebrate Genomics Alliance
        "GIGA", "Global Invertebrate Genomics Alliance",
        # Africa BioGenome Project
        "Africa BioGenome Project", "AfricaBP",
        # Aquatic Symbiosis Genomics
        "Aquatic Symbiosis Genomics", "ASG",
        # Wellcome Sanger Institute
        "Wellcome Sanger Institute", "Sanger Institute",
        # Genome Notes (Wellcome Open Research)
        "Genome Note", "Genome Notes",
        # Other projects
        "Tree of Life", "Biodiversity Genomics",
        "1000 Genomes", "Genome 10K", "G10K",
        "Ocean Genome Legacy",
        "Bat1K",
        "Zoonomia",
        "DNA Zoo",
    ]


def collect_sequencing_terms() -> list[str]:
    """
    Collect genomics and sequencing technology terminology.

    Returns:
        List of sequencing and assembly related terms.
    """
    return [
        # Sequencing technologies
        "PacBio", "Pacific Biosciences", "HiFi",
        "Oxford Nanopore", "ONT", "MinION", "PromethION",
        "Illumina", "short-read", "long-read",
        "Hi-C", "Hi-C sequencing", "proximity ligation",
        "10x Genomics", "10X Chromium", "linked-read",
        "Bionano", "optical mapping",
        # Assembly concepts
        "genome assembly", "de novo assembly",
        "chromosome-level", "chromosome-scale",
        "scaffold", "scaffolding", "scaffolds",
        "contig", "contigs", "contig N50",
        "N50", "L50", "NG50",
        "haplotype", "haplotype-resolved", "phased assembly",
        "primary assembly", "alternate assembly",
        "gap-free", "gapless",
        "telomere-to-telomere", "T2T",
        # Assembly tools / algorithms
        "hifiasm", "HiCanu", "Canu",
        "Flye", "wtdbg2", "Shasta",
        "Verkko", "NextDenovo",
        "SALSA", "YaHS", "ALLHiC",
        "purge_dups", "purge duplicates",
        # Quality assessment
        "BUSCO", "completeness",
        "Merqury", "k-mer",
        "QV", "quality value",
        # Curation
        "manual curation", "genome curation",
        "Pretext", "PretextView",
        "TreeVal", "rapid curation",
        "BlobToolKit", "BlobTools",
        # Annotation
        "gene annotation", "genome annotation",
        "BRAKER", "Augustus", "GeneMark",
        "repeat masking", "RepeatMasker", "RepeatModeler",
        "structural annotation", "functional annotation",
        # General genomics
        "reference genome", "whole genome sequencing", "WGS",
        "genome size", "flow cytometry",
        "karyotype", "chromosome number",
        "heterozygosity", "genome survey",
        "mitochondrial genome", "mitogenome",
        "transcriptome", "RNA-seq",
    ]


def collect_tools_databases_terms() -> list[str]:
    """
    Collect bioinformatics tools and database names.

    Returns:
        List of tool and database names.
    """
    return [
        # Primary databases
        "GenBank", "NCBI", "INSDC",
        "ENA", "European Nucleotide Archive",
        "DDBJ",
        "UniProt", "Swiss-Prot", "TrEMBL",
        # Genome browsers / portals
        "Ensembl", "UCSC Genome Browser",
        "Genome on a Tree", "GoaT",
        "Genomes on a Tree",
        "NCBI Genome",
        "Ensembl Rapid Release",
        # Alignment tools
        "BLAST", "BLAT",
        "minimap2", "BWA", "Bowtie2",
        "DIAMOND",
        # Phylogenetics
        "RAxML", "IQ-TREE", "ASTRAL",
        "OrthoFinder", "ORTHOMCL",
        # Other tools
        "Nextflow", "Snakemake",
        "Galaxy", "CyVerse",
        "samtools", "bcftools", "bedtools",
        "FastQC", "MultiQC",
        "Bandage", "GFA",
        # Standards
        "FAIR", "open access",
        "Creative Commons", "CC BY",
        "MIGS", "MIxS",
        # Europe PMC specific
        "Europe PMC", "PubMed Central", "PMC",
        "SciLite", "text mining",
    ]


def collect_species_terms() -> list[str]:
    """
    Collect taxonomic and species-related terms commonly found
    in biodiversity genomics literature.

    Returns:
        List of taxonomic terms.
    """
    return [
        # Major taxonomic groups targeted by DToL/EBP
        "Arthropoda", "Insecta", "Coleoptera", "Lepidoptera",
        "Diptera", "Hymenoptera", "Hemiptera", "Orthoptera",
        "Mollusca", "Annelida", "Bryozoa", "Cnidaria",
        "Chordata", "Vertebrata", "Mammalia", "Aves",
        "Reptilia", "Amphibia", "Actinopterygii", "Chondrichthyes",
        "Plantae", "Angiosperm", "Gymnosperm",
        "Fungi", "Ascomycota", "Basidiomycota",
        "Protista", "Chromista",
        "Nematoda", "Platyhelminthes", "Rotifera",
        "Echinodermata", "Porifera",
        # General bio terms
        "eukaryote", "eukaryotic",
        "species", "subspecies", "taxon", "taxa",
        "biodiversity", "phylogeny", "phylogenomics",
        "systematics", "taxonomy",
        "endangered species", "conservation genomics",
        "invasive species",
        "holotype", "voucher specimen",
        "type specimen", "museum specimen",
        "DNA barcode", "COI", "cytochrome oxidase",
        "ITS", "internal transcribed spacer",
        "rbcL", "matK",
        # UK-specific (DToL focus)
        "British", "UK biodiversity",
        "Natural History Museum", "NHM",
        "Royal Botanic Gardens", "Kew",
    ]


def collect_all_terms(
    sources: Optional[list[str]] = None,
) -> dict[str, list[str]]:
    """
    Collect terms from all (or specified) categories.

    Args:
        sources: List of source categories to include.
                 Options: 'genome_projects', 'sequencing_terms',
                          'tools_databases', 'species_terms'.
                 If None, collects from all sources.

    Returns:
        Dictionary mapping category names to term lists.
    """
    all_collectors = {
        "genome_projects": collect_genome_project_terms,
        "sequencing_terms": collect_sequencing_terms,
        "tools_databases": collect_tools_databases_terms,
        "species_terms": collect_species_terms,
    }

    if sources is None:
        sources = list(all_collectors.keys())

    terms = {}
    for source in sources:
        if source in all_collectors:
            terms[source] = all_collectors[source]()
            logger.info(f"Collected {len(terms[source])} terms from '{source}'")
        else:
            logger.warning(f"Unknown source: '{source}'. Skipping.")

    total = sum(len(t) for t in terms.values())
    logger.info(f"Total terms collected: {total} across {len(terms)} categories")

    return terms
