"""
CLI script to build the biodiversity genomics dictionary.

Usage:
    python scripts/build_dictionary.py
    python scripts/build_dictionary.py --output data/dictionaries/custom.json
"""

import argparse
import yaml
import logging

from src.dictionary.dictionary_builder import DictionaryBuilder

logging.basicConfig(level=logging.INFO, format="%(message)s")


def main():
    parser = argparse.ArgumentParser(
        description="Build biodiversity genomics dictionary"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for dictionary JSON (overrides config)",
    )
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    dict_config = config.get("dictionary", {})
    output_path = args.output or dict_config.get(
        "output_path", "data/dictionaries/biodiversity_terms.json"
    )
    sources = dict_config.get("sources", None)

    print("Building Biodiversity Genomics Dictionary")
    print("=" * 50)

    # Build
    builder = DictionaryBuilder(output_path=output_path)
    builder.build(sources=sources)

    # Print summary
    builder.print_summary()

    # Save
    saved_path = builder.save()
    print(f"\nDictionary saved to: {saved_path}")


if __name__ == "__main__":
    main()
