"""
CLI script to run full impact analysis and generate visualizations.

Usage:
    python -m scripts.analyze_impact
"""

import json
import logging
from pathlib import Path

import pandas as pd

from src.data.preprocessing import load_splits
from src.dictionary.dictionary_builder import DictionaryBuilder
from src.dictionary.dictionary_matcher import DictionaryMatcher
from src.models.model_utils import load_all_results, build_comparison_table, print_comparison_table
from src.analysis.impact_metrics import compute_all_impact_metrics
from src.analysis.trend_analysis import compute_trend_summary
from src.analysis.keyword_extraction import (
    extract_tfidf_keywords,
    extract_dictionary_keywords,
    compare_positive_negative_keywords,
)
from src.visualization.plot_trends import (
    plot_publications_timeline,
    plot_cumulative_growth,
    plot_journal_distribution,
    plot_model_comparison,
    plot_confusion_matrix,
    plot_wordcloud_from_keywords,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
FIGURES_DIR = "results/figures"
REPORTS_DIR = "results/reports"


def main():
    print("🧬 Biodiversity Publication Analyzer — Impact Analysis")
    print("=" * 60)

    # --- 1. Load data ---
    print("\n📂 Loading data...")
    splits = load_splits("data/processed")
    if not splits:
        print("❌ No data found. Run preprocessing first.")
        return

    all_data = pd.concat(splits.values(), ignore_index=True)
    positive_data = all_data[all_data["label"] == 1]
    negative_data = all_data[all_data["label"] == 0]

    print(f"   Total: {len(all_data)}, Positive: {len(positive_data)}, Negative: {len(negative_data)}")

    # --- 2. Model Comparison ---
    print("\n📊 Model Comparison...")
    results = load_all_results(REPORTS_DIR)
    comparison = build_comparison_table(results)
    print_comparison_table(comparison)

    # Plot model comparison
    plot_model_comparison(comparison, save_path=f"{FIGURES_DIR}/model_comparison.png")

    # Plot confusion matrices
    if "baselines" in results:
        for name, data in results["baselines"].items():
            cm = data.get("test", {}).get("confusion_matrix", [[0, 0], [0, 0]])
            plot_confusion_matrix(cm, model_name=f"TF-IDF + {name}",
                                  save_path=f"{FIGURES_DIR}/cm_{name}.png")

    if "transformer" in results:
        cm = results["transformer"].get("test_results", {}).get(
            "confusion_matrix", [[0, 0], [0, 0]]
        )
        plot_confusion_matrix(cm, model_name="SciBERT",
                              save_path=f"{FIGURES_DIR}/cm_scibert.png")

    # --- 3. Impact Metrics ---
    print("\n📈 Computing impact metrics...")
    impact = compute_all_impact_metrics(all_data, positive_only=True)

    print(f"\n  Citations:")
    c = impact["citations"]
    print(f"    Mean: {c['mean_citations']:.1f}, Median: {c['median_citations']:.1f}")
    print(f"    Max: {c['max_citations']}, Highly cited (10+): {c['highly_cited_10plus']}")

    print(f"\n  Journals:")
    j = impact["journals"]
    print(f"    Unique journals: {j['total_journals']}")
    for jj in j["top_journals"][:5]:
        print(f"    • {jj['journal']}: {jj['count']} ({jj['percentage']}%)")

    print(f"\n  Open Access:")
    oa = impact["open_access"]
    print(f"    OA rate: {oa['open_access_rate']}%")

    # --- 4. Trend Analysis ---
    print("\n📅 Publication trends...")
    trend = compute_trend_summary(positive_data)
    print(f"    Year range: {trend['year_range']}")
    print(f"    Peak year: {trend['peak_year']} ({trend['peak_count']} articles)")
    print(f"    Avg growth: {trend['average_growth_rate']}%")

    # Plot trends
    plot_publications_timeline(
        trend["yearly_counts"],
        title="Biodiversity Genomics Publications (Positive Set)",
        save_path=f"{FIGURES_DIR}/publications_timeline.png",
    )
    plot_cumulative_growth(
        trend["yearly_counts"],
        save_path=f"{FIGURES_DIR}/cumulative_growth.png",
    )

    # Plot journal distribution
    if impact["journals"]["top_journals"]:
        plot_journal_distribution(
            impact["journals"]["top_journals"],
            save_path=f"{FIGURES_DIR}/journal_distribution.png",
        )

    # --- 5. Keyword Analysis ---
    print("\n🔑 Keyword extraction...")
    pos_texts = positive_data["text"].tolist()
    neg_texts = negative_data["text"].tolist()

    pos_keywords = extract_tfidf_keywords(pos_texts, n=30)
    print("  Top biodiversity keywords:")
    for kw, score in pos_keywords[:10]:
        print(f"    • {kw}: {score:.4f}")

    # Word cloud
    plot_wordcloud_from_keywords(
        pos_keywords,
        title="Biodiversity Genomics — Top Keywords",
        save_path=f"{FIGURES_DIR}/wordcloud_positive.png",
    )

    neg_keywords = extract_tfidf_keywords(neg_texts, n=30)
    plot_wordcloud_from_keywords(
        neg_keywords,
        title="Non-Biodiversity — Top Keywords",
        save_path=f"{FIGURES_DIR}/wordcloud_negative.png",
    )

    # Keyword comparison
    keyword_comparison = compare_positive_negative_keywords(pos_texts, neg_texts)
    print(f"\n  Distinctive positive keywords: {len(keyword_comparison['positive_distinctive'])}")
    print(f"  Distinctive negative keywords: {len(keyword_comparison['negative_distinctive'])}")
    print(f"  Shared keywords: {len(keyword_comparison['shared'])}")

    # --- 6. Dictionary matching stats ---
    print("\n📖 Dictionary matching...")
    try:
        builder = DictionaryBuilder()
        builder.build()
        matcher = DictionaryMatcher(builder.dictionary)

        dict_keywords = extract_dictionary_keywords(pos_texts, matcher, n=20)
        print("  Top matched biodiversity terms:")
        for term, count in dict_keywords[:10]:
            print(f"    • {term}: {count} articles")
    except Exception as e:
        print(f"  Dictionary matching skipped: {e}")

    # --- 7. Save full report ---
    full_report = {
        "dataset": {
            "total": len(all_data),
            "positive": len(positive_data),
            "negative": len(negative_data),
        },
        "model_comparison": comparison,
        "impact_metrics": impact,
        "trend_summary": trend,
        "top_keywords_positive": [(k, round(s, 4)) for k, s in pos_keywords[:30]],
        "top_keywords_negative": [(k, round(s, 4)) for k, s in neg_keywords[:30]],
    }

    Path(REPORTS_DIR).mkdir(parents=True, exist_ok=True)
    report_path = f"{REPORTS_DIR}/impact_analysis.json"
    with open(report_path, "w") as f:
        json.dump(full_report, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"✅ Impact analysis complete!")
    print(f"   Report: {report_path}")
    print(f"   Figures: {FIGURES_DIR}/")
    print(f"     • model_comparison.png")
    print(f"     • publications_timeline.png")
    print(f"     • cumulative_growth.png")
    print(f"     • journal_distribution.png")
    print(f"     • wordcloud_positive.png")
    print(f"     • wordcloud_negative.png")
    print(f"     • cm_*.png (confusion matrices)")


if __name__ == "__main__":
    main()