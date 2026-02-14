# 🧬 Biodiversity Publication Analyzer

> NLP-powered tool to discover, classify, and analyze the impact of biodiversity genomics publications using Europe PMC API.

**Built as preparation for GSoC 2026 — Wellcome Sanger Institute / EMBL-EBI**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🎯 Project Overview

This project builds an end-to-end NLP pipeline to:

1. **Discover** biodiversity genomics publications from Europe PMC (Darwin Tree of Life, Earth BioGenome Project, etc.)
2. **Classify** articles as biodiversity-genomics-related or not using ML models
3. **Analyze** the impact and trends of these publications over time

### Pipeline

```
Europe PMC API → Data Collection → Dictionary Matching → Feature Extraction
    → Baseline Classifiers (TF-IDF + LogReg/SVM/RF)
    → Transformer Classifier (SciBERT fine-tuned)
    → Impact Analysis (citations, trends, journals)
    → Visualizations & Reports
```

---

## 🏗️ Project Structure

```
biodiversity-publication-analyzer/
├── configs/
│   └── default.yaml              # All hyperparameters & settings
├── src/
│   ├── data/                     # API client, collection, preprocessing
│   ├── dictionary/               # Term collection, dictionary building & matching
│   ├── models/                   # Baseline (TF-IDF) + Transformer classifiers
│   ├── analysis/                 # Impact metrics, trends, keywords
│   └── visualization/            # Plots: trends, classification, wordclouds
├── notebooks/                    # Step-by-step exploration & analysis
├── scripts/                      # CLI entry points
├── tests/                        # Unit tests
├── data/                         # Raw & processed data
└── results/                      # Figures, models, reports
```

---

## 📊 Results

*Coming soon — model training in progress.*

---

## 🚀 Quick Start

```bash
# Clone
git clone https://github.com/AswaniSahoo/biodiversity-publication-analyzer.git
cd biodiversity-publication-analyzer

# Install
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt

# Collect data
python scripts/collect_articles.py --config configs/default.yaml

# Train baseline
python scripts/train_baseline.py --config configs/default.yaml

# Train transformer
python scripts/train_transformer.py --config configs/default.yaml

# Analyze impact
python scripts/analyze_impact.py --config configs/default.yaml
```

---

## 📚 References

1. **Europe PMC** — [europepmc.org](https://europepmc.org/)
2. **Darwin Tree of Life** — [darwintreeoflife.org](https://www.darwintreeoflife.org/)
3. **Earth BioGenome Project** — [earthbiogenome.org](https://www.earthbiogenome.org/)
4. **SciBERT** — Beltagy et al. (2019) — [arXiv:1903.10676](https://arxiv.org/abs/1903.10676)
5. **WeatherBench2** — Rasp et al. (2023)

---

## 👤 Author

**Aswani Sahoo** — [@AswaniSahoo](https://github.com/AswaniSahoo)

---

## 📝 License

MIT License — see [LICENSE](LICENSE) for details.

---

<p align="center">
  <i>Built with ❤️ as preparation for GSoC 2026</i>
</p>
