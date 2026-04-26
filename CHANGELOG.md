# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Planned
- Grad-CAM visualisation for model interpretability
- ONNX export support
- Streamlit inference demo

---

## [1.0.0] — 2026-04-26

### Added
- Full training pipeline for EfficientNet-B3 and ViT-B/16 on HAM10000
- Hair removal preprocessing using Black-Hat morphology + TELEA inpainting (`src/preprocess.py`)
- Focal Loss with class-frequency weighting (`src/loss.py`)
- WeightedRandomSampler for class imbalance mitigation (`src/dataset.py`)
- 2-phase fine-tuning: frozen backbone → full unfreeze at epoch 5 (`src/train.py`)
- EarlyStopping on validation AUC with patience=7 (`src/train.py`)
- Per-class evaluation: Accuracy, Precision, Recall, F1, AUC (`src/evaluate.py`)
- Visualisation suite: ROC curves, confusion matrices, training history, model comparison (`src/visualize.py`)
- Stratified 70/15/15 train/val/test split (`src/dataset.py`)
- Experiment results, plots, and prediction CSVs (`results/`)
- Comprehensive README with architecture diagram, dataset details, and usage examples
- MIT License
- `.gitignore` for Python, PyTorch, and Jupyter environments

### Results
| Model | Accuracy | Macro F1 | Macro AUC |
|-------|----------|----------|-----------|
| EfficientNet-B3 | 92.6% | 91.9% | 0.95 |
| ViT-B/16 | **94.2%** | **93.8%** | **0.96** |

---

[Unreleased]: https://github.com/Yasir-Alazmi/skin-lesion-classification/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/Yasir-Alazmi/skin-lesion-classification/releases/tag/v1.0.0
