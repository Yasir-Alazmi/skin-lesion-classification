# Data Directory

This directory holds the HAM10000 dataset. **No images or metadata are committed to the repository** (see `.gitignore`).

---

## Download Instructions

### Option A — Kaggle CLI (Recommended)

```bash
pip install kaggle

# Place your kaggle.json API token in ~/.kaggle/kaggle.json
kaggle datasets download -d kmader/skin-lesion-analysis-toward-melanoma-detection
unzip skin-lesion-analysis-toward-melanoma-detection.zip -d data/raw/
```

### Option B — Harvard Dataverse

Download manually from:
> https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T

---

## Expected Directory Structure

After download, `data/raw/` should look like:

```
data/
├── raw/
│   ├── HAM10000_metadata.csv          ← main label file
│   ├── HAM10000_images_part_1/        ← 5,000 dermoscopic .jpg images
│   │   ├── ISIC_0024306.jpg
│   │   └── ...
│   └── HAM10000_images_part_2/        ← 5,015 dermoscopic .jpg images
│       ├── ISIC_0029306.jpg
│       └── ...
└── processed/                         ← auto-generated cache (hair-removed images)
```

---

## Dataset Statistics

| Property | Value |
|----------|-------|
| Total images | 10,015 |
| Image size | variable (~600×450 px) |
| Resized to | 224×224 (during training) |
| Classes | 7 |
| License | CC BY-NC 4.0 |

### Class Distribution

| Class | Code | Count | % |
|-------|------|-------|---|
| Melanocytic Nevi | nv | 6,705 | 66.9% |
| Melanoma | mel | 1,113 | 11.1% |
| Benign Keratosis | bkl | 1,099 | 11.0% |
| Basal Cell Carcinoma | bcc | 514 | 5.1% |
| Actinic Keratoses | akiec | 327 | 3.3% |
| Vascular Lesions | vasc | 142 | 1.4% |
| Dermatofibroma | df | 115 | 1.1% |

---

## Citation

```bibtex
@article{tschandl2018ham10000,
  title   = {The HAM10000 dataset, a large collection of multi-source
             dermatoscopic images of common pigmented skin lesions},
  author  = {Tschandl, Philipp and Rosendahl, Cliff and Kittler, Harald},
  journal = {Scientific Data},
  volume  = {5},
  pages   = {180161},
  year    = {2018},
  doi     = {10.1038/sdata.2018.161}
}
```
