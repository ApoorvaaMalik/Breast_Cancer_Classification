# Breast Cancer Diagnosis Classification 🔬

A machine learning project that classifies breast tumours as **Benign** or **Malignant** using the Breast Cancer Wisconsin Diagnostic dataset. The project covers exploratory data analysis, correlation-based feature selection, and classification with a Random Forest model.

---

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Methodology](#methodology)
- [Results](#results)
- [How to Run](#how-to-run)
- [Notes](#notes)
- [Author](#author)

---

## Overview

Breast cancer diagnosis from digitized cell nuclei measurements is a classic binary classification problem. This project builds an end-to-end pipeline that:

1. Loads and cleans the raw dataset
2. Explores class balance and feature distributions
3. Visualizes feature correlations
4. Removes redundant, highly-correlated features
5. Trains a Random Forest classifier on the reduced feature set
6. Evaluates the model with accuracy and a confusion matrix

The goal is not just to build a classifier, but to show the reasoning behind feature selection — which features actually help separate benign from malignant cases, and which are redundant.

## Dataset

The dataset used is `breast_cancer_data.csv` — the **Breast Cancer Wisconsin (Diagnostic)** dataset. It contains **569 samples** with **30 numeric features** computed from digitized images of fine needle aspirates (FNA) of breast masses, describing characteristics of the cell nuclei (radius, texture, perimeter, area, smoothness, compactness, concavity, symmetry, etc., each reported as mean, standard error, and "worst"/largest value).

| Column | Description |
|---|---|
| `id` | Patient/sample ID (dropped — not predictive) |
| `diagnosis` | Target label: `M` = Malignant, `B` = Benign |
| `Unnamed: 32` | Empty column artifact from the CSV export (dropped) |
| 30 feature columns | Numeric measurements of cell nuclei characteristics |

## Project Structure

```
.
├── breast_cancer_data.csv               # Dataset 
├── breast_cancer_classification.ipynb   # Main analysis & modeling notebook
├── plots/                               # Generated visualizations (EDA, heatmaps, confusion matrix)
├── requirements.txt                     # Python dependencies
└── README.md                            # Project documentation
```

## Requirements

All dependencies are listed in `requirements.txt`. Install them with:

```bash
pip install -r requirements.txt
```

Core libraries used: `numpy`, `pandas`, `seaborn`, `matplotlib`, `scikit-learn`.

## Methodology

### 1. Data Loading & Cleaning
The raw CSV is loaded with `pandas`. Non-predictive columns (`id`, `Unnamed: 32`) and the target column (`diagnosis`) are separated from the feature matrix, giving a clean feature set `x` and label vector `y`.

### 2. Exploratory Data Analysis (EDA)
- A **count plot** visualizes the class balance between Benign and Malignant samples.
- Features are standardized (z-score) and plotted in three batches of 10 using **strip plots**, split by diagnosis, to visually inspect which features separate the two classes well and which overlap heavily.

### 3. Correlation Analysis
A **correlation heatmap** across all 30 features surfaces multicollinearity — features that are highly correlated with each other and therefore carry redundant information (e.g. `radius_mean`, `perimeter_mean`, and `area_mean` all measure tumour size in different ways).

### 4. Feature Selection
Based on the correlation heatmap, 14 highly correlated features are dropped, including:
`perimeter_mean`, `radius_mean`, `compactness_mean`, `concave points_mean`, `radius_se`, `perimeter_se`, `radius_worst`, `perimeter_worst`, `compactness_worst`, `concavity_worst`, `compactness_se`, `concave points_se`, `texture_worst`, `area_worst`.

A second correlation heatmap on the reduced feature set confirms that redundancy is significantly reduced, while the most informative features for classification are retained.

### 5. Model Training
The reduced feature set is split into **80% training / 20% test** data (`random_state=21`). A `RandomForestClassifier` (`random_state=42`) is trained on the training split.

### 6. Evaluation
Model performance is assessed using:
- **Accuracy score** on the held-out test set
- **Confusion matrix**, visualized as a percentage heatmap for an intuitive read on false positives/negatives

## Results

The pipeline produces the following visual outputs, saved under `plots/`:

- Class distribution count plot (Benign vs Malignant)
- Three batches of standardized feature strip plots (mean, SE, and worst-value groups)
- Full 30-feature correlation heatmap
- Reduced 16-feature correlation heatmap
- Confusion matrix heatmap

The Random Forest classifier, trained on the 16 selected features, achieves strong classification performance in separating benign from malignant tumours, as reflected in the accuracy score and confusion matrix generated by the notebook.

## How to Run

1. Place `breast_cancer_data.csv` in the project root.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the notebook top to bottom:
   ```bash
   jupyter notebook breast_cancer_classification.ipynb
   ```
4. Plots are saved to the `plots/` folder for reference and reuse.

## Notes

- Random seeds are fixed for reproducibility: `random_state=42` for the model, `random_state=21` for the train/test split.
- Feature reduction is done via visual inspection of the correlation heatmap rather than an automated method (e.g. VIF or recursive feature elimination) — feel free to adjust `drop_list` to experiment with a different feature subset.
- This project is intended as an educational/exploratory ML pipeline rather than a clinical diagnostic tool.

## Author

**Apoorva Malik**

GitHub: [ApoorvaaMalik](https://github.com/ApoorvaaMalik)
