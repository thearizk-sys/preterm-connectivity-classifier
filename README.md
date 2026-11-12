# preterm-connectivity-classifier
# Preterm vs. Term Brain Connectivity Classifier

A machine learning classification pipeline using **Principal Component Analysis (PCA)** and **L1-regularised Logistic Regression** to predict term vs. preterm neurodevelopmental outcomes from high-dimensional neonatal brain connectivity matrices.

## Data Privacy & Reproducibility Note
To comply with clinical data governance and patient confidentiality requirements, the original neuroimaging connectivity matrices and subject metadata are **not** included in this repository. Instead, `preterm_classifier.py` includes a **synthetic cohort generator** that simulates symmetric, zero-diagonal connectivity matrices (`N × R × R`) with realistic class imbalance so the full pipeline executes out of the box.

## Methodology & Clinical ML Challenges
1. **Curse of Dimensionality (`p >> n`):** Symmetric `R × R` connectivity matrices yield thousands of inter-regional correlation edges across a modest subject cohort. The pipeline extracts the strict upper triangle (`k=1`) to eliminate redundant edges and self-connections, standardises features strictly on the training fold, and applies **PCA** alongside **L1 (Lasso) regularisation** to prevent overfitting.
2. **Severe Class Imbalance:** Because clinical cohorts contain substantially more full-term than preterm subjects, standard unregularised classifiers optimise for majority-class accuracy and collapse to `0.00` minority-class sensitivity. The enhanced model addresses this by tuning the L1 inverse regularisation strength (`C`) via stratified 5-fold `GridSearchCV` optimised for **`recall_macro`**.

## Key Findings (Original Clinical Cohort)
* **Baseline Logistic Regression:** Collapsed to **0.000 sensitivity** on unseen test data due to majority-class bias.
* **L1-Regularised Classifier (`recall_macro` GridSearchCV):** Improved cross-validated sensitivity to **0.771** and held-out test sensitivity to **0.706 (70.6%)** while maintaining strong generalisation across both classes.

## Tech Stack & Usage
* **Language:** Python 3.x
* **Libraries:** `numpy`, `pandas`, `scikit-learn`, `matplotlib`

```bash
pip install -r requirements.txt
python preterm_classifier.py
```
