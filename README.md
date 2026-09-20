# PIU Risk Prediction — Kaggle HBN Validation Pipeline

This repository contains the leak-free, reproducible machine-learning validation code for the Problematic Internet Use (PIU) research project using the Kaggle Child Mind Institute Healthy Brain Network (HBN) dataset.

## Methodological Integrity & Leakage Controls

All synthetic benchmark generators and legacy pipeline files have been completely replaced. The current pipeline enforces strict target-leakage controls to ensure publication-grade validity:

1. **Target Schema**: Predicts Problematic Internet Use severity (`sii`) across 4 ordinal categories (Class 0: None, Class 1: Mild, Class 2: Moderate, Class 3: Severe).
2. **Predictor Isolation**: All 20 Parent-Child Internet Addiction Test (PCIAT) survey items (`PCIAT_01` to `PCIAT_20`), aggregate PCIAT scores (`PCIAT_Total`), and seasonal markers are **strictly excluded** from feature matrix $X$. The models predict PIU severity solely from independent physical, anthropometric, sleep, and demographic indicators.
3. **Encapsulated Preprocessing**: Preprocessing operations (median/mode imputation, standard scaling, and one-hot encoding) are encapsulated within `scikit-learn` `Pipeline` and `ColumnTransformer` objects to ensure parameters are learned exclusively from training folds.

## Current Workflow

1. Load the real participant-level dataset (`train.csv`).
2. Clean missing target rows (`sii`) and isolate non-PCIAT predictor features.
3. Execute a stratified 80/20 holdout train-test split ($N = 966$ train / $N = 242$ test).
4. Perform 5-fold Stratified Cross-Validation (`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`) and hyperparameter grid search (`GridSearchCV`) optimizing for Macro F1-score across:
   - Logistic Regression ($L_2$-regularized with `class_weight='balanced'`)
   - Decision Tree Classifier (`class_weight='balanced'`)
   - Random Forest Classifier (`class_weight='balanced'`)
5. Evaluate best-fitted pipelines once on the unseen 20% holdout test partition.
6. Compute comprehensive evaluation metrics:
   - Overall Accuracy & Balanced Accuracy
   - Macro Precision, Macro Recall, and Macro F1-Score
   - Per-Class Precision, Recall, and F1-Score Breakdown
   - Quadratic Weighted Kappa (QWK)
   - Multi-Class One-vs-Rest (OVR) ROC-AUC (Macro-OVR vs. Micro-OVR)
7. Export benchmark metrics and publication-ready diagnostic figures.

## Running the Analysis

Place your dataset file in the same directory as `model_validation.py` and name it:

```text
train.csv


Execute the validation script in your environment:

Bash
python model_validation.py
To generate the comparative performance chart, run:

python generate_chart.py

Generated Artifacts
Running the pipeline creates the following output files:

model_results.csv (Raw summary metrics table)

per_class_results.csv (Per-class precision, recall, and F1 breakdown)

confusion_matrices_proof.png (3-panel confusion matrix grid with counts and row recall percentages)

roc_curves_proof.png (Multi-class One-vs-Rest ROC curves with Macro-AUC and Micro-AUC scores)

model_comparison_hbn.png (Benchmark performance comparison chart)

These artifacts are ignored by Git (.gitignore) to ensure that performance figures are generated strictly from reproducible script executions.

