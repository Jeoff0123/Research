# PIU Risk Prediction — HBN Validation Pipeline

This repository contains the machine-learning validation code for the PIU research project.

## Important data/methodology note

The previous repository version generated a synthetic 1,200-row dataset and constructed a synthetic `riskLevel` target from variables that were also used as predictors. Those files and generated benchmark artifacts have been removed from the repository.

The current pipeline requires a real participant-level HBN CSV and does not manufacture observations or performance results.

## Current workflow

1. Load the real participant-level HBN dataset.
2. Construct `PIU_Severity` from `PCIAT_Total` using the documented score bands:
   - 0–30: None
   - 31–49: Mild
   - 50–79: Moderate
   - 80–100: Severe
3. Exclude all PCIAT item variables, `PCIAT_Total`, identifiers, and PCIAT-derived engineered variables from the predictor matrix.
4. Split data using an 80/20 stratified holdout.
5. Perform imputation, encoding, and scaling inside a scikit-learn pipeline so preprocessing is learned only from training folds.
6. Tune Logistic Regression, Decision Tree, and Random Forest models using 5-fold stratified cross-validation and macro-F1.
7. Evaluate once on the untouched test set using accuracy, macro precision, macro recall, macro-F1, and macro one-vs-rest ROC-AUC.
8. Generate confusion matrices, ROC curves, and a model-comparison chart from the measured results.

## Running the analysis

Place the real participant-level CSV beside `model_validation.py` and name it:

`hbn_piu_participant_data.csv`

Then run:

`python model_validation.py`

The script will create:

- `model_results.csv`
- `confusion_matrices_hbn.png`
- `roc_curves_hbn.png`
- `model_comparison_hbn.png`

These generated outputs are ignored by Git so that results are not accidentally committed without the corresponding run and data version.

## Important interpretation

This pipeline is designed to test whether non-PCIAT participant characteristics can predict independently defined PIU severity. It is not a reconstruction of the PCIAT score.

The final thesis should report the actual dataset version, inclusion/exclusion criteria, missing-data handling, class distribution, model-selection procedure, held-out test results, and limitations. Do not reuse performance numbers from the removed synthetic benchmark.
