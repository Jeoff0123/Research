"""
HBN PIU severity classification
Research-ready validation pipeline using participant-level HBN data.

IMPORTANT:
- This script does NOT generate synthetic data.
- PCIAT items and PCIAT_Total are excluded from predictors because PCIAT_Total
  defines the target severity.
- Target bands:
    0-30   = None
    31-49  = Mild
    50-79  = Moderate
    80-100 = Severe
- Put the real HBN participant-level CSV at DATASET_PATH before running.

Outputs:
- model_results.csv
- confusion_matrices_hbn.png
- roc_curves_hbn.png
- model_comparison_hbn.png
"""

from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc,
)
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, label_binarize
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
TEST_SIZE = 0.20
DATASET_PATH = Path("hbn_piu_participant_data.csv")

TARGET = "PIU_Severity"
TARGET_ORDER = ["None", "Mild", "Moderate", "Severe"]

PCIAT_PATTERN = re.compile(r"^PCIAT(?:_|-)", re.IGNORECASE)
EXCLUDED_EXACT = {
    "id", "ID", "subjectkey", "SubjectKey", "participant_id", "ParticipantID",
    "riskLevel", "PIU_Severity", "PCIAT_Total",
    "Interaction_Sleep_ScreenTime", "PCIAT_Emotional_Volatility",
    "Physical_Sedentary_Ratio",
}


def find_column(df, candidates):
    lookup = {str(c).lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    return None


def build_target(df):
    total_col = find_column(df, ["PCIAT_Total", "PCIAT-Total"])
    if total_col is None:
        raise ValueError(
            "PCIAT_Total was not found. Supply the real HBN participant-level "
            "dataset containing PCIAT_Total."
        )

    total = pd.to_numeric(df[total_col], errors="coerce")
    invalid = total.notna() & ~total.between(0, 100)
    if invalid.any():
        raise ValueError(
            f"{invalid.sum()} PCIAT_Total values fall outside 0-100. "
            "Check the source data and score coding."
        )

    target = pd.Series(pd.NA, index=df.index, dtype="string")
    target.loc[total.between(0, 30)] = "None"
    target.loc[total.between(31, 49)] = "Mild"
    target.loc[total.between(50, 79)] = "Moderate"
    target.loc[total.between(80, 100)] = "Severe"

    if total.notna().sum() != target.notna().sum():
        raise ValueError("Some non-missing PCIAT scores could not be assigned a class.")

    return target


def select_predictors(df):
    excluded = set(EXCLUDED_EXACT)
    for col in df.columns:
        if PCIAT_PATTERN.match(str(col)):
            excluded.add(col)

    predictors = [
        c for c in df.columns
        if c not in excluded and not df[c].isna().all()
    ]

    if not predictors:
        raise ValueError("No predictors remain after excluding PCIAT-derived variables.")

    return predictors


def make_preprocessor(X):
    numeric_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    return ColumnTransformer([
        ("num", numeric_pipe, numeric_cols),
        ("cat", categorical_pipe, categorical_cols),
    ])


def evaluate_model(name, estimator, X_test, y_test, class_names):
    y_pred = estimator.predict(X_test)
    y_proba = estimator.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

    estimator_classes = estimator.classes_
    class_to_index = {c: i for i, c in enumerate(estimator_classes)}
    y_proba_aligned = np.zeros((len(y_test), len(class_names)))
    for j, cls in enumerate(class_names):
        if cls in class_to_index:
            y_proba_aligned[:, j] = y_proba[:, class_to_index[cls]]

    y_test_bin = label_binarize(y_test, classes=class_names)
    auc_macro = roc_auc_score(
        y_test_bin, y_proba_aligned, multi_class="ovr", average="macro"
    )

    return {
        "Model": name,
        "Accuracy": acc,
        "Precision_macro": precision,
        "Recall_macro": recall,
        "F1_macro": f1,
        "ROC_AUC_macro_OVR": auc_macro,
        "y_pred": y_pred,
        "y_proba": y_proba_aligned,
    }


def main():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"{DATASET_PATH} was not found. Replace DATASET_PATH with the path "
            "to the real HBN participant-level CSV. The old repository CSV was "
            "synthetic and must not be used as evidence of HBN performance."
        )

    df = pd.read_csv(DATASET_PATH)
    if df.empty:
        raise ValueError("The supplied dataset is empty.")

    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns):,}")

    df[TARGET] = build_target(df)
    df = df.loc[df[TARGET].notna()].copy()

    predictors = select_predictors(df)
    X = df[predictors].copy()
    y = pd.Series(df[TARGET].astype(str), index=df.index)

    observed_classes = [c for c in TARGET_ORDER if c in set(y)]
    if len(observed_classes) < 2:
        raise ValueError("At least two PIU severity classes are required.")

    class_counts = y.value_counts().reindex(observed_classes, fill_value=0)
    print("\nTarget distribution:")
    print(class_counts.to_string())
    print(f"\nPredictors used: {len(predictors)}")
    print("PCIAT items and PCIAT_Total are excluded from X.")

    if int(class_counts.min()) < 6:
        raise ValueError("The smallest target class has fewer than 6 observations; "
                         "5-fold stratified CV is not reliable.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    preprocessor = make_preprocessor(X_train)

    models = {
        "Logistic Regression": (
            LogisticRegression(max_iter=3000, random_state=RANDOM_STATE),
            {
                "model__C": [0.01, 0.1, 1.0, 10.0],
                "model__class_weight": [None, "balanced"],
            },
        ),
        "Decision Tree": (
            DecisionTreeClassifier(random_state=RANDOM_STATE),
            {
                "model__max_depth": [3, 5, 10, None],
                "model__criterion": ["gini", "entropy"],
                "model__class_weight": [None, "balanced"],
            },
        ),
        "Random Forest": (
            RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
            {
                "model__n_estimators": [100, 200, 400],
                "model__max_depth": [5, 10, None],
                "model__class_weight": [None, "balanced"],
                "model__max_features": ["sqrt", "log2"],
            },
        ),
    }

    results = []
    fitted = {}

    for name, (model, param_grid) in models.items():
        pipeline = Pipeline([
            ("preprocess", preprocessor),
            ("model", model),
        ])

        grid = GridSearchCV(
            pipeline, param_grid, scoring="f1_macro", cv=cv, n_jobs=-1, refit=True
        )
        grid.fit(X_train, y_train)

        evaluated = evaluate_model(
            name, grid.best_estimator_, X_test, y_test, observed_classes
        )
        fitted[name] = (grid.best_estimator_, evaluated)

        results.append({
            "Model": name,
            "CV_best_macro_F1": grid.best_score_,
            "Test_Accuracy": evaluated["Accuracy"],
            "Test_Precision_macro": evaluated["Precision_macro"],
            "Test_Recall_macro": evaluated["Recall_macro"],
            "Test_F1_macro": evaluated["F1_macro"],
            "Test_ROC_AUC_macro_OVR": evaluated["ROC_AUC_macro_OVR"],
            "Best_Params": str(grid.best_params_),
        })

        print(f"\n{name}")
        print(f"Best CV macro-F1: {grid.best_score_:.4f}")
        print(f"Best parameters: {grid.best_params_}")

    results_df = pd.DataFrame(results)
    results_df.to_csv("model_results.csv", index=False)

    display_df = results_df.copy()
    metric_cols = [
        "CV_best_macro_F1", "Test_Accuracy", "Test_Precision_macro",
        "Test_Recall_macro", "Test_F1_macro", "Test_ROC_AUC_macro_OVR",
    ]
    for col in metric_cols:
        display_df[col] = display_df[col].map(lambda v: f"{v:.4f}")

    print("\n=== HELD-OUT TEST SET RESULTS ===")
    print(display_df.to_string(index=False))

    # Confusion matrices
    n_models = len(fitted)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4.5), squeeze=False)
    axes = axes.ravel()

    for ax, (name, (_, evaluated)) in zip(axes, fitted.items()):
        cm = confusion_matrix(y_test, evaluated["y_pred"], labels=observed_classes)
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=observed_classes
        )
        disp.plot(ax=ax, colorbar=False)
        ax.set_title(f"{name}\nHeld-out test set")

    plt.tight_layout()
    plt.savefig("confusion_matrices_hbn.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ROC curves are micro-averaged; the table reports macro OVR AUC.
    fig, ax = plt.subplots(figsize=(8, 6))
    y_test_bin = label_binarize(y_test, classes=observed_classes)

    for name, (_, evaluated) in fitted.items():
        fpr, tpr, _ = roc_curve(
            y_test_bin.ravel(), evaluated["y_proba"].ravel()
        )
        micro_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, linewidth=2,
                label=f"{name} (micro-AUC={micro_auc:.4f})")

    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Multi-Class ROC Curves — Held-Out Test Set")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("roc_curves_hbn.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Dynamic chart generated only from measured results.
    chart_metrics = {
        "Accuracy": "Test_Accuracy",
        "Precision": "Test_Precision_macro",
        "Recall": "Test_Recall_macro",
        "F1": "Test_F1_macro",
        "ROC-AUC": "Test_ROC_AUC_macro_OVR",
    }

    x = np.arange(len(chart_metrics))
    width = 0.8 / len(results_df)
    fig, ax = plt.subplots(figsize=(11, 6))

    for i, row in results_df.iterrows():
        values = [row[col] * 100 for col in chart_metrics.values()]
        offset = (i - (len(results_df) - 1) / 2) * width
        bars = ax.bar(x + offset, values, width, label=row["Model"])
        for bar in bars:
            ax.annotate(
                f"{bar.get_height():.1f}%",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 3), textcoords="offset points",
                ha="center", va="bottom", fontsize=8,
            )

    ax.set_ylabel("Score (%)")
    ax.set_title("Held-Out Test Performance Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(chart_metrics.keys())
    ax.set_ylim(0, 105)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig("model_comparison_hbn.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("\nSaved: model_results.csv, confusion_matrices_hbn.png, "
          "roc_curves_hbn.png, model_comparison_hbn.png")


if __name__ == "__main__":
    main()
