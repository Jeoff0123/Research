import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, cohen_kappa_score, confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc
)

# Set seed for statistical reproducibility
np.random.seed(42)

DATASET_PATH = Path('train.csv')

# ---------------------------------------------------------
# 1. LOAD AND CLEAN KAGGLE HBN DATASET
# ---------------------------------------------------------
print(f"Loading dataset from '{DATASET_PATH}'...")
df = pd.read_csv(DATASET_PATH)

# Target variable handling
target_col = 'sii'
df = df.dropna(subset=[target_col]).copy()
df[target_col] = df[target_col].astype(int)

# Report Target Class Distribution
sii_counts = df[target_col].value_counts().sort_index()
sii_pcts = (df[target_col].value_counts(normalize=True).sort_index() * 100).round(2)

print("\n=== KAGGLE HBN TARGET ('sii') CLASS DISTRIBUTION ===")
for cls, count in sii_counts.items():
    print(f"Class {cls}: {count} participants ({sii_pcts[cls]}%)")
print("Note: Class 3 (Severe PIU) represents a minority class (< 3%). Evaluation reports QWK and Macro metrics.\n")

# ---------------------------------------------------------
# 2. FEATURE ISOLATION (LEAKAGE PREVENTION)
# ---------------------------------------------------------
# Exclude target, identifiers, and ALL PCIAT items/aggregates
pciat_cols = [c for c in df.columns if 'PCIAT' in c or 'pciat' in c.lower()]
non_predictor_cols = ['id', 'id_seq', 'PCIAT-Season', target_col] + pciat_cols

feature_cols = [c for c in df.columns if c not in non_predictor_cols]
X = df[feature_cols].copy()
y = df[target_col].copy()

# Separate Numeric and Categorical Feature Spaces
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

print(f"Independent Predictors (X): {len(feature_cols)} features ({len(num_cols)} numeric, {len(cat_cols)} categorical).")

# ---------------------------------------------------------
# 3. STRATIFIED 80/20 HOLD-OUT TRAIN-TEST SPLIT
# ---------------------------------------------------------
# Split performed BEFORE any data transformations
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

# ---------------------------------------------------------
# 4. PREPROCESSING PIPELINES (PREVENTING CV LEAKAGE)
# ---------------------------------------------------------
num_transformer_linear = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

num_transformer_tree = Pipeline([
    ('imputer', SimpleImputer(strategy='median'))
])

cat_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor_linear = ColumnTransformer([
    ('num', num_transformer_linear, num_cols),
    ('cat', cat_transformer, cat_cols)
])

preprocessor_tree = ColumnTransformer([
    ('num', num_transformer_tree, num_cols),
    ('cat', cat_transformer, cat_cols)
])

# ---------------------------------------------------------
# 5. MODEL ESTIMATORS & HYPERPARAMETER GRID
# ---------------------------------------------------------
models = {
    'Logistic Reg. (L2)': (
        Pipeline([
            ('preprocessor', preprocessor_linear),
            ('classifier', LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'))
        ]),
        {'classifier__C': [0.01, 0.1, 1.0, 10.0]}
    ),
    'Decision Tree': (
        Pipeline([
            ('preprocessor', preprocessor_tree),
            ('classifier', DecisionTreeClassifier(random_state=42, class_weight='balanced'))
        ]),
        {'classifier__max_depth': [3, 5, 10, None], 'classifier__criterion': ['gini', 'entropy']}
    ),
    'Random Forest': (
        Pipeline([
            ('preprocessor', preprocessor_tree),
            ('classifier', RandomForestClassifier(random_state=42, class_weight='balanced'))
        ]),
        {'classifier__n_estimators': [50, 100, 200], 'classifier__max_depth': [5, 10, None]}
    )
}

# ---------------------------------------------------------
# 6. MODEL EVALUATION & DIAGNOSTICS
# ---------------------------------------------------------
benchmark_results = []
results_for_csv = []
trained_pipelines = {}

classes = sorted(y.unique())

for name, (pipe, param_grid) in models.items():
    grid = GridSearchCV(pipe, param_grid, cv=5, scoring='f1_macro', n_jobs=-1)
    grid.fit(X_train, y_train)
    
    best_pipe = grid.best_estimator_
    trained_pipelines[name] = best_pipe
    
    y_pred = best_pipe.predict(X_test)
    y_proba = best_pipe.predict_proba(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    qwk = cohen_kappa_score(y_test, y_pred, weights='quadratic')
    auc_macro_ovr = roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro')
    
    benchmark_results.append({
        'Model': name,
        'Acc.': f"{acc * 100:.2f}%",
        'Prec. (Macro)': f"{prec * 100:.2f}%",
        'Rec. (Macro)': f"{rec * 100:.2f}%",
        'F1 (Macro)': f"{f1 * 100:.2f}%",
        'QWK': f"{qwk:.4f}",
        'AUC (Macro OVR)': f"{auc_macro_ovr:.4f}"
    })
    
    results_for_csv.append({
        'Model': name,
        'Test_Accuracy': acc,
        'Test_Precision_macro': prec,
        'Test_Recall_macro': rec,
        'Test_F1_macro': f1,
        'Test_QWK': qwk,
        'Test_ROC_AUC_macro_OVR': auc_macro_ovr
    })

proof_table = pd.DataFrame(benchmark_results)
print("=== LEAKAGE-CONTROLLED BENCHMARK PERFORMANCE TABLE (TABLE 4.1) ===")
print(proof_table.to_string(index=False))

# Save numeric csv for chart generator script
pd.DataFrame(results_for_csv).to_csv('model_results.csv', index=False)
print("\nExported clean results to 'model_results.csv'.")

# ---------------------------------------------------------
# 7. GENERATE DIAGNOSTIC FIGURES
# ---------------------------------------------------------
# Confusion Matrices
fig, axes = plt.subplots(1, len(models), figsize=(16, 4.5), dpi=300)
labels = [f"Class {c}" for c in classes]

for idx, (name, pipe) in enumerate(trained_pipelines.items()):
    y_pred = pipe.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=axes[idx], cmap='Blues', colorbar=False)
    axes[idx].set_title(f"{name}\nTest Split Predictions", fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('confusion_matrices_proof.png', bbox_inches='tight')
print("Saved leakage-controlled confusion matrices: 'confusion_matrices_proof.png'")

# ROC Curves (Macro OVR Distinction)
fig_roc, ax_roc = plt.subplots(figsize=(8, 6), dpi=300)
y_test_bin = label_binarize(y_test, classes=classes)
colors_dict = {'Logistic Reg. (L2)': 'blue', 'Decision Tree': 'orange', 'Random Forest': 'green'}

for name, pipe in trained_pipelines.items():
    y_proba = pipe.predict_proba(X_test)
    fpr, tpr, _ = roc_curve(y_test_bin.ravel(), y_proba.ravel())
    roc_micro_auc = auc(fpr, tpr)
    ax_roc.plot(fpr, tpr, color=colors_dict[name], lw=2, 
                label=f'{name} (Micro-AUC = {roc_micro_auc:.4f})')

ax_roc.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random Chance')
ax_roc.set_xlim([0.0, 1.0])
ax_roc.set_ylim([0.0, 1.05])
ax_roc.set_xlabel('False Positive Rate', fontsize=11)
ax_roc.set_ylabel('True Positive Rate', fontsize=11)
ax_roc.set_title('Multi-Class ROC Curves (20% Holdout Test Split)', fontsize=12, fontweight='bold')
ax_roc.legend(loc="lower right")
ax_roc.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('roc_curves_proof.png', bbox_inches='tight')
print("Saved leakage-controlled ROC curves: 'roc_curves_proof.png'")