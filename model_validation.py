import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, cohen_kappa_score, confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc
)

# Set seed for reproducibility
np.random.seed(42)

DATASET_PATH = Path('train.csv')

# ---------------------------------------------------------
# 1. LOAD AND CLEAN DATASET
# ---------------------------------------------------------
print(f"Loading dataset from '{DATASET_PATH}'...")
df = pd.read_csv(DATASET_PATH)

target_col = 'sii'
df = df.dropna(subset=[target_col]).copy()
df[target_col] = df[target_col].astype(int)

# ---------------------------------------------------------
# 2. PREVENT LEAKAGE (STRICT FEATURE SEPARATION)
# ---------------------------------------------------------
# Exclude target, identifiers, and ALL PCIAT items/aggregates
pciat_cols = [c for c in df.columns if 'PCIAT' in c or 'pciat' in c.lower()]
non_predictor_cols = ['id', 'id_seq', 'PCIAT-Season', target_col] + pciat_cols

feature_cols = [c for c in df.columns if c not in non_predictor_cols]
X = df[feature_cols].copy()
y = df[target_col].copy()

num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

print(f"Features: {len(feature_cols)} ({len(num_cols)} numeric, {len(cat_cols)} categorical).")

# ---------------------------------------------------------
# 3. STRATIFIED 80/20 HOLDOUT TEST SPLIT
# ---------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

# ---------------------------------------------------------
# 4. ENCAPSULATED PREPROCESSING PIPELINES
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

# Explicit 5-Fold Stratified CV Strategy
cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ---------------------------------------------------------
# 5. MODEL ESTIMATORS & HYPERPARAMETER GRIDS
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
# 6. EVALUATION PIPELINE
# ---------------------------------------------------------
classes = sorted(y.unique())
results_summary = []
per_class_results = []
trained_pipelines = {}

for name, (pipe, param_grid) in models.items():
    grid = GridSearchCV(pipe, param_grid, cv=cv_strategy, scoring='f1_macro', n_jobs=-1)
    grid.fit(X_train, y_train)
    
    best_pipe = grid.best_estimator_
    trained_pipelines[name] = best_pipe
    
    y_pred = best_pipe.predict(X_test)
    y_proba = best_pipe.predict_proba(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    prec_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
    rec_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
    qwk = cohen_kappa_score(y_test, y_pred, weights='quadratic')
    
    # Explicit Distinctions for ROC AUC
    auc_macro_ovr = roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro')
    auc_micro_ovr = roc_auc_score(y_test, y_proba, multi_class='ovr', average='micro')
    
    results_summary.append({
        'Model': name,
        'Test_Accuracy': acc,
        'Test_Balanced_Accuracy': bal_acc,
        'Test_Precision_macro': prec_macro,
        'Test_Recall_macro': rec_macro,
        'Test_F1_macro': f1_macro,
        'Test_QWK': qwk,
        'Test_ROC_AUC_macro_OVR': auc_macro_ovr,
        'Test_ROC_AUC_micro_OVR': auc_micro_ovr
    })
    
    # Per-Class Breakdown
    prec_pc = precision_score(y_test, y_pred, average=None, zero_division=0)
    rec_pc = recall_score(y_test, y_pred, average=None, zero_division=0)
    f1_pc = f1_score(y_test, y_pred, average=None, zero_division=0)
    
    for idx_c, c in enumerate(classes):
        per_class_results.append({
            'Model': name,
            'Class': f"Class {c}",
            'Precision': prec_pc[idx_c],
            'Recall': rec_pc[idx_c],
            'F1': f1_pc[idx_c]
        })

df_results_summary = pd.DataFrame(results_summary)
df_per_class = pd.DataFrame(per_class_results)

# Export clean CSV files
df_results_summary.to_csv('model_results.csv', index=False)
df_per_class.to_csv('per_class_results.csv', index=False)

print("\n=== SUMMARY METRICS ===")
print(df_results_summary.to_string(index=False))

# ---------------------------------------------------------
# 7. IMPROVED VISUALIZATIONS
# ---------------------------------------------------------
# Figure 1: Enhanced Confusion Matrices (Counts + Row Recall Percentages)
fig, axes = plt.subplots(1, len(models), figsize=(16, 4.8), dpi=300)
labels = [f"Class {c}" for c in classes]

for idx, (name, pipe) in enumerate(trained_pipelines.items()):
    y_pred = pipe.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    # Calculate row-normalized recall percentages
    cm_perc = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    annot = np.empty_like(cm, dtype=object)
    for r in range(cm.shape[0]):
        for c in range(cm.shape[1]):
            annot[r, c] = f"{cm[r, c]}\n({cm_perc[r, c]:.1f}%)"
            
    sns.heatmap(cm, annot=annot, fmt="", cmap='Blues', cbar=False, ax=axes[idx],
                xticklabels=labels, yticklabels=labels, annot_kws={"size": 9, "weight": "bold"})
    axes[idx].set_title(f"{name}\nTest Split Predictions", fontsize=11, fontweight='bold', pad=10)
    axes[idx].set_xlabel('Predicted Label', fontsize=10, fontweight='bold')
    if idx == 0:
        axes[idx].set_ylabel('True Label', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('confusion_matrices_proof.png', bbox_inches='tight')
plt.close()

# Figure 2: Enhanced ROC Curves (Macro OVR vs. Micro OVR)
fig_roc, ax_roc = plt.subplots(figsize=(8.5, 6.5), dpi=300)
y_test_bin = label_binarize(y_test, classes=classes)
colors_dict = {'Logistic Reg. (L2)': '#1f77b4', 'Decision Tree': '#ff7f0e', 'Random Forest': '#2ca02c'}

for name, pipe in trained_pipelines.items():
    y_proba = pipe.predict_proba(X_test)
    
    # Micro-average ROC
    fpr_micro, tpr_micro, _ = roc_curve(y_test_bin.ravel(), y_proba.ravel())
    roc_micro_auc = auc(fpr_micro, tpr_micro)
    
    # Macro-average ROC
    fpr_grid = np.unique(np.concatenate([roc_curve(y_test_bin[:, i], y_proba[:, i])[0] for i in range(len(classes))]))
    mean_tpr = np.zeros_like(fpr_grid)
    for i in range(len(classes)):
        fpr_i, tpr_i, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
        mean_tpr += np.interp(fpr_grid, fpr_i, tpr_i)
    mean_tpr /= len(classes)
    roc_macro_auc = auc(fpr_grid, mean_tpr)
    
    ax_roc.plot(fpr_grid, mean_tpr, color=colors_dict[name], lw=2.2,
                label=f'{name} (Macro-AUC = {roc_macro_auc:.4f})')
    ax_roc.plot(fpr_micro, tpr_micro, color=colors_dict[name], lw=1.5, linestyle=':',
                label=f'  └─ {name} (Micro-AUC = {roc_micro_auc:.4f})')

ax_roc.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random Chance')
ax_roc.set_xlim([-0.01, 1.0])
ax_roc.set_ylim([0.0, 1.02])
ax_roc.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=11, fontweight='bold')
ax_roc.set_ylabel('True Positive Rate (Sensitivity)', fontsize=11, fontweight='bold')
ax_roc.set_title('Multi-Class One-vs-Rest ROC Curves (20% Holdout Test Split)', fontsize=12, fontweight='bold', pad=12)
ax_roc.legend(loc="lower right", fontsize=9, frameon=True, facecolor='white', framealpha=0.95)
ax_roc.grid(True, linestyle='--', alpha=0.4)

plt.tight_layout()
plt.savefig('roc_curves_proof.png', bbox_inches='tight')
plt.close()

print("Execution complete. Exported 'model_results.csv' and updated figures.")