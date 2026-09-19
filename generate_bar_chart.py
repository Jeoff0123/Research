import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
)

# Set seed for statistical reproducibility
np.random.seed(42)

DATASET_PATH = 'train.csv'

# ---------------------------------------------------------
# 1. LOAD AND PREPROCESS REAL TRAIN.CSV DATASET
# ---------------------------------------------------------
print(f"Loading real dataset from '{DATASET_PATH}'...")
df = pd.read_csv(DATASET_PATH)

# Identify target variable ('sii' or 'riskLevel')
target_col = 'sii' if 'sii' in df.columns else 'riskLevel'

# Drop rows missing the primary target label
df = df.dropna(subset=[target_col]).copy()
df[target_col] = df[target_col].astype(int)

# Identify PCIAT domain columns and aggregate total score
pciat_cols = [c for c in df.columns if 'PCIAT' in c and c != 'PCIAT-Season' and c != 'PCIAT-PCIAT_Total']

if pciat_cols:
    for col in pciat_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['PCIAT_Total'] = df[pciat_cols].sum(axis=1, skipna=True)
    
    # Emotional Volatility sub-scale (if PCIAT item questions are available)
    emo_cols = [c for c in pciat_cols if any(k in c for k in ['13', '16', '18', '20'])]
    if emo_cols:
        df['PCIAT_Emotional_Volatility'] = df[emo_cols].mean(axis=1, skipna=True)

# Safe conversion for Sleep & Screen time interaction feature
screen_col = [c for c in df.columns if 'hoursday' in c.lower() or 'screen' in c.lower()]
sleep_col = [c for c in df.columns if 'sds' in c.lower() or 'sleep' in c.lower()]

if screen_col and sleep_col:
    screen_vals = pd.to_numeric(df[screen_col[0]], errors='coerce').fillna(0)
    sleep_vals = pd.to_numeric(df[sleep_col[0]], errors='coerce').fillna(0)
    df['Interaction_Sleep_ScreenTime'] = screen_vals * sleep_vals

# Safe conversion for Physical Sedentary Ratio
fat_col = [c for c in df.columns if 'fat' in c.lower()]
paq_col = [c for c in df.columns if 'paq' in c.lower()]

if fat_col and paq_col:
    fat_vals = pd.to_numeric(df[fat_col[0]], errors='coerce').fillna(0)
    paq_vals = pd.to_numeric(df[paq_col[0]], errors='coerce').fillna(0)
    df['Physical_Sedentary_Ratio'] = fat_vals / (paq_vals + 0.1)

# Separate ID, non-predictive columns, and target
drop_cols = ['id', 'id_seq', 'PCIAT-Season', target_col]
feature_cols = [c for c in df.columns if c not in drop_cols]

X = df[feature_cols].copy()
y = df[target_col].copy()

# Ensure all feature columns are numeric where possible
for col in X.columns:
    if X[col].dtype == 'object':
        try:
            X[col] = pd.to_numeric(X[col])
        except (ValueError, TypeError):
            pass

num_cols = X.select_dtypes(include=[np.number]).columns
cat_cols = X.select_dtypes(exclude=[np.number]).columns

num_imputer = SimpleImputer(strategy='median')
X[num_cols] = num_imputer.fit_transform(X[num_cols])

if len(cat_cols) > 0:
    X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

print(f"Dataset Loaded & Preprocessed: {X.shape[0]} valid participant records with {X.shape[1]} predictor features.")

# ---------------------------------------------------------
# 2. STRATIFIED 80/20 TRAIN-TEST SPLIT
# ---------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

# Standardize features for linear model convergence
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------
# 3. HYPERPARAMETER TUNING & EVALUATION PIPELINE
# ---------------------------------------------------------
models = {
    'Logistic Reg. (L2)': (
        LogisticRegression(max_iter=1000, random_state=42),
        {'C': [0.01, 0.1, 1.0, 10.0]},
        X_train_scaled, X_test_scaled
    ),
    'Decision Tree': (
        DecisionTreeClassifier(random_state=42),
        {'max_depth': [3, 5, 10, None], 'criterion': ['gini', 'entropy']},
        X_train, X_test
    ),
    'Random Forest': (
        RandomForestClassifier(random_state=42),
        {'n_estimators': [50, 100, 200], 'max_depth': [5, 10, None]},
        X_train, X_test
    )
}

benchmark_results = []
trained_estimators = {}
model_scores_dict = {}

for name, (model, param_grid, tr_x, te_x) in models.items():
    grid = GridSearchCV(model, param_grid, cv=5, scoring='f1_macro', n_jobs=-1)
    grid.fit(tr_x, y_train)
    
    best_estimator = grid.best_estimator_
    trained_estimators[name] = (best_estimator, te_x)
    
    y_pred = best_estimator.predict(te_x)
    y_proba = best_estimator.predict_proba(te_x)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    auc_val = roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro')
    
    model_scores_dict[name] = [acc * 100, prec * 100, rec * 100, f1 * 100, auc_val * 100]
    
    benchmark_results.append({
        'Model': name,
        'Acc.': f"{acc * 100:.2f}%",
        'Prec.': f"{prec * 100:.2f}%",
        'Rec.': f"{rec * 100:.2f}%",
        'F1': f"{f1 * 100:.2f}%",
        'AUC': f"{auc_val:.4f}"
    })

proof_table = pd.DataFrame(benchmark_results)
print("\n=== VERIFIED MULTI-DOMAIN BENCHMARK PERFORMANCE TABLE (TABLE 4.1) ===")
print(proof_table.to_string(index=False))

# ---------------------------------------------------------
# 4. GENERATE MODEL COMPARISON BAR CHART
# ---------------------------------------------------------
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
x = np.arange(len(metrics))
width = 0.25

plt.figure(figsize=(10, 5.5), dpi=300)
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

for idx, (model_name, scores_list) in enumerate(model_scores_dict.items()):
    offset = (idx - 1) * width
    rects = plt.bar(x + offset, scores_list, width, label=model_name, color=colors[idx], edgecolor='black', linewidth=0.8)
    for rect in rects:
        height = rect.get_height()
        plt.annotate(f'{height:.1f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8, fontweight='bold')

plt.ylabel('Performance Metric Score (%)', fontsize=11, fontweight='bold')
plt.title('Benchmark Performance Comparison Across Supervised ML Classifiers', fontsize=12, fontweight='bold', pad=15)
plt.xticks(x, metrics, fontsize=10, fontweight='bold')
plt.ylim(30, 105)
plt.legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.9, fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('model_comparison_bar_chart.png', dpi=300)
print("\nSaved model comparison bar chart: 'model_comparison_bar_chart.png'")

# ---------------------------------------------------------
# 5. GENERATE CONFUSION MATRICES
# ---------------------------------------------------------
classes = sorted(y.unique())
labels = [f"Class {c}" for c in classes]

fig, axes = plt.subplots(1, len(models), figsize=(16, 4.5))

for idx, (name, (model, te_x)) in enumerate(trained_estimators.items()):
    y_pred = model.predict(te_x)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=axes[idx], cmap='Blues', colorbar=False)
    axes[idx].set_title(f"{name}\nTest Split Predictions", fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('confusion_matrices_proof.png', dpi=300)
print("Saved updated confusion matrices figure: 'confusion_matrices_proof.png'")

# ---------------------------------------------------------
# 6. GENERATE ROC CURVES
# ---------------------------------------------------------
fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
y_test_bin = label_binarize(y_test, classes=classes)
colors_dict = {'Logistic Reg. (L2)': 'blue', 'Decision Tree': 'orange', 'Random Forest': 'green'}

for name, (model, te_x) in trained_estimators.items():
    y_proba = model.predict_proba(te_x)
    fpr, tpr, _ = roc_curve(y_test_bin.ravel(), y_proba.ravel())
    roc_auc = auc(fpr, tpr)
    ax_roc.plot(fpr, tpr, color=colors_dict[name], lw=2, label=f'{name} (micro-AUC = {roc_auc:.4f})')

ax_roc.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random Chance')
ax_roc.set_xlim([0.0, 1.0])
ax_roc.set_ylim([0.0, 1.05])
ax_roc.set_xlabel('False Positive Rate', fontsize=11)
ax_roc.set_ylabel('True Positive Rate', fontsize=11)
ax_roc.set_title('Multi-Class Multi-Domain ROC Curves (20% Test Split)', fontsize=12, fontweight='bold')
ax_roc.legend(loc="lower right")
ax_roc.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('roc_curves_proof.png', dpi=300)
print("Saved updated ROC curves figure: 'roc_curves_proof.png'")

