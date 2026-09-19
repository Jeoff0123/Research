"""Generate a comparison chart from measured model_results.csv.

Run model_validation.py first. This script intentionally does not contain
hardcoded performance values.
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

RESULTS_PATH = Path("model_results.csv")
OUTPUT_PATH = "model_comparison_hbn.png"

if not RESULTS_PATH.exists():
    raise FileNotFoundError(
        "model_results.csv was not found. Run model_validation.py first."
    )

results = pd.read_csv(RESULTS_PATH)

required = {
    "Model",
    "Test_Accuracy",
    "Test_Precision_macro",
    "Test_Recall_macro",
    "Test_F1_macro",
    "Test_ROC_AUC_macro_OVR",
}
missing = required - set(results.columns)
if missing:
    raise ValueError(f"model_results.csv is missing columns: {sorted(missing)}")

metrics = {
    "Accuracy": "Test_Accuracy",
    "Precision": "Test_Precision_macro",
    "Recall": "Test_Recall_macro",
    "F1": "Test_F1_macro",
    "ROC-AUC": "Test_ROC_AUC_macro_OVR",
}

x = np.arange(len(metrics))
width = 0.8 / len(results)

fig, ax = plt.subplots(figsize=(11, 6), dpi=300)

for i, row in results.iterrows():
    values = [row[column] * 100 for column in metrics.values()]
    offset = (i - (len(results) - 1) / 2) * width
    bars = ax.bar(x + offset, values, width, label=row["Model"])

    for bar in bars:
        ax.annotate(
            f"{bar.get_height():.1f}%",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )

ax.set_ylabel("Score (%)")
ax.set_title("Held-Out Test Performance Comparison")
ax.set_xticks(x)
ax.set_xticklabels(metrics.keys())
ax.set_ylim(0, 105)
ax.legend()
ax.grid(axis="y", linestyle="--", alpha=0.4)

plt.tight_layout()
plt.savefig(OUTPUT_PATH, bbox_inches="tight")
plt.close(fig)

print(f"Chart saved to {OUTPUT_PATH}")
