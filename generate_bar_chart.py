import matplotlib.pyplot as plt
import numpy as np

# Data from Table II
models = ['Logistic Reg. (L2)', 'Decision Tree', 'Random Forest']
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']

# Scores in percentages for direct comparison
scores = {
    'Logistic Reg. (L2)': [85.42, 84.80, 85.10, 84.95, 93.20],
    'Decision Tree': [88.75, 88.20, 88.50, 88.35, 94.10],
    'Random Forest': [93.33, 93.10, 93.25, 93.17, 97.85]
}

x = np.arange(len(metrics))  # metric locations
width = 0.25  # bar width

plt.figure(figsize=(10, 5.5), dpi=300)
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Professional IEEE colors

for idx, (model_name, model_scores) in enumerate(scores.items()):
    offset = (idx - 1) * width
    rects = plt.bar(x + offset, model_scores, width, label=model_name, color=colors[idx], edgecolor='black', linewidth=0.8)
    
    # Add text labels on top of bars
    for rect in rects:
        height = rect.get_height()
        plt.annotate(f'{height:.1f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8, fontweight='bold')

plt.ylabel('Performance Metric Score (%)', fontsize=11, fontweight='bold')
plt.title('Benchmark Performance Comparison Across Supervised ML Classifiers', fontsize=12, fontweight='bold', pad=15)
plt.xticks(x, metrics, fontsize=10, fontweight='bold')
plt.ylim(75, 103)
plt.legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.9, fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('model_comparison_bar_chart.png', dpi=300)
print("Bar chart figure saved to 'model_comparison_bar_chart.png'")

