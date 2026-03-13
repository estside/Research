import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
PLOTS_DIR = BASE_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Data from Grouped 5-Fold CV Output
labels = ['Mutation\nOnly', 'Window\nOnly', 'ESM-2\nOnly', 'Combined\nMLP', 'Ridge\n(Combined)', 'Random\nForest']
mae = [0.912, 0.935, 0.935, 0.920, 1.216, 0.900]
pearson = [0.145, 0.000, 0.294, 0.359, 0.271, 0.284]

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 6), dpi=300)

rects1 = ax.bar(x - width/2, mae, width, label='MAE (kcal/mol)', color='#66b3ff', edgecolor='black', linewidth=0.5)
rects2 = ax.bar(x + width/2, pearson, width, label='Pearson r', color='#ffcc99', edgecolor='black', linewidth=0.5)

ax.set_ylabel('Score', fontsize=12, fontweight='bold')
ax.set_title('Grouped 5-Fold CV: Model Performance Comparison', fontsize=15, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=11)
ax.legend(loc='upper right')

# Add subtle grid for readability
ax.yaxis.grid(True, linestyle='--', alpha=0.7)
ax.set_axisbelow(True)

fig.tight_layout()
output_path = PLOTS_DIR / "5fold_graph.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Successfully generated '{output_path}'")
