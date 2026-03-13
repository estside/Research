import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
PLOTS_DIR = BASE_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Data gathered from your previous runs
labels = [
    'Baseline\n(Mutation Only)', 
    'Local Window\n(±3 Residues)', 
    'Global ESM-2\n(8M Params)', 
    'Combined\n(8M + Window)', 
    'Combined 150M\n(1-Fold Peak)',
    'Combined 150M\n(5-Fold CV Avg)'
]

# Metrics
mse = [2.302, 1.018, 2.199, 0.732, 0.734, 1.425]
mae = [1.016, 0.717, 1.136, 0.674, 0.651, 0.905]
r2 = [0.098, -0.150, 0.334, 0.360, 0.358, 0.120] # Approximated R2 for CV average
pearson = [0.365, 0.285, 0.584, 0.630, 0.664, 0.538]

x = np.arange(len(labels))
width = 0.2

fig, ax = plt.subplots(figsize=(13, 7), dpi=300)

rects1 = ax.bar(x - 1.5*width, mse, width, label='MSE', color='#ff9999')
rects2 = ax.bar(x - 0.5*width, mae, width, label='MAE (kcal/mol)', color='#66b3ff')
rects3 = ax.bar(x + 0.5*width, r2, width, label='R²', color='#99ff99')
rects4 = ax.bar(x + 1.5*width, pearson, width, label='Pearson r', color='#ffcc99')

ax.set_ylabel('Metric Score', fontsize=12)
ax.set_title('Performance Comparison of $\Delta\Delta$G Prediction Architectures', fontsize=15, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=10)
ax.legend(loc='upper right')
ax.axhline(0, color='black', linewidth=0.8, linestyle='--') 

fig.tight_layout()
output_path = PLOTS_DIR / "graph.png"
plt.savefig(output_path)
print(f"Successfully generated updated '{output_path}'")
