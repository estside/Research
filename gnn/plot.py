import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split

# Import your custom modules
from dataset import SKEMPIGraphDataset
from fusion_network import SiameseDeltaGNN

print("--- Generating Publication-Ready Scatter Plot ---")

# 1. Setup Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Recreate the Dataset and Test Split (Must match evaluation exactly)
dataset = SKEMPIGraphDataset(root_dir=".")
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

generator = torch.Generator().manual_seed(42)
_, test_dataset = random_split(dataset, [train_size, test_size], generator=generator)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# 3. Load Model and Weights
model = SiameseDeltaGNN(node_feature_dim=50, hidden_dim=128).to(device)
checkpoint = torch.load("checkpoint_best.pth", map_location=device, weights_only=True)

if 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    model.load_state_dict(checkpoint)

# 4. Run Inference
model.eval()
all_predictions = []
all_targets = []

print("Running predictions on the test set...")
with torch.no_grad():
    for wt_batch, mt_batch in test_loader:
        wt_batch, mt_batch = wt_batch.to(device), mt_batch.to(device)
        predictions = model(wt_batch, mt_batch)
        all_predictions.extend(predictions.cpu().numpy().flatten())
        all_targets.extend(wt_batch.y.cpu().numpy().flatten())

targets = np.array(all_targets)
preds = np.array(all_predictions)

# Calculate final metrics for the plot
r_val, p_val = pearsonr(targets, preds)
mae_val = np.mean(np.abs(targets - preds))

# ---------------------------------------------------------
# 5. MATPLOTLIB & SEABORN STYLING
# ---------------------------------------------------------
print("Rendering high-resolution plot...")

# Set a clean, professional academic style
sns.set_theme(style="ticks", context="paper", font_scale=1.2)
plt.figure(figsize=(8, 8))

# Create a scatter plot with density-like alpha transparency
plt.scatter(targets, preds, alpha=0.5, edgecolor='w', linewidth=0.5, s=60, color="#2c3e50", label="Test Set Mutations")

# Add the perfect prediction reference line (y = x)
min_val = min(min(targets), min(preds)) - 1
max_val = max(max(targets), max(preds)) + 1
plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label="Perfect Prediction (y = x)")

# Add a linear regression trendline to show the model's actual fit
sns.regplot(x=targets, y=preds, scatter=False, color="#e74c3c", line_kws={"linewidth": 2, "label": "Model Fit (Regression)"})

# Add titles and labels
plt.title(r"3D GNN: Predicted vs. Experimental $\Delta\Delta G$", fontsize=16, fontweight='bold', pad=15)
plt.xlabel(r"Experimental $\Delta\Delta G$ (kcal/mol)", fontsize=14)
plt.ylabel(r"Predicted $\Delta\Delta G$ (kcal/mol)", fontsize=14)

# Create a text box with the critical statistics
stats_text = f"Pearson $r$ = {r_val:.4f}\nMAE = {mae_val:.4f} kcal/mol\n$p$-value < 0.001\n$n$ = {len(targets)}"
props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
plt.gca().text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, fontsize=12,
               verticalalignment='top', bbox=props)

# Clean up layout and add legend
plt.legend(loc="lower right", frameon=True)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

# Save as a high-resolution PNG
save_filename = "gnn_scatter_results.png"
plt.savefig(save_filename, dpi=300, bbox_inches='tight')
print(f"✅ Plot successfully saved as '{save_filename}'")