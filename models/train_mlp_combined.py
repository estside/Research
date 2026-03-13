import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr

from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor

# ==========================================================
# 1. LOAD DATA
# ==========================================================

print("Loading 150M ESM-2 and Windowed datasets...")
BASE_DIR = Path(__file__).resolve().parents[1]
DATASETS_DIR = BASE_DIR / "datasets"
PLOTS_DIR = BASE_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

esm_df = pd.read_csv(DATASETS_DIR / "skempi_esm2_features.csv")
window_df = pd.read_csv(DATASETS_DIR / "skempi_window_features.csv")


merged_df = pd.merge(esm_df, window_df, on=['#Pdb', 'ddG'], how='inner')

# Add this line to reveal your final dataset size!
print(f"FINAL MERGED DATASET SIZE: {len(merged_df)} mutations")

# ==========================================================
# 2. FEATURE BREAKDOWN
# ==========================================================

esm_features = [c for c in merged_df.columns if c.startswith('esm_dim_')]
window_features = [c for c in merged_df.columns if c.startswith('L') or c.startswith('R')]
base_features = ['delta_vol', 'delta_hydro', 'loc']

print("\nFeature Breakdown:")
print(f"  - ESM Features: {len(esm_features)}")
print(f"  - Window Features: {len(window_features)}")
print(f"  - Base Features: {len(base_features)}")

# Feature sets
X_base = merged_df[base_features].values
X_window = merged_df[window_features].values
X_esm = merged_df[esm_features].values
X_combined = merged_df[esm_features + window_features + base_features].values

y = merged_df['ddG'].values
groups = merged_df['#Pdb'].values

# ==========================================================
# 3. MODEL DEFINITIONS
# ==========================================================

def build_mlp(input_type):
    if input_type == "base":
        return MLPRegressor(hidden_layer_sizes=(64,),
                            alpha=0.1,
                            max_iter=1000,
                            early_stopping=True,
                            random_state=42)
    else:
        return MLPRegressor(hidden_layer_sizes=(512,256,64),
                            alpha=0.5,
                            max_iter=1000,
                            early_stopping=True,
                            random_state=42)

models = {
    "Mutation_Only": lambda: build_mlp("base"),
    "Window_Only": lambda: build_mlp("window"),
    "ESM_Only": lambda: build_mlp("esm"),
    "Combined_MLP": lambda: build_mlp("combined"),
    "Ridge": lambda: Ridge(alpha=1.0),
    "RandomForest": lambda: RandomForestRegressor(
        n_estimators=200,
        max_depth=20,
        random_state=42,
        n_jobs=-1
    )
}

feature_map = {
    "Mutation_Only": X_base,
    "Window_Only": X_window,
    "ESM_Only": X_esm,
    "Combined_MLP": X_combined,
    "Ridge": X_combined,
    "RandomForest": X_combined
}

# ==========================================================
# 4. GROUPED 5-FOLD CROSS VALIDATION
# ==========================================================

gkf = GroupKFold(n_splits=5)

results = {}
all_true_combined = []
all_pred_combined = []

print("\nRunning Grouped 5-Fold Cross-Validation...")

for name in models:

    print(f"\nModel: {name}")

    mae_list = []
    mse_list = []
    pearson_list = []

    X_current = feature_map[name]

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X_current, y, groups)):

        X_train, X_test = X_current[train_idx], X_current[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Scale ONLY for linear/MLP models
        if name != "RandomForest":
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        model = models[name]()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        mse = mean_squared_error(y_test, preds)
        pearson = pearsonr(y_test, preds)[0]

        mae_list.append(mae)
        mse_list.append(mse)
        pearson_list.append(pearson)

        print(f"  Fold {fold+1}: MAE={mae:.3f}, Pearson={pearson:.3f}")

        if name == "Combined_MLP":
            all_true_combined.extend(y_test)
            all_pred_combined.extend(preds)

    results[name] = {
        "MAE_mean": np.mean(mae_list),
        "MAE_std": np.std(mae_list),
        "MSE_mean": np.mean(mse_list),
        "Pearson_mean": np.mean(pearson_list),
        "Pearson_std": np.std(pearson_list)
    }

# ==========================================================
# 5. FINAL RESULTS
# ==========================================================

print("\n" + "="*40)
print("FINAL GROUPED 5-FOLD RESULTS")
print("="*40)

for name, metrics in results.items():
    print(f"\n{name}")
    print(f"  MAE: {metrics['MAE_mean']:.3f} ± {metrics['MAE_std']:.3f}")
    print(f"  MSE: {metrics['MSE_mean']:.3f}")
    print(f"  Pearson r: {metrics['Pearson_mean']:.3f} ± {metrics['Pearson_std']:.3f}")

# ==========================================================
# 6. FINAL SCATTER PLOT (Combined Model)
# ==========================================================

# ==========================================================
# 6. FINAL SCATTER PLOT (Combined Model)
# ==========================================================
import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(8, 6), dpi=300)

# Plot the actual vs predicted points
plt.scatter(all_true_combined, all_pred_combined, alpha=0.6, color='#2ca02c', edgecolors='w', s=60, label='Predicted Mutations')

# Create the ideal prediction dashed line
min_val = min(min(all_true_combined), min(all_pred_combined)) - 0.5
max_val = max(max(all_true_combined), max(all_pred_combined)) + 0.5
plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label="Ideal Prediction (y=x)")

plt.xlabel("Experimental $\Delta\Delta$G (kcal/mol)", fontsize=12, fontweight='bold')
plt.ylabel("Predicted $\Delta\Delta$G (kcal/mol)", fontsize=12, fontweight='bold')
plt.title("Grouped 5-Fold CV: Combined MLP Model", fontsize=14, pad=15)

plt.legend(loc='upper left')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()

output_path = PLOTS_DIR / "scatter_grouped_combined.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\nSaved high-resolution scatter plot: {output_path}")
