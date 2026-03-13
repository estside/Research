import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
from pathlib import Path

# 1. Load both datasets
print("Loading 150M ESM-2 and Windowed datasets...")
BASE_DIR = Path(__file__).resolve().parents[1]
DATASETS_DIR = BASE_DIR / "datasets"
esm_df = pd.read_csv(DATASETS_DIR / "skempi_esm2_features.csv")
window_df = pd.read_csv(DATASETS_DIR / "skempi_window_features.csv")

# 2. Merge them safely
merged_df = pd.merge(esm_df, window_df, on=['#Pdb', 'ddG'], how='inner')

# 3. Feature breakdown
esm_features = [col for col in merged_df.columns if col.startswith('esm_dim_')]
window_features = [col for col in merged_df.columns if col.startswith('L') or col.startswith('R')]
base_features = ['delta_vol', 'delta_hydro', 'loc']

all_features = esm_features + window_features + base_features

print(f"\nFeature Breakdown:")
print(f"  - Global Context (ESM-2 150M): {len(esm_features)} features")
print(f"  - Local Context (Window): {len(window_features)} features")
print(f"  - Core Biophysical (Base): {len(base_features)} features")
print(f"  - TOTAL Input Size: {len(all_features)} features")

X = merged_df[all_features]
y = merged_df['ddG']
groups = merged_df['#Pdb']

# 4. Rigorous Split
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups=groups))

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

# 5. Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. Train the Heavily Regularized MLP
print("\nTraining Combined MLP with High Regularization...")

mlp = MLPRegressor(
    # A wider first layer to absorb the 649 inputs, funneling down gently
    hidden_layer_sizes=(512, 256, 64), 
    activation='relu',
    solver='adam',
    # CRITICAL: Alpha increased from 0.01 to 0.5. 
    # This heavily penalizes the model for relying too much on any single ESM dimension, forcing generalization.
    alpha=0.5, 
    learning_rate_init=0.001,
    max_iter=1000,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.1
)

mlp.fit(X_train_scaled, y_train)

# 7. Evaluation
preds = mlp.predict(X_test_scaled)

mse = mean_squared_error(y_test, preds)
mae = mean_absolute_error(y_test, preds)
r2 = r2_score(y_test, preds)
p_corr, _ = pearsonr(y_test, preds)

print("\n" + "="*30)
print("150M COMBINED MODEL RESULTS")
print("="*30)
print(f"MSE: {mse:.3f}")
print(f"MAE: {mae:.3f} kcal/mol")
print(f"R^2: {r2:.3f}")
print(f"Pearson r: {p_corr:.3f}")
print("="*30)

print(f"\nTraining iterations to converge: {mlp.n_iter_}")
