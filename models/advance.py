import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr
from pathlib import Path

# 1. Load Data
print("Loading datasets for Advanced Analysis...")
BASE_DIR = Path(__file__).resolve().parents[1]
DATASETS_DIR = BASE_DIR / "datasets"
esm_df = pd.read_csv(DATASETS_DIR / "skempi_esm2_features.csv")
window_df = pd.read_csv(DATASETS_DIR / "skempi_window_features.csv")
merged_df = pd.merge(esm_df, window_df, on=['#Pdb', 'ddG'], how='inner')

# 2. Setup Features
esm_features = [c for c in merged_df.columns if c.startswith('esm_dim_')]
window_features = [c for c in merged_df.columns if c.startswith('L') or c.startswith('R')]
base_features = ['delta_vol', 'delta_hydro', 'loc']

X_base = merged_df[base_features].values
X_window = merged_df[window_features].values
X_esm = merged_df[esm_features].values
y = merged_df['ddG'].values
groups = merged_df['#Pdb'].values

# Splitting (Using 1 Fold 80/20 grouped split for consistent, fast comparison)
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(merged_df, y, groups=groups))

# ==========================================================
# OPTION B: FULL FEATURE IMPORTANCE (Random Forest)
# ==========================================================
print("\n" + "="*50)
print("OPTION B: FULL FEATURE IMPORTANCE (All 649 Features)")
print("="*50)
X_all = merged_df[base_features + window_features + esm_features]
X_train_rf, X_test_rf = X_all.iloc[train_idx], X_all.iloc[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

rf = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
rf.fit(X_train_rf, y_train)

importances = pd.DataFrame({
    'feature': X_all.columns,
    'importance': rf.feature_importances_
}).sort_values(by='importance', ascending=False)

print("Top 15 Most Important Features:")
print(importances.head(15).to_string(index=False))


# ==========================================================
# OPTION A: PCA EMBEDDING ANALYSIS (MLP)
# ==========================================================
print("\n" + "="*50)
print("OPTION A: PCA DIMENSIONALITY REDUCTION ON ESM-2")
print("="*50)

# Standard Scale ESM features before PCA
scaler_esm = StandardScaler()
X_esm_train = scaler_esm.fit_transform(X_esm[train_idx])
X_esm_test = scaler_esm.transform(X_esm[test_idx])

# Standard Scale Base + Window features
X_rest_train = np.hstack((X_base[train_idx], X_window[train_idx]))
X_rest_test = np.hstack((X_base[test_idx], X_window[test_idx]))
scaler_rest = StandardScaler()
X_rest_train = scaler_rest.fit_transform(X_rest_train)
X_rest_test = scaler_rest.transform(X_rest_test)

dims_to_test = [640, 100, 50, 20]

for dim in dims_to_test:
    if dim == 640:
        X_train_pca = X_esm_train
        X_test_pca = X_esm_test
    else:
        pca = PCA(n_components=dim, random_state=42)
        X_train_pca = pca.fit_transform(X_esm_train)
        X_test_pca = pca.transform(X_esm_test)
    
    # Combine PCA-reduced ESM + Base + Window
    X_train_final = np.hstack((X_train_pca, X_rest_train))
    X_test_final = np.hstack((X_test_pca, X_rest_test))
    
    # Train the Combined MLP
    mlp = MLPRegressor(hidden_layer_sizes=(512, 256, 64), alpha=0.5, max_iter=1000, early_stopping=True, random_state=42)
    mlp.fit(X_train_final, y_train)
    preds = mlp.predict(X_test_final)
    
    pearson = pearsonr(y_test, preds)[0]
    mae = mean_absolute_error(y_test, preds)
    
    total_features = dim + len(base_features) + len(window_features)
    print(f"ESM Dims: {dim:3d} | Total Model Features: {total_features:3d} | Pearson r: {pearson:.3f} | MAE: {mae:.3f} kcal/mol")
