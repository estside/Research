import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
from pathlib import Path

# 1. Load the dataset with enhanced window features
print("Loading Enhanced Windowed Feature Dataset...")
BASE_DIR = Path(__file__).resolve().parents[1]
DATASETS_DIR = BASE_DIR / "datasets"
df = pd.read_csv(DATASETS_DIR / "skempi_window_features_abs.csv")

# 2. Dynamically identify features (REMOVED DELTAS, KEPT ABSOLUTE VALUES)
base_features = ['wt_vol', 'mut_vol', 'wt_hydro', 'mut_hydro', 'loc']
window_features = [col for col in df.columns if col.startswith('L') or col.startswith('R')]
all_features = base_features + window_features

X = df[all_features]
y = df['ddG']
groups = df['#Pdb']

print(f"Total features being used: {len(all_features)}")
print(f"Features: {all_features}")

# 3. Grouped Split (80% Train, 20% Test)
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups=groups))

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4. Train the Random Forest
print("\nTraining Random Forest with Absolute Context...")
model = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42)
model.fit(X_train, y_train)

# 5. Predictions and Evaluation
preds = model.predict(X_test)

mse = mean_squared_error(y_test, preds)
mae = mean_absolute_error(y_test, preds)
r2 = r2_score(y_test, preds)
p_corr, _ = pearsonr(y_test, preds)

print("\n" + "="*30)
print("ABSOLUTE WINDOWED MODEL RESULTS")
print("="*30)
print(f"MSE: {mse:.3f}")
print(f"MAE: {mae:.3f} kcal/mol")
print(f"R^2: {r2:.3f}")
print(f"Pearson r: {p_corr:.3f}")
print("="*30)

# 6. Feature Importance Analysis
importances = pd.DataFrame({
    'feature': all_features,
    'importance': model.feature_importances_
}).sort_values(by='importance', ascending=False)

print("\nTop 10 Most Important Features:")
print(importances.head(10))
