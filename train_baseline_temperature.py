import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

# 1. Load the engineered dataset
print("Loading ML-ready dataset...")
df = pd.read_csv('skempi_ml_baseline_features.csv')
df["delta_temp"] = df["Temperature"] - 298
# print(df["Temperature"].value_counts())

# Define our inputs (X) and target (y)
features = ['delta_hydro', 'delta_vol', 'delta_charge', 'iMutation_Location_encoded', 'delta_temp'] if 'delta_temp' in df.columns else ['delta_hydro', 'delta_vol', 'delta_charge', 'iMutation_Location_encoded']
target = 'ddG'
groups = df['#Pdb'] # We use this to prevent data leakage!

X = df[features]
y = df[target]

# 2. Split the data rigorously (80% Train, 20% Test) grouped by PDB ID
print("Splitting data by Protein Complex to prevent leakage...")
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups=groups))

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

print(f"Training on {len(X_train)} mutations. Testing on {len(X_test)} mutations.\n")

# 3. Model 1: Linear Regression
print("--- Training Linear Regression with Temperature ---")
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_predictions = lr_model.predict(X_test)

# 4. Model 2: Random Forest (Can capture non-linear relationships)
print("--- Training Random Forest with Temperature ---")
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)

# 5. Evaluation Function
def evaluate_model(name, y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Pearson correlation is the gold standard metric in protein stability papers
    pearson_corr, _ = pearsonr(y_true, y_pred)
    
    print(f"Results for {name}:")
    print(f"  MSE: {mse:.3f} (Lower is better)")
    print(f"  MAE: {mae:.3f} kcal/mol (Average error margin)")
    print(f"  R^2: {r2:.3f} (Closer to 1 is better)")
    print(f"  Pearson r: {pearson_corr:.3f} (Closer to 1 is better)\n")

# 6. Print the Results!
evaluate_model("Linear Regression", y_test, lr_predictions)
evaluate_model("Random Forest", y_test, rf_predictions)

# Optional: Look at feature importance from the Random Forest
print("Random Forest Feature Importance with temperature:")
for feature, importance in zip(features, rf_model.feature_importances_):
    print(f"  {feature}: {importance:.3f}")


# plt.scatter(df["Temperature"], df["ddG"])
# plt.xlabel("Temperature")
# plt.ylabel("ddG")
# plt.show()