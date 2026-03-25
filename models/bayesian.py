import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

print("1. Loading and Merging Datasets...")
# Setup paths exactly as in advance.py
BASE_DIR = Path(__file__).resolve().parents[1]
DATASETS_DIR = BASE_DIR / "datasets"

# Load the data
esm_df = pd.read_csv(DATASETS_DIR / "skempi_esm2_features.csv")
window_df = pd.read_csv(DATASETS_DIR / "skempi_window_features.csv")
merged_df = pd.merge(esm_df, window_df, on=['#Pdb', 'ddG'], how='inner')

print("2. Isolating and Compressing ESM-2 Features (PCA)...")
# Extract just the 640 ESM dimensions
esm_features = [c for c in merged_df.columns if c.startswith('esm_dim_')]
X_esm = merged_df[esm_features].values

# Scale and apply 50-component PCA
scaler_esm = StandardScaler()
X_esm_scaled = scaler_esm.fit_transform(X_esm)

pca = PCA(n_components=50, random_state=42)
X_esm_pca_50d = pca.fit_transform(X_esm_scaled)

print("3. Discretizing Continuous Variables for the PGM...")
# A. Cluster the 50D ESM-2 Manifold into 3 discrete Structural States
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
merged_df['ESM2_State'] = kmeans.fit_predict(X_esm_pca_50d)
merged_df['ESM2_State'] = merged_df['ESM2_State'].map({0: 'State_A', 1: 'State_B', 2: 'State_C'})

# B. Discretize the biophysical columns using qcut (Low, Medium, High)
# Include all flanking residues
columns_to_discretize = [
    'delta_vol', 'delta_hydro', 
    'L3_hydro', 'L2_hydro', 'L1_hydro', 
    'R1_hydro', 'R2_hydro', 'R3_hydro', 
    'ddG'
]

for col in columns_to_discretize:
    merged_df[f'{col}_discrete'] = pd.qcut(merged_df[col], q=3, labels=['Low', 'Medium', 'High'], duplicates='drop')

# Make sure the spatial location 'loc' is treated as a distinct categorical string
merged_df['loc'] = merged_df['loc'].astype(str)

print("4. Building the Causal Directed Acyclic Graph (DAG)...")
# Define the causal architecture based on biological priors
causal_model = BayesianNetwork([
    # Origin -> Micro-Environment
    ('loc', 'L3_hydro_discrete'), ('loc', 'L2_hydro_discrete'), ('loc', 'L1_hydro_discrete'),
    ('loc', 'R1_hydro_discrete'), ('loc', 'R2_hydro_discrete'), ('loc', 'R3_hydro_discrete'),
    
    ('delta_vol_discrete', 'L3_hydro_discrete'), ('delta_vol_discrete', 'L2_hydro_discrete'), 
    ('delta_vol_discrete', 'L1_hydro_discrete'), ('delta_vol_discrete', 'R1_hydro_discrete'), 
    ('delta_vol_discrete', 'R2_hydro_discrete'), ('delta_vol_discrete', 'R3_hydro_discrete'),
    
    ('delta_hydro_discrete', 'L3_hydro_discrete'), ('delta_hydro_discrete', 'L2_hydro_discrete'), 
    ('delta_hydro_discrete', 'L1_hydro_discrete'), ('delta_hydro_discrete', 'R1_hydro_discrete'), 
    ('delta_hydro_discrete', 'R2_hydro_discrete'), ('delta_hydro_discrete', 'R3_hydro_discrete'),
    
    # Micro-Environment -> Macro-Structure
    ('L3_hydro_discrete', 'ESM2_State'), ('L2_hydro_discrete', 'ESM2_State'), 
    ('L1_hydro_discrete', 'ESM2_State'), ('R1_hydro_discrete', 'ESM2_State'), 
    ('R2_hydro_discrete', 'ESM2_State'), ('R3_hydro_discrete', 'ESM2_State'),
    
    # Origin -> Macro-Structure
    ('loc', 'ESM2_State'),
    ('delta_vol_discrete', 'ESM2_State'),
    ('delta_hydro_discrete', 'ESM2_State'),
    
    # All paths -> Final Target Delta Delta G
    ('ESM2_State', 'ddG_discrete'),
    ('L3_hydro_discrete', 'ddG_discrete'), ('L2_hydro_discrete', 'ddG_discrete'), 
    ('L1_hydro_discrete', 'ddG_discrete'), ('R1_hydro_discrete', 'ddG_discrete'), 
    ('R2_hydro_discrete', 'ddG_discrete'), ('R3_hydro_discrete', 'ddG_discrete'),
    ('loc', 'ddG_discrete')
])
print("5. Training the Bayesian Network...")
# Learn the Conditional Probability Tables (CPTs)
causal_model.fit(merged_df, estimator=MaximumLikelihoodEstimator)
print("   -> Model trained successfully!")

print(f"   -> Valid DAG? {causal_model.check_model()}")

print("\n6. Running Causal Inference (Do-Calculus)...")
inference = VariableElimination(causal_model)

# Example Hypothesis: If the mutation forces a massive change in L2 hydrophobicity 
# and sits in the Core (Location 0), what is the probability distribution of the thermodynamic shift?
query_result = inference.query(
    variables=['ddG_discrete'], 
    evidence={'L2_hydro_discrete': 'High', 'loc': '0'}
)

print(query_result)
print("\n" + "="*50)
print("HYPOTHESIS TESTING: L2 vs R2 Causal Weight")
print("="*50)

# Scenario A: High L2 shock, Low R2 shock (Core location)
query_L2_dominant = inference.query(
    variables=['ddG_discrete'], 
    evidence={'loc': '0', 'L2_hydro_discrete': 'High', 'R2_hydro_discrete': 'Low'}
)

# Scenario B: Low L2 shock, High R2 shock (Core location)
query_R2_dominant = inference.query(
    variables=['ddG_discrete'], 
    evidence={'loc': '0', 'L2_hydro_discrete': 'Low', 'R2_hydro_discrete': 'High'}
)
# Scenario C: High L2 shock, High R2 shock (Core location)
query_R3_dominant = inference.query(
    variables=['ddG_discrete'], 
    evidence={'loc': '0', 'L2_hydro_discrete': 'High', 'R2_hydro_discrete': 'High'}
)
# Scenario D: Low L2 shock, Low R2 shock (Core location)
query_R4_dominant = inference.query(
    variables=['ddG_discrete'], 
    evidence={'loc': '0', 'L2_hydro_discrete': 'Low', 'R2_hydro_discrete': 'Low'}
)

print("\nScenario A (L2 High, R2 Low): Probability of HIGH destabilization")
print(f"{query_L2_dominant.values[2]:.4f} (or {query_L2_dominant.values[2]*100:.2f}%)")

print("\nScenario B (L2 Low, R2 High): Probability of HIGH destabilization")
print(f"{query_R2_dominant.values[2]:.4f} (or {query_R2_dominant.values[2]*100:.2f}%)")

print("\nScenario C (L2 High, R2 High): Probability of HIGH destabilization")
print(f"{query_R3_dominant.values[2]:.4f} (or {query_R3_dominant.values[2]*100:.2f}%)")

print("\nScenario D (L2 Low, R2 Low): Probability of HIGH destabilization")
print(f"{query_R4_dominant.values[2]:.4f} (or {query_R4_dominant.values[2]*100:.2f}%)")

print("\n" + "="*50)
print("EXPANDING CAUSAL QUERIES: LOC 1 AND LOC 2")
print("="*50)

# We will loop through the remaining locations
locations_to_test = ['1', '2']

for loc in locations_to_test:
    print(f"\n--- Results for Location: {loc} ---")
    
    # Scenario A: L2 High, R2 Low
    q_A = inference.query(
        variables=['ddG_discrete'], 
        evidence={'loc': loc, 'L2_hydro_discrete': 'High', 'R2_hydro_discrete': 'Low'}
    )
    
    # Scenario B: L2 Low, R2 High
    q_B = inference.query(
        variables=['ddG_discrete'], 
        evidence={'loc': loc, 'L2_hydro_discrete': 'Low', 'R2_hydro_discrete': 'High'}
    )
    
    # Scenario C: L2 High, R2 High
    q_C = inference.query(
        variables=['ddG_discrete'], 
        evidence={'loc': loc, 'L2_hydro_discrete': 'High', 'R2_hydro_discrete': 'High'}
    )
    
    # Scenario D: L2 Low, R2 Low
    q_D = inference.query(
        variables=['ddG_discrete'], 
        evidence={'loc': loc, 'L2_hydro_discrete': 'Low', 'R2_hydro_discrete': 'Low'}
    )

    # Note: pgmpy sorts discrete states alphabetically (High, Low, Medium).
    # To be absolutely safe across all updates, we can dynamically find the exact index for 'High'
    high_idx = q_A.state_names['ddG_discrete'].index('High')

    print(f"Scenario A (L2 High, R2 Low) : {q_A.values[high_idx]*100:.2f}% probability of HIGH destabilization")
    print(f"Scenario B (L2 Low, R2 High) : {q_B.values[high_idx]*100:.2f}% probability of HIGH destabilization")
    print(f"Scenario C (L2 High, R2 High): {q_C.values[high_idx]*100:.2f}% probability of HIGH destabilization")
    print(f"Scenario D (L2 Low, R2 Low)  : {q_D.values[high_idx]*100:.2f}% probability of HIGH destabilization")