import pandas as pd
from pathlib import Path

# 1. Biophysical dictionaries
hydrophobicity = {
    'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5, 
    'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5, 
    'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6, 
    'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
}

volume = {
    'G': 60, 'A': 88, 'S': 89, 'C': 108, 'D': 111, 
    'T': 116, 'N': 114, 'P': 112, 'V': 140, 'E': 138, 
    'Q': 143, 'H': 153, 'M': 162, 'I': 166, 'L': 166, 
    'K': 168, 'R': 173, 'F': 189, 'Y': 193, 'W': 227
}

charge = {
    'R': 1, 'K': 1, 'H': 0, 'D': -1, 'E': -1,
    'A': 0, 'N': 0, 'C': 0, 'Q': 0, 'G': 0, 
    'I': 0, 'L': 0, 'M': 0, 'F': 0, 'P': 0, 
    'S': 0, 'T': 0, 'W': 0, 'Y': 0, 'V': 0
}

# 2. Load the cleaned dataset
print("Loading cleaned dataset...")
BASE_DIR = Path(__file__).resolve().parents[1]
DATASETS_DIR = BASE_DIR / "datasets"
df = pd.read_csv(DATASETS_DIR / "skempi_cleaned_single_muts.csv")
print(f"Original rows loaded: {len(df)}")

# 3. THE FIX: Relaxed Regex
# This looks for ANY letter, followed by digits, followed by ANY letter.
# It ignores chain prefixes like "A_" or hidden spaces.
extracted = df['Mutation(s)_cleaned'].astype(str).str.extract(r'([a-zA-Z])(\d+)([a-zA-Z])')
df['wt_AA'] = extracted[0].str.upper()
df['Position'] = extracted[1].astype(float)
df['mut_AA'] = extracted[2].str.upper()

# Drop rows where the regex completely failed to find a mutation pattern
df = df.dropna(subset=['wt_AA', 'mut_AA'])
print(f"Rows after parsing mutations: {len(df)}")

# 4. Map the biophysical properties
df['wt_hydro'] = df['wt_AA'].map(hydrophobicity)
df['wt_vol'] = df['wt_AA'].map(volume)
df['wt_charge'] = df['wt_AA'].map(charge)

df['mut_hydro'] = df['mut_AA'].map(hydrophobicity)
df['mut_vol'] = df['mut_AA'].map(volume)
df['mut_charge'] = df['mut_AA'].map(charge)

# Calculate the DELTA (Change)
df['delta_hydro'] = df['mut_hydro'] - df['wt_hydro']
df['delta_vol'] = df['mut_vol'] - df['wt_vol']
df['delta_charge'] = df['mut_charge'] - df['wt_charge']

# 5. Define ML columns dynamically (Fixing the Temperature issue!)
ml_columns = [
    '#Pdb', 
    'delta_hydro', 
    'delta_vol', 
    'delta_charge', 
    'iMutation_Location_encoded', 
    'ddG'
]

# If you went back and added Temperature to your first script, this will include it.
# If you didn't, it will just skip it safely without crashing.
if 'Temperature' in df.columns:
    ml_columns.insert(4, 'Temperature')
    print("Temperature column found and added to ML features!")
else:
    print("\n[Note] Temperature column is missing. If you want it, add 'Temperature' to columns_to_keep in your FIRST script (skempi.py) and re-run it.")

# 6. Final cleanup and save
ml_df = df[ml_columns].dropna()
print(f"Final ML-ready rows: {len(ml_df)}")

output_name = DATASETS_DIR / "skempi_ml_baseline_features.csv"
ml_df.to_csv(output_name, index=False)

print(f"\nSuccess! Engineered features saved to {output_name}")
print(ml_df.head())
