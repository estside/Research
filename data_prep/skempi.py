import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATASETS_DIR = BASE_DIR / "datasets"

# 1. Load the dataset WITHOUT skiprows
df = pd.read_csv(DATASETS_DIR / "skempi.csv")

# 2. BULLETPROOFING: Strip invisible leading/trailing spaces from all column names
df.columns = df.columns.str.strip()

# Print the columns just to be 100% sure what Pandas is seeing
print("Actual columns in your file:", df.columns.tolist())

# 3. Keep ONLY single mutations
# If this still throws an error, check the printed list above to see the exact spelling!
df = df[~df['Mutation(s)_cleaned'].astype(str).str.contains(',')]

# 4. Clean the Temperature column
df['Temperature'] = df['Temperature'].astype(str).str.extract(r'(\d+)').astype(float)

# 5. Calculate the target variable: ddG
R = 1.987e-3 # Ideal gas constant
df['ddG'] = R * df['Temperature'] * np.log(df['Affinity_mut_parsed'] / df['Affinity_wt_parsed'])

# 6. Encode mutation locations automatically
df['iMutation_Location_encoded'] = pd.factorize(df['iMutation_Location(s)'].astype(str).str.strip())[0]

# 7. Keep the relevant columns
columns_to_keep = [
    '#Pdb',                     
    'Protein 1', 
    'Protein 2', 
    'Mutation(s)_cleaned', 
    'iMutation_Location_encoded',
    'Temperature', 
    'ddG'
]

# Ensure we only keep columns that actually exist to avoid another KeyError
existing_columns = [col for col in columns_to_keep if col in df.columns]
final_df = df[existing_columns]

# 8. Drop rows with missing ddG values
final_df = final_df.dropna(subset=['ddG'])
# Save the cleaned dataframe to a new CSV file
output_filename = DATASETS_DIR / "skempi_cleaned_single_muts.csv"
final_df.to_csv(output_filename, index=False)

print(f"\nData successfully saved to {output_filename}")

print("\nSuccess! Here is your cleaned data:")
print(final_df.head())
