import pandas as pd
import numpy as np
import os
from Bio import SeqIO
from pathlib import Path

# 1. Biophysical Properties
hydrophobicity = {'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5, 'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5, 'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6, 'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2}
volume = {'G': 60, 'A': 88, 'S': 89, 'C': 108, 'D': 111, 'T': 116, 'N': 114, 'P': 112, 'V': 140, 'E': 138, 'Q': 143, 'H': 153, 'M': 162, 'I': 166, 'L': 166, 'K': 168, 'R': 173, 'F': 189, 'Y': 193, 'W': 227}

def get_prop(aa, prop_dict):
    return prop_dict.get(aa.upper(), 0)

# 2. Configuration
WINDOW_SIZE = 3 
BASE_DIR = Path(__file__).resolve().parents[1]
DATASETS_DIR = BASE_DIR / "datasets"
FASTA_DIR = BASE_DIR / "fasta_files"

df = pd.read_csv(DATASETS_DIR / "skempi_cleaned_single_muts.csv")
new_rows = []

print("Extracting Enhanced Window Features (Deltas + Absolute Values)...")

for idx, row in df.iterrows():
    parts = row['#Pdb'].split('_')
    pdb_code = parts[0]
    mutant_chain_id = parts[1]
    
    import re
    match = re.search(r'([a-zA-Z])(\d+)([a-zA-Z])', str(row['Mutation(s)_cleaned']))
    if not match: continue
    wt_aa, pos, mut_aa = match.groups()
    pos = int(pos)
    
    fasta_path = FASTA_DIR / f"{pdb_code}.fasta"
    if not fasta_path.exists(): continue
    
    sequence = ""
    for record in SeqIO.parse(str(fasta_path), "fasta"):
        if f"Chain {mutant_chain_id}" in record.description or f"|{mutant_chain_id}|" in record.description:
            sequence = str(record.seq)
            break
    
    if not sequence: continue
    
    idx_in_seq = pos - 1 
    if idx_in_seq >= len(sequence): continue
    
    start = max(0, idx_in_seq - WINDOW_SIZE)
    end = min(len(sequence), idx_in_seq + WINDOW_SIZE + 1)
    
    left_neighbors = sequence[start:idx_in_seq]
    right_neighbors = sequence[idx_in_seq+1:end]
    
    left_neighbors = left_neighbors.rjust(WINDOW_SIZE, 'X')
    right_neighbors = right_neighbors.ljust(WINDOW_SIZE, 'X')
    
    # Calculate Absolute Values
    wt_vol = get_prop(wt_aa, volume)
    mut_vol = get_prop(mut_aa, volume)
    wt_hydro = get_prop(wt_aa, hydrophobicity)
    mut_hydro = get_prop(mut_aa, hydrophobicity)
    
    # Build Feature Dictionary for this row
    feat = {
        '#Pdb': row['#Pdb'],
        'ddG': row['ddG'],
        'wt_vol': wt_vol,
        'mut_vol': mut_vol,
        'delta_vol': mut_vol - wt_vol,
        'wt_hydro': wt_hydro,
        'mut_hydro': mut_hydro,
        'delta_hydro': mut_hydro - wt_hydro,
        'loc': row['iMutation_Location_encoded']
    }
    
    # Add neighboring properties (Hydrophobicity)
    for i, aa in enumerate(left_neighbors):
        feat[f'L{WINDOW_SIZE-i}_hydro'] = get_prop(aa, hydrophobicity)
    for i, aa in enumerate(right_neighbors):
        feat[f'R{i+1}_hydro'] = get_prop(aa, hydrophobicity)
        
    new_rows.append(feat)

window_df = pd.DataFrame(new_rows)
window_df.to_csv(DATASETS_DIR / "skempi_window_features_abs.csv", index=False)
print(f"Success! Created {len(window_df)} windowed samples with 13 features each.")
