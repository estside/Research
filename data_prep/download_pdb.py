import pandas as pd
from Bio.PDB import PDBList
from pathlib import Path
import os
from concurrent.futures import ThreadPoolExecutor

print("--- Initializing Multi-threaded PDB Downloader ---")

# 1. Setup Paths
BASE_DIR = Path(__file__).resolve().parents[1]
DATASETS_DIR = BASE_DIR / "datasets"
PDB_DIR = BASE_DIR / "pdb_files"
PDB_DIR.mkdir(parents=True, exist_ok=True)

# 2. Load the SKEMPI Dataset
csv_path = DATASETS_DIR / "skempi_cleaned_single_muts.csv"
df = pd.read_csv(csv_path)

# 3. Extract Unique PDB IDs
df['base_pdb'] = df['#Pdb'].str[:4].str.lower()
unique_pdbs = df['base_pdb'].unique()
print(f"Found {len(unique_pdbs)} unique protein complexes to download.")

# 4. Define the Download Function
def download_single_pdb(pdb_id):
    """Function to be executed by threads."""
    pdbl = PDBList(verbose=False) # Keep it quiet to avoid messy logs
    final_pdb_path = PDB_DIR / f"{pdb_id}.pdb"
    ent_file = PDB_DIR / f"pdb{pdb_id}.ent"

    if final_pdb_path.exists():
        return f"[{pdb_id.upper()}] already exists. Skipping."

    try:
        # Biopython downloads as 'pdbXXXX.ent'
        pdbl.retrieve_pdb_file(pdb_id, pdir=str(PDB_DIR), file_format="pdb")
        
        # Rename logic
        if ent_file.exists():
            ent_file.replace(final_pdb_path)
            return f"[{pdb_id.upper()}] Downloaded and renamed."
        return f"[{pdb_id.upper()}] Downloaded, but .ent file not found for renaming."
        
    except Exception as e:
        return f"Failed to download {pdb_id.upper()}: {e}"

# 5. Execute with ThreadPoolExecutor
# max_workers=10 is a safe start; going too high might get your IP throttled by RCSB.
MAX_WORKERS = 10 

print(f"Starting downloads with {MAX_WORKERS} threads...")

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    results = list(executor.map(download_single_pdb, unique_pdbs))

# Print a summary of the first few and last few results
for res in results[:5]: print(res)
print("...")
for res in results[-5:]: print(res)

print(f"\n✅ All tasks processed! Check '{PDB_DIR.name}/' for your files.")