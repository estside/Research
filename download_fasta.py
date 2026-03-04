import pandas as pd
import requests
import os
import time

# 1. Create a neat folder to store all the downloaded sequences
folder_name = "fasta_files"
if not os.path.exists(folder_name):
    os.makedirs(folder_name)
    print(f"Created folder: {folder_name}/")

# 2. Read your newly cleaned dataset
print("Reading skempi_cleaned_single_muts.csv...")
df = pd.read_csv('skempi_cleaned_single_muts.csv')

# 3. Extract just the 4-letter PDB IDs
# The #Pdb column looks like "1CSE_E_I". We split by '_' and grab the first part.
df['PDB_ID'] = df['#Pdb'].str.split('_').str[0]

# Get a list of unique PDB IDs so we don't waste time downloading duplicates
unique_pdbs = df['PDB_ID'].dropna().unique()
print(f"Found {len(unique_pdbs)} unique PDB complexes to download.\n")

# 4. Loop through and download each FASTA file
for pdb_id in unique_pdbs:
    file_path = os.path.join(folder_name, f"{pdb_id}.fasta")
    
    # Skip the download if the file is already there (great if your script gets interrupted!)
    if os.path.exists(file_path):
        print(f"[{pdb_id}] already exists. Skipping...")
        continue
        
    # The official PDB API endpoint for FASTA files
    url = f"https://www.rcsb.org/fasta/entry/{pdb_id}/display"
    
    print(f"Downloading [{pdb_id}]...")
    response = requests.get(url)
    
    if response.status_code == 200:
        with open(file_path, 'w') as f:
            f.write(response.text)
    else:
        print(f"  -> Failed to download {pdb_id} (Status Code: {response.status_code})")
        
    # Pause for 0.2 seconds between downloads so the PDB servers don't block our IP address
    time.sleep(0.2)

print("\nAll FASTA sequences successfully downloaded!")