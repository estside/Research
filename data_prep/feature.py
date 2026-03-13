import os
# CRITICAL: Set these BEFORE any other imports to prevent deadlocks and lock-ups on Mac
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["GRPC_ENABLE_FORK_SUPPORT"] = "0"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
from pathlib import Path

import pandas as pd
import torch
import re
from Bio import SeqIO
# Move transformers import later if possible, but for now let's keep it here with the env vars set
from transformers import AutoTokenizer, EsmModel

# Limit torch threads to prevent resource contention/locks on Mac
torch.set_num_threads(1)

print("Starting ESM-2 Feature Extraction script...")

# 1. Load the 150M Pre-trained Transformer 
# This model outputs 640-dimensional embeddings (double the size of the 8M model)
model_name = "facebook/esm2_t30_150M_UR50D"
print(f"Loading larger model and tokenizer: {model_name}...")

# Using your specific token to bypass download restrictions
hf_token = "hf_e42133VdIZaWvNzvURRQ"

tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
model = EsmModel.from_pretrained(model_name, token=hf_token)
print("150M Model loaded successfully.")
model.eval()
# 2. Load your cleaned data
BASE_DIR = Path(__file__).resolve().parents[1]
DATASETS_DIR = BASE_DIR / "datasets"
FASTA_DIR = BASE_DIR / "fasta_files"
df = pd.read_csv(DATASETS_DIR / "skempi_cleaned_single_muts.csv")

embeddings_list = []

print("Extracting Self-Attention Embeddings via ESM-2...")

# Process the full dataset
for idx, row in df.iterrows():
    pdb_info = row['#Pdb'].split('_')
    pdb_code = pdb_info[0]
    
    mut_str = str(row['Mutation(s)_cleaned']).strip()
    
    # Improved Parse: Try WT-Chain-Pos-Mut (e.g., LI38G)
    match = re.search(r'^([a-zA-Z])([a-zA-Z])(\d+)([a-zA-Z])$', mut_str)
    if match:
        wt_aa, mutant_chain_id, pos, mut_aa = match.groups()
    else:
        # Fallback to WT-Pos-Mut (e.g., L38G)
        match = re.search(r'([a-zA-Z])(\d+)([a-zA-Z])', mut_str)
        if not match: 
            print(f"Skipping {mut_str}: Regex failed")
            continue
        wt_aa, pos, mut_aa = match.groups()
        mutant_chain_id = pdb_info[1] # Try first chain
    
    pos = int(pos)
    
    # Load FASTA sequence
    fasta_path = FASTA_DIR / f"{pdb_code}.fasta"
    if not fasta_path.exists(): 
        # print(f"Skipping {pdb_code}: FASTA not found")
        continue
    
    wt_sequence = ""
    for record in SeqIO.parse(str(fasta_path), "fasta"):
        desc = record.description.upper()
        # Robust chain matching (Auth or Index)
        if f"CHAIN {mutant_chain_id.upper()}" in desc or f"AUTH {mutant_chain_id.upper()}" in desc or f"|{mutant_chain_id.upper()}|" in desc:
            wt_sequence = str(record.seq)
            break
            
    if not wt_sequence: 
        # print(f"Skipping {pdb_code}: Chain {mutant_chain_id} not found in FASTA")
        continue
    
    if pos > len(wt_sequence):
        # print(f"Skipping {pdb_code}: Pos {pos} out of range ({len(wt_sequence)})")
        continue
    
    # Verify WT matches sequence (Safety check)
    if wt_sequence[pos - 1] != wt_aa.upper():
        # Sometimes PDB numbering doesn't match sequence index. 
        # For now, let's just skip if it's a mismatch.
        # print(f"Skipping {pdb_code}: Sequence mismatch at {pos} ({wt_sequence[pos-1]} vs {wt_aa})")
        continue
    
    # Create the Mutant Sequence
    seq_list = list(wt_sequence)
    seq_list[pos - 1] = mut_aa.upper()
    mt_sequence = "".join(seq_list)
    
    # 3. Pass through the Transformer
    with torch.no_grad():
        # Tokenize (Hugging Face ESM uses standard BERT-style tokenization)
        wt_inputs = tokenizer(wt_sequence, return_tensors="pt")
        mt_inputs = tokenizer(mt_sequence, return_tensors="pt")
        
        # Get hidden states
        wt_outputs = model(**wt_inputs)
        mt_outputs = model(**mt_inputs)
        
        # Extract the vector exactly at the mutation site
        # pos + 1 because of <cls> token
        wt_vector = wt_outputs.last_hidden_state[0, pos] 
        mt_vector = mt_outputs.last_hidden_state[0, pos]
        
        # Calculate the delta (Mutation context shift)
        delta_vector = (mt_vector - wt_vector).numpy()
    
    # Store the results
    feature_dict = {'#Pdb': row['#Pdb'], 'Mutation': mut_str, 'ddG': row['ddG']}
    
    # Unpack dimensions
    for i, val in enumerate(delta_vector):
        feature_dict[f'esm_dim_{i}'] = val
        
    embeddings_list.append(feature_dict)
    if len(embeddings_list) % 5 == 0:
        print(f"Computed {len(embeddings_list)} embeddings...")

# 4. Save the new high-dimensional dataset
embeddings_df = pd.DataFrame(embeddings_list)
embeddings_df.to_csv(DATASETS_DIR / "skempi_esm2_features.csv", index=False)
print(f"Successfully extracted embeddings for {len(embeddings_df)} samples!")
