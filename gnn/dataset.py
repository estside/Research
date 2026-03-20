import torch
from torch_geometric.data import Dataset, Data
import pandas as pd
from pathlib import Path
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Import the graph constructor we wrote earlier
from gnn import create_protein_graph

class SKEMPIGraphDataset(Dataset):
    def __init__(self, root_dir, transform=None, pre_transform=None):
        """
        Custom PyTorch Geometric Dataset for 3D Protein Graphs.
        Stitches the PDB structures together with the Delta Delta G targets and ESM-2 features.
        """
        super(SKEMPIGraphDataset, self).__init__(root_dir, transform, pre_transform)
        
        print("Initializing 3D Graph Dataset...")
        self.base_dir = Path(__file__).resolve().parents[1]
        self.pdb_dir = self.base_dir / "pdb_files"
        
        # 1. Load the merged dataset (Target ddG + ESM-2 Features)
        esm_df = pd.read_csv(self.base_dir / "datasets" / "skempi_esm2_features.csv")
        base_df = pd.read_csv(self.base_dir / "datasets" / "skempi_cleaned_single_muts.csv")
        self.df = pd.merge(esm_df, base_df, on=['#Pdb', 'ddG'], how='inner')
        
        # 2. Compress the ESM-2 features down to 50D (just like your 1D pipeline)
        print("Compressing ESM-2 Features via PCA...")
        esm_cols = [c for c in self.df.columns if str(c).startswith('esm_dim_')]
        X_esm = self.df[esm_cols].values
        
        scaler = StandardScaler()
        X_esm_scaled = scaler.fit_transform(X_esm)
        
        pca = PCA(n_components=50, random_state=42)
        self.esm_50d = pca.fit_transform(X_esm_scaled)
        
        # 3. Filter out rows where the PDB file failed to download
        valid_indices = []
        for idx, row in self.df.iterrows():
            pdb_id = str(row['#Pdb'])[:4].lower()
            if (self.pdb_dir / f"{pdb_id}.pdb").exists():
                valid_indices.append(idx)
                
        self.df = self.df.iloc[valid_indices].reset_index(drop=True)
        self.esm_50d = self.esm_50d[valid_indices]
        print(f"Dataset ready! Successfully matched {len(self.df)} mutations with 3D PDB files.")

    def len(self):
        return len(self.df)

    def get(self, idx):
        """
        Fetches a single data point: The WT Graph, the MT Graph, and the Target ddG.
        """
        row = self.df.iloc[idx]
        pdb_id = str(row['#Pdb'])[:4].lower()
        pdb_filepath = self.pdb_dir / f"{pdb_id}.pdb"
        
        # 1. Target Value
        y = torch.tensor([[row['ddG']]], dtype=torch.float)
        
        # 2. Build the Wild-Type Graph from the PDB file
        wt_graph, _ = create_protein_graph(pdb_filepath, distance_threshold=8.0)
        
        # 3. Create the Mutant Graph (For now, structurally identical, but we inject the ESM feature)
        # We clone the WT graph so we don't accidentally overwrite it in memory
        mt_graph = wt_graph.clone()
        
        # 4. Inject the 50D ESM-2 Delta vector as a global graph property
        esm_tensor = torch.tensor(self.esm_50d[idx], dtype=torch.float).unsqueeze(0)
        
        # We attach the ESM-2 embeddings to the graph objects
        # We will use these inside the neural network during the global pooling step
        wt_graph.esm_feature = torch.zeros_like(esm_tensor) # WT is baseline (0)
        mt_graph.esm_feature = esm_tensor                   # MT carries the evolutionary shock
        
        wt_graph.y = y
        mt_graph.y = y
        
        return wt_graph, mt_graph

if __name__ == "__main__":
    # --- Quick Test ---
    # This will parse your CSVs and check your PDB folder to build the index!
    dataset = SKEMPIGraphDataset(root_dir=".")
    
    if len(dataset) > 0:
        wt, mt = dataset[0]
        print("\n--- PyTorch Dataset Validated ---")
        print(f"Target ddG: {wt.y.item():.4f} kcal/mol")
        print(f"WT Graph Nodes: {wt.num_nodes}, MT Graph Nodes: {mt.num_nodes}")
        print(f"MT Global ESM-2 Feature Shape: {mt.esm_feature.shape}")