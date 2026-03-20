import torch
from torch_geometric.data import Data
from Bio.PDB import PDBParser
import numpy as np
from pathlib import Path
import scipy.spatial.distance as dist

def create_protein_graph(pdb_filepath, distance_threshold=8.0):
    """
    Parses a PDB file and converts it into a PyTorch Geometric 3D Graph.
    Nodes = Alpha Carbons (Ca)
    Edges = Residues within `distance_threshold` Angstroms.
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_filepath)
    
    # 1. Extract Alpha-Carbon Coordinates
    ca_coordinates = []
    residue_ids = []
    
    # Iterate through the first model and all chains/residues
    for model in structure:
        for chain in model:
            for residue in chain:
                # Check if it's a standard amino acid and has an Alpha-Carbon
                if residue.has_id('CA') and residue.get_resname() != 'HOH':
                    ca_atom = residue['CA']
                    ca_coordinates.append(ca_atom.get_coord())
                    residue_ids.append(residue.get_resname())
                    
    if len(ca_coordinates) == 0:
        raise ValueError(f"No Alpha-Carbons found in {pdb_filepath}")

    # Convert coordinates to a PyTorch tensor
    pos = torch.tensor(np.array(ca_coordinates), dtype=torch.float)
    
    # 2. Build the Edges (Distance Matrix)
    # Calculate all pairwise distances between the Ca atoms
    distance_matrix = dist.squareform(dist.pdist(ca_coordinates))
    
    # Find indices where distance is less than our threshold (and ignore self-loops where dist == 0)
    edge_sources, edge_targets = np.where((distance_matrix < distance_threshold) & (distance_matrix > 0))
    
    # PyTorch Geometric expects edges in shape [2, num_edges]
    edge_index = torch.tensor(np.array([edge_sources, edge_targets]), dtype=torch.long)
    
    # 3. Compute Edge Weights (Actual physical distances in 3D space)
    edge_weights = distance_matrix[edge_sources, edge_targets]
    edge_attr = torch.tensor(edge_weights, dtype=torch.float).unsqueeze(1)
    
    # 4. Dummy Node Features (We will inject the 50D ESM-2 embeddings here later)
    # For now, let's just create a placeholder feature vector of size [num_nodes, 50]
    num_nodes = pos.shape[0]
    x = torch.zeros((num_nodes, 50), dtype=torch.float) 
    
    # 5. Construct the PyTorch Geometric Data Object
    graph_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, pos=pos)
    
    return graph_data, residue_ids

if __name__ == "__main__":
    # --- Quick Test ---
    # Make sure to place at least one sample .pdb file in a 'pdb_files/' folder to test this!
    BASE_DIR = Path(__file__).resolve().parents[1]
    sample_pdb = BASE_DIR / "pdb_files" / "1a22.pdb" 
    
    # Create a dummy PDB file if it doesn't exist just for the sake of not crashing
    if not sample_pdb.exists():
        print(f"Please place a valid PDB file at: {sample_pdb}")
    else:
        print(f"Parsing 3D Structure from: {sample_pdb.name}")
        graph, res_names = create_protein_graph(sample_pdb)
        
        print("\n--- 3D Graph Constructed Successfully! ---")
        print(f"Number of Nodes (Amino Acids): {graph.num_nodes}")
        print(f"Number of Edges (Interactions < 8A): {graph.num_edges}")
        print(f"Node Feature Matrix Shape: {graph.x.shape}")
        print(f"Coordinate Matrix Shape: {graph.pos.shape}")