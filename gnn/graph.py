import torch
import numpy as np
from Bio.PDB import PDBParser
from torch_geometric.data import Data

def create_protein_graph(pdb_filepath, distance_threshold=8.0):
    """
    Reads a .pdb file and converts the 3D coordinates into a PyTorch Geometric Graph.
    Connects amino acids (nodes) if their Alpha-Carbons are within the distance threshold.
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_filepath)
    
    # Extract Alpha-Carbon (CA) coordinates
    coords = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if 'CA' in residue:
                    coords.append(residue['CA'].get_coord())
                    
    coords = np.array(coords)
    if len(coords) == 0:
        raise ValueError("No Alpha-Carbon (CA) atoms found in the uploaded PDB file.")
        
    # Calculate pairwise distances
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    distances = np.linalg.norm(diff, axis=-1)
    
    # Create edges based on the distance threshold (8 Angstroms)
    edge_index = []
    edge_attr = []
    
    for i in range(len(coords)):
        for j in range(len(coords)):
            if i != j and distances[i, j] < distance_threshold:
                edge_index.append([i, j])
                edge_attr.append([distances[i, j]]) # The actual physical distance
                
    # Format for PyTorch Geometric
    if len(edge_index) > 0:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    else:
        # Edge case: No nodes are within 8A of each other
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 1), dtype=torch.float)
    
    # Create dummy 50D node features (We inject the ESM-2 vector globally later)
    x = torch.zeros((len(coords), 50), dtype=torch.float)
    
    graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return graph, coords