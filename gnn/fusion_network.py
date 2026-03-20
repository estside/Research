import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from torch.nn import Linear, Dropout
from torch_geometric.data import Data, Batch

class SiameseDeltaGNN(torch.nn.Module):
    def __init__(self, node_feature_dim=50, hidden_dim=128):
        super(SiameseDeltaGNN, self).__init__()
        
        self.gat1 = GATConv(node_feature_dim, hidden_dim, heads=4, concat=False)
        self.gat2 = GATConv(hidden_dim, hidden_dim, heads=4, concat=False)
        
        # UPDATE: 128 (3D Graph Delta) + 50 (1D ESM-2 Delta) = 178 total features
        self.fc1 = Linear(hidden_dim + 50, 64)
        self.dropout = Dropout(p=0.3)
        self.fc2 = Linear(64, 1)

    def forward_once(self, x, edge_index, edge_attr, batch):
        """Passes a single graph through the GAT layers and pools it."""
        x = self.gat1(x, edge_index, edge_attr)
        x = F.relu(x)
        
        x = self.gat2(x, edge_index, edge_attr)
        x = F.relu(x)
        
        # Compress the entire 3D graph into a single mathematical vector
        x = global_mean_pool(x, batch)
        return x

    def forward(self, wt_data, mt_data):
        wt_embedding = self.forward_once(wt_data.x, wt_data.edge_index, wt_data.edge_attr, wt_data.batch)
        mt_embedding = self.forward_once(mt_data.x, mt_data.edge_index, mt_data.edge_attr, mt_data.batch)
        
        # Isolate the 3D structural shock
        delta_graph = mt_embedding - wt_embedding
        
        # THE FUSION: Concatenate the 3D shock with the 1D ESM-2 evolutionary shock
        combined_delta = torch.cat([delta_graph, mt_data.esm_feature], dim=1)
        
        out = self.fc1(combined_delta)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out

if __name__ == "__main__":
    print("Initializing Siamese 1D-to-3D Fusion Network...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize the model
    model = SiameseDeltaGNN(node_feature_dim=50, hidden_dim=128).to(device)
    
    # Create Dummy WT Graph
    num_nodes = 372
    num_edges = 3474
    wt_data = Data(
        x=torch.rand((num_nodes, 50)).to(device),
        edge_index=torch.randint(0, num_nodes, (2, num_edges)).to(device),
        edge_attr=torch.rand((num_edges, 1)).to(device),
        batch=torch.zeros(num_nodes, dtype=torch.long).to(device)
    )
    
    # Create Dummy MT Graph (Slightly different features to simulate a mutation)
    mt_data = Data(
        x=torch.rand((num_nodes, 50)).to(device),
        edge_index=torch.randint(0, num_nodes, (2, num_edges)).to(device),
        edge_attr=torch.rand((num_edges, 1)).to(device),
        batch=torch.zeros(num_nodes, dtype=torch.long).to(device)
    )
    
    # Run both graphs through the Siamese network!
    prediction = model(wt_data, mt_data)
    
    print("\n--- Siamese Architecture Validated ---")
    print(f"Predicted Output Shape: {prediction.shape} (Should be [1, 1])")
    print(f"Sample ddG Prediction: {prediction.item():.4f} kcal/mol")