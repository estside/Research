import torch
import numpy as np
from scipy.stats import pearsonr
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split

# Import your custom modules
from dataset import SKEMPIGraphDataset
from fusion_network import SiameseDeltaGNN

print("--- Initializing GNN Evaluation Script ---")

# 1. Setup Device
device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
print(f"Using device: {device}")

# 2. Recreate the Dataset and Test Split
dataset = SKEMPIGraphDataset(root_dir=".")
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

# IMPORTANT: Setting a manual seed ensures you get the exact same test split every time.
# (If you didn't use a seed during training, this will just evaluate a fresh, clean 20% slice)
generator = torch.Generator().manual_seed(42)
_, test_dataset = random_split(dataset, [train_size, test_size], generator=generator)

test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
print(f"Evaluating on {len(test_dataset)} test samples.")

# 3. Initialize Model Architecture
model = SiameseDeltaGNN(node_feature_dim=50, hidden_dim=128).to(device)

# 4. Load the Saved Checkpoint
checkpoint_path = "checkpoint_best.pth"
print(f"\nLoading weights from {checkpoint_path}...")

try:
    # weights_only=True is safer for loading standard PyTorch files
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    
    # Check if we saved a full checkpoint dictionary (with epoch info) or just the raw weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Successfully loaded peak model from Epoch {checkpoint.get('epoch', 'Unknown')}")
    else:
        model.load_state_dict(checkpoint)
        print("✅ Successfully loaded model state dict.")
except FileNotFoundError:
    print(f"❌ ERROR: Could not find '{checkpoint_path}'. Make sure it's in the same folder you are running this script from!")
    exit()

# 5. Run Inference
model.eval()
all_predictions = []
all_targets = []

print("Running predictions... (This should only take a few seconds)")

with torch.no_grad():
    for wt_batch, mt_batch in test_loader:
        wt_batch, mt_batch = wt_batch.to(device), mt_batch.to(device)
        
        predictions = model(wt_batch, mt_batch)
        
        # Move off GPU and flatten to 1D lists
        all_predictions.extend(predictions.cpu().numpy().flatten())
        all_targets.extend(wt_batch.y.cpu().numpy().flatten())

# 6. Calculate Statistical Metrics
correlation, p_value = pearsonr(all_targets, all_predictions)
final_mae = np.mean(np.abs(np.array(all_targets) - np.array(all_predictions)))

print("\n" + "="*50)
print(f"🏆 FINAL EVALUATION RESULTS 🏆")
print("="*50)
print(f"Final Test MAE:                 {final_mae:.4f} kcal/mol")
print(f"Final Pearson Correlation (r):  {correlation:.4f}")
print(f"Statistical Significance (p):   {p_value:.4e}")
print("="*50)