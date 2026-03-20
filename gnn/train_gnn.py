import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
import numpy as np
from scipy.stats import pearsonr
import time

# Import your custom modules
from dataset import SKEMPIGraphDataset
from fusion_network import SiameseDeltaGNN

print("--- Initializing 1D-to-3D Fusion Training Pipeline ---")

# 1. Setup Device (Optimized for Colab NVIDIA T4)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Hardware Acceleration: {device}")

# 2. Load Dataset and Split
dataset = SKEMPIGraphDataset(root_dir=".")

# 80/20 Train-Test Split
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

print(f"Training Samples: {len(train_dataset)}")
print(f"Testing Samples: {len(test_dataset)}")

# Batch size kept conservative to prevent Colab GPU Out-of-Memory errors
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# 3. Initialize Model, Optimizer, and Loss
model = SiameseDeltaGNN(node_feature_dim=50, hidden_dim=128).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
criterion = torch.nn.MSELoss()

# 4. The Training Loop
# 4. The Training Loop
epochs = 10
best_test_mae = float('inf')  # Track the best performance

print("\n--- Starting Training ---")
start_time = time.time()

for epoch in range(epochs):
    model.train()
    total_loss = 0
    
    for wt_batch, mt_batch in train_loader:
        wt_batch, mt_batch = wt_batch.to(device), mt_batch.to(device)
        optimizer.zero_grad()
        
        predictions = model(wt_batch, mt_batch)
        loss = criterion(predictions, wt_batch.y)
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * wt_batch.num_graphs
        
    avg_train_loss = total_loss / len(train_dataset)
    
    # Validation
    model.eval()
    total_test_error = 0
    with torch.no_grad():
        for wt_batch, mt_batch in test_loader:
            wt_batch, mt_batch = wt_batch.to(device), mt_batch.to(device)
            predictions = model(wt_batch, mt_batch)
            total_test_error += F.l1_loss(predictions, wt_batch.y, reduction='sum').item()
            
    avg_test_mae = total_test_error / len(test_dataset)
    
    print(f"Epoch {epoch+1:02d}/{epochs} | Train MSE: {avg_train_loss:.4f} | Test MAE: {avg_test_mae:.4f}")

    # --- NEW: SAVE CHECKPOINT EVERY EPOCH ---
    checkpoint = {
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_train_loss,
    }
    
    # Save the "Last" model
    torch.save(checkpoint, "checkpoint_last.pth")
    
    # Save the "Best" model if performance improved
    if avg_test_mae < best_test_mae:
        best_test_mae = avg_test_mae
        torch.save(checkpoint, "checkpoint_best.pth")
        print(f"⭐ New Best Model Saved (MAE: {best_test_mae:.4f})")

print(f"\nTraining Complete in {(time.time() - start_time)/60:.2f} minutes.")