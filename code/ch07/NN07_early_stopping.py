import torch
import torch.nn as nn
import numpy as np

# Simulate early stopping
np.random.seed(42)
train_loss = np.exp(-np.linspace(0, 3, 100)) + np.random.randn(100)*0.05
val_loss = np.exp(-np.linspace(0, 2, 100)) + np.linspace(0, 0.3, 100) + np.random.randn(100)*0.05

best_val = float("inf")
best_epoch = 0
patience = 10
wait = 0

for epoch in range(100):
    if val_loss[epoch] < best_val:
        best_val = val_loss[epoch]
        best_epoch = epoch
        wait = 0
    else:
        wait += 1
    if wait >= patience:
        print(f"Early stopping at epoch {epoch} (best was epoch {best_epoch})")
        break

print(f"Best val loss: {best_val:.4f} at epoch {best_epoch}")
