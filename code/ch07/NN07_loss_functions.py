import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# MSE, MAE, CrossEntropy comparison
pred = torch.linspace(0, 1, 100)
target = torch.tensor(0.3).expand_as(pred)

mse = nn.MSELoss()
mae = nn.L1Loss()

mse_vals = [mse(p.reshape(1), torch.tensor([0.3])).item() for p in pred]
mae_vals = [mae(p.reshape(1), torch.tensor([0.3])).item() for p in pred]

plt.figure(figsize=(10, 4))
plt.subplot(121)
plt.plot(pred, mse_vals, label="MSE")
plt.plot(pred, mae_vals, label="MAE")
plt.axvline(0.3, color="r", ls="--", label="target")
plt.legend(); plt.title("Regression Losses")

# CE loss surface
logits = torch.linspace(-5, 5, 100)
ce_vals = []
for l in logits:
    ce_vals.append(nn.CrossEntropyLoss()(l.reshape(1,1), torch.tensor([0])).item())

plt.subplot(122)
plt.plot(logits, ce_vals)
plt.axvline(0, color="r", ls="--")
plt.title("Cross-Entropy Loss (2-class)")
plt.tight_layout()
plt.savefig("images/ch07/NN07_loss_surfaces.png", dpi=150)
