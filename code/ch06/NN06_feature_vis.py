import torch
import torch.nn as nn
import matplotlib.pyplot as plt

model = nn.Sequential(
    nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
    nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
)

x = torch.randn(1, 3, 32, 32)
features = []
h = x
for name, layer in model.named_children():
    h = layer(h)
    if isinstance(layer, nn.ReLU):
        features.append(h.detach())

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
for row, feat in enumerate(features):
    for col in range(min(4, feat.shape[1])):
        axes[row][col].imshow(feat[0, col].numpy(), cmap="viridis")
        axes[row][col].set_title(f"Layer{row+1} feat{col}")
plt.tight_layout()
plt.savefig("images/ch06/NN06_feature_maps.png", dpi=150)
