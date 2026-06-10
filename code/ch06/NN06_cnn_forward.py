import torch
import torch.nn as nn

conv = nn.Conv2d(1, 4, kernel_size=3, padding=1)
x = torch.randn(1, 1, 28, 28)
out = conv(x)
print(f"Input: {x.shape} -> Conv2d(1,4,3x3,padding=1) -> Output: {out.shape}")

pool = nn.MaxPool2d(2)
out2 = pool(out)
print(f"After MaxPool(2x2): {out2.shape}")

cnn = nn.Sequential(
    nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
    nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear(64*7*7, 128), nn.ReLU(),
    nn.Linear(128, 10),
)
total_params = sum(p.numel() for p in cnn.parameters())
print(f"CNN: {total_params:,} params")
print(f"Forward: {x.shape} -> {cnn(x).shape}")
