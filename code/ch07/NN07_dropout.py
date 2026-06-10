import torch
import torch.nn as nn

# Dropout behavior
dropout = nn.Dropout(p=0.5)
x = torch.ones(1, 10)
y = dropout(x)
print(f"Input: {x}")
print(f"Output: {y}")
print(f"Avg after dropout: {y.mean().item():.2f} (should be ~1.0)")
print(f"Non-zero: {(y!=0).sum().item()}/10 (should be ~5)")

# Dropout in training vs eval mode
model = nn.Sequential(
    nn.Linear(10, 20),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(20, 1),
)

model.train()
x = torch.randn(4, 10)
y_train = model(x)
print(f"\nTrain mode output: {y_train}")

model.eval()
y_eval = model(x)
print(f"Eval mode output: {y_eval}")
