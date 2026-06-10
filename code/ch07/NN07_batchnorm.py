import torch
import torch.nn as nn

# BatchNorm behavior
bn = nn.BatchNorm1d(4, affine=False, momentum=0.1)
x = torch.tensor([[1.0,2,3,4],[5,6,7,8],[9,10,11,12]])

with torch.no_grad():
    out = bn(x)
print(f"Input mean: {x.mean(dim=0)}")
print(f"Output mean: {torch.round(out.mean(dim=0), decimals=4)} (should be ~0)")
print(f"Output std: {torch.round(out.std(dim=0), decimals=4)} (should be ~1)")

# Train vs Eval
model = nn.Sequential(
    nn.Linear(10, 20),
    nn.BatchNorm1d(20),
    nn.ReLU(),
    nn.Linear(20, 1),
)
model.train()
x = torch.randn(16, 10)
_ = model(x)  # running stats update

model.eval()
out = model(x)
print(f"\nEval output (with running stats): {out.mean().item():.4f}")
