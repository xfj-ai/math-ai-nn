# nn.Module / Sequential / Parameter
import torch
import torch.nn as nn

# Method 1: nn.Module subclass
class MyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)
    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        return self.fc2(x)

model = MyNet()
print("Module subclass:")
print(model)

# Method 2: nn.Sequential
model2 = nn.Sequential(
    nn.Linear(784, 128),
    nn.Sigmoid(),
    nn.Linear(128, 10),
)
print("\nSequential:")
print(model2)

# Parameters
for name, param in model.named_parameters():
    print(f"  {name}: {param.shape}, grad={param.requires_grad}")

# Custom parameter
class CustomLayer(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W = nn.Parameter(torch.randn(d_in, d_out) * 0.1)
        self.b = nn.Parameter(torch.zeros(d_out))
    def forward(self, x):
        return x @ self.W + self.b

layer = CustomLayer(784, 128)
print(f"\nCustom layer params: {sum(p.numel() for p in layer.parameters())}")

