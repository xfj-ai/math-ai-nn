# PyTorch optimizers comparison
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

torch.manual_seed(42)
N = 500
X = torch.randn(N, 1)
y = 2*X + 1 + torch.randn(N, 1)*0.3

def make_model():
    return nn.Sequential(nn.Linear(1, 1))

def train(model, optimizer, epochs=200):
    losses = []
    for _ in range(epochs):
        pred = model(X)
        loss = nn.MSELoss()(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses

models_optimizers = {
    "SGD lr=0.01": (make_model(), optim.SGD(make_model().parameters(), lr=0.01)),
    "Momentum": (make_model(), optim.SGD(make_model().parameters(), lr=0.01, momentum=0.9)),
    "Adam": (make_model(), optim.Adam(make_model().parameters(), lr=0.01)),
}

for name, (model, opt) in models_optimizers.items():
    losses = train(model, opt)
    plt.plot(losses, label=name)
plt.yscale("log"); plt.legend(); plt.xlabel("Epoch"); plt.ylabel("MSE")
plt.title("PyTorch Optimizer Comparison")
plt.savefig("images/ch04/NN04_pytorch_optimizers.png", dpi=150)
print("PyTorch optimizers comparison done")

