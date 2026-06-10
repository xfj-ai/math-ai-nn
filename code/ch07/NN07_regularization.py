import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Overfitting demo
torch.manual_seed(42)
N = 30
X = torch.randn(N, 1)
y = 2*X + 1 + torch.randn(N, 1)*0.5

big_model = nn.Sequential(
    nn.Linear(1, 128), nn.ReLU(),
    nn.Linear(128, 128), nn.ReLU(),
    nn.Linear(128, 1),
)

# Train without regularization
opt = torch.optim.SGD(big_model.parameters(), lr=0.01)
losses = []
for _ in range(2000):
    pred = big_model(X)
    loss = nn.MSELoss()(pred, y)
    opt.zero_grad(); loss.backward(); opt.step()
    losses.append(loss.item())

plt.plot(losses)
plt.yscale("log")
plt.xlabel("Iteration"); plt.ylabel("MSE Loss")
plt.title("Overfitting Demo (big model, small data)")
plt.savefig("images/ch07/NN07_overfitting_demo.png", dpi=150)
print(f"Final loss (no reg): {losses[-1]:.4f}")

# L2 regularization
model_l2 = nn.Sequential(
    nn.Linear(1, 128), nn.ReLU(),
    nn.Linear(128, 128), nn.ReLU(),
    nn.Linear(128, 1),
)
opt_l2 = torch.optim.SGD(model_l2.parameters(), lr=0.01, weight_decay=0.01)
losses_l2 = []
for _ in range(2000):
    pred = model_l2(X)
    loss = nn.MSELoss()(pred, y)
    opt_l2.zero_grad(); loss.backward(); opt_l2.step()
    losses_l2.append(loss.item())

plt.figure()
plt.plot(losses, label="No reg")
plt.plot(losses_l2, label="L2 reg")
plt.yscale("log"); plt.legend()
plt.savefig("images/ch07/NN07_l2_regularization.png", dpi=150)
print(f"Final loss (L2): {losses_l2[-1]:.4f}")
