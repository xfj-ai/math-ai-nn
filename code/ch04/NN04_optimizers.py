# Momentum vs AdaGrad vs RMSprop
import numpy as np, matplotlib.pyplot as plt

def f(x): return x[0]**2 + 5*x[1]**2  # ill-conditioned

def grad(x):
    return np.array([2*x[0], 10*x[1]])

x0 = np.array([1.5, 1.0])
lr = 0.1; epochs = 50

# GD
x = x0.copy()
loss_gd = [f(x)]
for _ in range(epochs):
    x -= lr * grad(x)
    loss_gd.append(f(x))

# Momentum
x = x0.copy(); v = np.zeros(2); beta = 0.9
loss_mom = [f(x)]
for _ in range(epochs):
    v = beta*v + lr*grad(x)
    x -= v
    loss_mom.append(f(x))

# AdaGrad
x = x0.copy(); G = np.zeros(2); eps = 1e-8
loss_ada = [f(x)]
for _ in range(epochs):
    g = grad(x)
    G += g**2
    x -= lr * g / (np.sqrt(G) + eps)
    loss_ada.append(f(x))

# RMSprop
x = x0.copy(); s = np.zeros(2); beta2 = 0.9
loss_rms = [f(x)]
for _ in range(epochs):
    g = grad(x)
    s = beta2*s + (1-beta2)*g**2
    x -= lr * g / (np.sqrt(s) + eps)
    loss_rms.append(f(x))

for name, loss in [("GD", loss_gd), ("Momentum", loss_mom), ("AdaGrad", loss_ada), ("RMSprop", loss_rms)]:
    plt.plot(loss, label=name)
plt.yscale("log"); plt.legend(); plt.title("Optimizer Comparison")
plt.xlabel("Step"); plt.ylabel("Loss")
plt.savefig("images/ch04/NN04_optimizer_comparison.png", dpi=150)

