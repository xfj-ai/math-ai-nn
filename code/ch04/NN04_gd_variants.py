# GD vs SGD vs Mini-batch
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
N = 200
X = np.random.randn(N, 1) * 2
y = 2.5 * X.squeeze() + 1.0 + np.random.randn(N) * 0.5

def mse(w, b):
    y_pred = w * X.squeeze() + b
    return np.mean((y - y_pred)**2)

def grad(w, b, batch_indices=None):
    if batch_indices is None:
        xb, yb = X.squeeze(), y
    else:
        xb, yb = X.squeeze()[batch_indices], y[batch_indices]
    y_pred = w * xb + b
    dw = -2 * np.mean(xb * (yb - y_pred))
    db = -2 * np.mean(yb - y_pred)
    return dw, db

# Compare
w_gd, b_gd = 0.0, 0.0
w_sgd, b_sgd = 0.0, 0.0
w_mb, b_mb = 0.0, 0.0
lr = 0.01; epochs = 100
loss_gd, loss_sgd, loss_mb = [], [], []

for ep in range(epochs):
    # GD
    dw, db = grad(w_gd, b_gd)
    w_gd -= lr*dw; b_gd -= lr*db
    loss_gd.append(mse(w_gd, b_gd))
    # SGD
    idx = np.random.randint(N)
    dw, db = grad(w_sgd, b_sgd, [idx])
    w_sgd -= lr*dw; b_sgd -= lr*db
    loss_sgd.append(mse(w_sgd, b_sgd))
    # Mini-batch (32)
    idxs = np.random.choice(N, 32)
    dw, db = grad(w_mb, b_mb, idxs)
    w_mb -= lr*dw; b_mb -= lr*db
    loss_mb.append(mse(w_mb, b_mb))

plt.plot(loss_gd, label="GD (full)")
plt.plot(loss_sgd, alpha=0.5, label="SGD")
plt.plot(loss_mb, label="Mini-batch 32")
plt.legend(); plt.xlabel("Epoch"); plt.ylabel("MSE"); plt.yscale("log")
plt.title("GD vs SGD vs Mini-batch Convergence")
plt.savefig("images/ch04/NN04_gd_sgd_minibatch.png", dpi=150)
print(f"Final loss - GD: {loss_gd[-1]:.4f}, SGD: {loss_sgd[-1]:.4f}, MB: {loss_mb[-1]:.4f}")

