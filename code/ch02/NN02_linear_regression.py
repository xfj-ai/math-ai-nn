# Linear regression
import numpy as np, matplotlib.pyplot as plt

# Generate data: y = 2x + 1 + noise
np.random.seed(42)
X = np.linspace(0, 10, 50)
y_true = 2*X + 1
y = y_true + np.random.randn(50)*2

# Analytical solution: w = (X^T X)^-1 X^T y
X_mat = np.c_[np.ones_like(X), X]
w_analytical = np.linalg.inv(X_mat.T @ X_mat) @ X_mat.T @ y
print(f"Analytical: y = {w_analytical[1]:.4f}x + {w_analytical[0]:.4f}")

# GD solution
w_gd = np.array([0.0, 0.0])
lr = 0.01
for _ in range(1000):
    pred = X_mat @ w_gd
    grad = (2/len(X)) * X_mat.T @ (pred - y)
    w_gd -= lr * grad
print(f"GD: y = {w_gd[1]:.4f}x + {w_gd[0]:.4f}")

plt.scatter(X, y, alpha=0.5, label="data")
plt.plot(X, X_mat @ w_analytical, "r-", lw=2, label="analytical")
plt.plot(X, X_mat @ w_gd, "g--", lw=2, label="GD")
plt.legend()
plt.savefig("images/ch02/NN02_linear_regression.png", dpi=150)

