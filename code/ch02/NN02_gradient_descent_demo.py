# Gradient descent demo
import numpy as np, matplotlib.pyplot as plt

# 1D: f(x) = x^2
def f1(x): return x**2
def g1(x): return 2*x

def gd_1d(start, lr, steps=30):
    x = start; path = [x]
    for _ in range(steps):
        x -= lr * g1(x); path.append(x)
    return np.array(path)

# Compare learning rates
plt.figure(figsize=(12, 4))
for i, lr in enumerate([0.01, 0.1, 0.5]):
    path = gd_1d(2.0, lr)
    plt.subplot(1, 3, i+1)
    xr = np.linspace(-2, 2, 100)
    plt.plot(xr, f1(xr))
    plt.scatter(path, f1(path), c=range(len(path)), cmap="viridis", s=20)
    plt.title(f"lr={lr}, steps={len(path)-1}")
plt.tight_layout()
plt.savefig("images/ch02/NN02_learning_rate_compare.png", dpi=150)

# 2D: f(x,y) = x^2 + 2y^2
def gd_2d(start, lr=0.1, steps=50):
    x, y = start
    path = [(x, y)]
    for _ in range(steps):
        x -= lr * 2*x  # df/dx
        y -= lr * 4*y  # df/dy
        path.append((x, y))
    return np.array(path)

path = gd_2d((2.0, 1.5))
print(f"2D GD: started at (2,1.5), ended at {path[-1].round(4)}")
plt.savefig("images/ch02/NN02_gd_2d_path.png", dpi=150)

