# GD warmup demo
import numpy as np
import matplotlib.pyplot as plt

def f(x): return x**2 + 2*x + 1
def grad(x): return 2*x + 2

def gd(start, lr=0.1, steps=20):
    x = start
    path = [x]
    for _ in range(steps):
        x = x - lr * grad(x)
        path.append(x)
    return np.array(path)

path = gd(1.0)
xr = np.linspace(-3, 1, 100)

plt.figure(figsize=(10, 4))
plt.subplot(121)
plt.plot(xr, f(xr), "b-")
plt.scatter(path, f(path), c=range(len(path)), cmap="viridis")
plt.axvline(-1, color="gray", ls=":")
plt.title("GD Path")

plt.subplot(122)
plt.plot(range(len(path)), path, "bo-")
plt.axhline(-1, color="r", ls="--")
plt.title("Convergence")
plt.tight_layout()
plt.savefig("images/ch01/NN01_gradient_descent_intuition.png", dpi=150)

