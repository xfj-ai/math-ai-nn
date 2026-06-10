# Loss landscape
import numpy as np, matplotlib.pyplot as plt

def f(x, y): return x**2 + 5*y**2

x = np.linspace(-2, 2, 100)
y = np.linspace(-2, 2, 100)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

plt.figure(figsize=(12, 4))
plt.subplot(131); plt.contour(X, Y, Z, levels=20, cmap="viridis")
plt.title("Contour"); plt.xlabel("x"); plt.ylabel("y")

plt.subplot(132); plt.contourf(X, Y, np.log(Z), levels=20, cmap="viridis")
plt.colorbar(); plt.title("Log Contour"); plt.xlabel("x"); plt.ylabel("y")

from mpl_toolkits.mplot3d import Axes3D
ax = plt.subplot(133, projection="3d")
ax.plot_surface(X, Y, np.log(Z), cmap="viridis", alpha=0.8)
ax.set_title("3D Log Surface"); ax.set_xlabel("x"); ax.set_ylabel("y")

plt.tight_layout()
plt.savefig("images/ch04/NN04_loss_landscape.png", dpi=150)

