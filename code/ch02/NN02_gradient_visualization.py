# 3D gradient visualization
import numpy as np, matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def f(x, y): return x**2 + 2*y**2

x = np.linspace(-2, 2, 40)
y = np.linspace(-2, 2, 40)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(X, Y, Z, cmap="viridis", alpha=0.8)
ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("f(x,y)")
ax.set_title("f(x,y) = x^2 + 2y^2")
plt.savefig("images/ch02/NN02_gradient_3d.png", dpi=150)

