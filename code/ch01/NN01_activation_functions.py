# Activation functions plot
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-5, 5, 200)
def sig(x): return 1/(1+np.exp(-x))
def relu(x): return np.maximum(0, x)

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
funcs = {
    "Step": (np.where(x>=0, 1, 0), None),
    "Sigmoid": (sig(x), sig(x)*(1-sig(x))),
    "ReLU": (relu(x), np.where(x>0, 1, 0)),
    "Tanh": (np.tanh(x), 1-np.tanh(x)**2),
}
for ax, (name, (y, dy)) in zip(axes.flat, funcs.items()):
    ax.plot(x, y, "b-", lw=2, label=name)
    if dy is not None:
        ax.plot(x, dy, "r--", lw=1.5, label="deriv")
    ax.axhline(0, color="gray", ls=":", lw=.5)
    ax.legend()
    ax.set_title(name)
plt.tight_layout()
plt.savefig("images/ch01/NN01_activation_functions.png", dpi=150)

