# Functions visualization
import numpy as np, matplotlib.pyplot as plt

x = np.linspace(-3, 3, 200)
funcs = {
    "Linear y=x": (x,),
    "Quadratic y=x^2": (x**2,),
    "Exponential y=e^x": (np.exp(x),),
    "Log y=ln(x+1)": (np.log(x+1),),
}
fig, axes = plt.subplots(2,2,figsize=(10,8))
for ax, (n, (y,)) in zip(axes.flat, funcs.items()):
    ax.plot(x, y, "b-", lw=2)
    ax.axhline(0, color="gray", ls=":", lw=.5)
    ax.axvline(0, color="gray", ls=":", lw=.5)
    ax.set_title(n)
plt.tight_layout()
plt.savefig("images/ch02/NN02_function_family.png", dpi=150)

