# Taylor approximation
import numpy as np, matplotlib.pyplot as plt

def f(x): return np.exp(x)  # e^x around x=0

def taylor(x, n):
    # e^x = sum_{k=0}^{n} x^k/k!
    result = np.zeros_like(x)
    term = np.ones_like(x)
    for k in range(n+1):
        result += term
        term *= x / (k+1)
    return result

x = np.linspace(-2, 2, 100)
plt.figure(figsize=(10, 6))
plt.plot(x, f(x), "k-", lw=2, label="e^x exact")
for n in [0, 1, 2, 3, 5]:
    plt.plot(x, taylor(x, n), "--", lw=1.5, label=f"n={n}")
plt.ylim(-1, 8); plt.legend(); plt.grid()
plt.title("Taylor Expansion of e^x")
plt.savefig("images/ch02/NN02_taylor_approx.png", dpi=150)

