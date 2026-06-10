# Numerical derivative
import numpy as np, matplotlib.pyplot as plt

def f(x): return x**3 + 2*x
def df_analytic(x): return 3*x**2 + 2

def df_numerical(f, x, h=1e-5):
    return (f(x+h) - f(x-h))/(2*h)

x = 1.0
print(f"Analytic: {df_analytic(x):.6f}")
print(f"Numerical: {df_numerical(f, x):.6f}")
print(f"Diff: {abs(df_analytic(x)-df_numerical(f,x)):.2e}")

# Test different h values
hs = [10**-i for i in range(1, 10)]
errors = [abs(df_analytic(1)-df_numerical(f,1,h)) for h in hs]
plt.loglog(hs, errors, "bo-")
plt.xlabel("h"); plt.ylabel("Error"); plt.grid()
plt.savefig("images/ch02/NN02_derivative_visual.png", dpi=150)

