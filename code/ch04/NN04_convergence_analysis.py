# Convergence analysis
import numpy as np, matplotlib.pyplot as plt

# Quadratic: f(x) = 0.5*x^T A x, condition number kappa = lambda_max/lambda_min
# GD step size: optimal = 2/(lambda_max + lambda_min)
kappa = 10
lambda_max, lambda_min = kappa, 1.0
lr_opt = 2/(lambda_max + lambda_min)
lr_large = 1.95/lambda_max
lr_small = 0.5/lambda_max

x0 = 1.0
x_opt, x_large, x_small = x0, x0, x0
for _ in range(100):
    x_opt -= lr_opt * lambda_max * x_opt
    x_large -= lr_large * lambda_max * x_large
    x_small -= lr_small * lambda_max * x_small

print(f"Optimal lr=1/(L+mu): final x={x_opt:.6f}")
print(f"Large lr={lr_large:.4f}: final x={x_large:.6f}")
print(f"Small lr={lr_small:.4f}: final x={x_small:.6f}")

# Condition number visualization
lambdas = np.linspace(0.1, 10, 50)
cond_nums = [10, 50, 100]
plt.figure(figsize=(10, 4))
for cn in cond_nums:
    rates = [1 - 2*l/(cn*1.0+1.0)*l for l in lambdas]
    plt.plot(lambdas, rates, label=f"kappa={cn}")
plt.legend(); plt.xlabel("lambda"); plt.ylabel("convergence rate")
plt.title("Effect of Condition Number")
plt.savefig("images/ch04/NN04_convergence_rate.png", dpi=150)

