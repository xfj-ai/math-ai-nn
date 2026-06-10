# Scaling Law Visualization
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Simulated scaling law data
np.random.seed(42)
params = np.logspace(6, 11, 50)  # 1M to 100B params
data = np.logspace(7, 12, 50)    # 10M to 1T tokens
compute = np.logspace(15, 22, 50)  # FLOPs

# Loss follows power law: L ≈ c * x^(-α)
alpha_p = 0.076  # params scaling exponent
alpha_d = 0.095  # data scaling exponent
alpha_c = 0.050  # compute scaling exponent

loss_p = 2.5 * params ** (-alpha_p) + np.random.randn(50) * 0.01
loss_d = 2.5 * data ** (-alpha_d) + np.random.randn(50) * 0.01
loss_c = 2.5 * compute ** (-alpha_c) + np.random.randn(50) * 0.01

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

axes[0].loglog(params, loss_p, 'b-', linewidth=2)
axes[0].scatter(params, loss_p, c='blue', s=10, alpha=0.6)
axes[0].set_xlabel('Parameters')
axes[0].set_ylabel('Test Loss')
axes[0].set_title('Scaling with Parameters')
axes[0].grid(True, alpha=0.3)

axes[1].loglog(data, loss_d, 'g-', linewidth=2)
axes[1].scatter(data, loss_d, c='green', s=10, alpha=0.6)
axes[1].set_xlabel('Training Data Tokens')
axes[1].set_ylabel('Test Loss')
axes[1].set_title('Scaling with Data')
axes[1].grid(True, alpha=0.3)

axes[2].loglog(compute, loss_c, 'r-', linewidth=2)
axes[2].scatter(compute, loss_c, c='red', s=10, alpha=0.6)
axes[2].set_xlabel('Compute (FLOPs)')
axes[2].set_ylabel('Test Loss')
axes[2].set_title('Scaling with Compute')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('images/ch09/NN09_scaling_law.png', dpi=150, bbox_inches='tight')
print("✅ Scaling law visualization saved")
