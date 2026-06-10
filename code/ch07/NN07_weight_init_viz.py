# Weight Initialization Methods Comparison
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

np.random.seed(42)
n_layers = 20
n_neurons = 200

def simulate_init(method, n_layers=20, n_neurons=200):
    """Simulate forward pass gradients with different init methods"""
    x = np.random.randn(n_neurons)
    grad_norms = []
    for _ in range(n_layers):
        if method == 'zeros':
            W = np.zeros((n_neurons, n_neurons))
        elif method == 'large':
            W = np.random.randn(n_neurons, n_neurons) * 2.0
        elif method == 'xavier':
            W = np.random.randn(n_neurons, n_neurons) * np.sqrt(2.0 / n_neurons)
        elif method == 'he':
            W = np.random.randn(n_neurons, n_neurons) * np.sqrt(1.0 / n_neurons)
        x = np.tanh(W @ x)
        grad_norms.append(np.linalg.norm(x))
    return grad_norms

methods = ['zeros', 'large', 'xavier', 'he']
colors = ['red', 'orange', 'green', 'blue']
labels = ["Zeros (vanishing)", "Large (exploding)", "Xavier (balanced)", "He (ReLU)"]

fig, ax = plt.subplots(figsize=(10, 6))

for method, color, label in zip(methods, colors, labels):
    norms = simulate_init(method)
    ax.plot(range(1, n_layers+1), norms, color=color, linewidth=2, label=label, marker='o', markersize=3)

ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('Layer')
ax.set_ylabel('Gradient Norm')
ax.set_title('Weight Initialization Effect on Gradient Flow')
ax.set_yscale('log')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('images/ch07/NN07_weight_init_compare.png', dpi=150, bbox_inches='tight')
print("✅ Weight initialization visualization saved")
