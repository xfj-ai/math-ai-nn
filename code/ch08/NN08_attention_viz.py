# Attention Weights Visualization
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Simulate attention weights for a sentence
np.random.seed(42)
tokens = ['The', 'cat', 'sat', 'on', 'the', 'mat', '.']
n = len(tokens)

# Create realistic attention pattern
attention = np.random.rand(n, n) * 0.3
# Each token attends most to itself and adjacent tokens
for i in range(n):
    attention[i, max(0,i-1):min(n,i+2)] += 0.5
    attention[i, i] += 0.3
# Special patterns: subject attends to predicate, etc.
attention[1, 2] += 0.4  # cat -> sat
attention[4, 5] += 0.3  # the -> mat
# Normalize
attention = attention / attention.sum(axis=1, keepdims=True)

fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(attention, cmap='YlOrRd', vmin=0, vmax=0.6)

# Labels
ax.set_xticks(range(n))
ax.set_yticks(range(n))
ax.set_xticklabels(tokens)
ax.set_yticklabels(tokens)
ax.set_xlabel('Key (attended to)')
ax.set_ylabel('Query (attending from)')
ax.set_title('Self-Attention Weights Heatmap')

# Add text annotations
for i in range(n):
    for j in range(n):
        val = attention[i, j]
        if val > 0.25:
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=8, color='white')
        elif val > 0.1:
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=7, color='black')

plt.colorbar(im, ax=ax, fraction=0.046)
plt.tight_layout()
plt.savefig('images/ch08/NN08_attention_visualization.png', dpi=150, bbox_inches='tight')
print("✅ Attention visualization saved")
