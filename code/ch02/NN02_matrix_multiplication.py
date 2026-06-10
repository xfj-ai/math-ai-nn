# Matrix Multiplication Visualization
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

A = np.array([[1, 2, 3], [4, 5, 6]])
B = np.array([[7, 8], [9, 10], [11, 12]])
C = A @ B

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))

# Matrix A
im1 = ax1.imshow(A, cmap='Blues', aspect='auto')
ax1.set_title('A (2×3)')
ax1.set_xticks(range(3))
ax1.set_yticks(range(2))
for i in range(2):
    for j in range(3):
        ax1.text(j, i, str(A[i, j]), ha='center', va='center', fontsize=12)
plt.colorbar(im1, ax=ax1, fraction=0.046)

# Matrix B
im2 = ax2.imshow(B, cmap='Oranges', aspect='auto')
ax2.set_title('B (3×2)')
ax2.set_xticks(range(2))
ax2.set_yticks(range(3))
for i in range(3):
    for j in range(2):
        ax2.text(j, i, str(B[i, j]), ha='center', va='center', fontsize=12)
plt.colorbar(im2, ax=ax2, fraction=0.046)

# Result C
im3 = ax3.imshow(C, cmap='Greens', aspect='auto')
ax3.set_title('C = A × B (2×2)')
ax3.set_xticks(range(2))
ax3.set_yticks(range(2))
for i in range(2):
    for j in range(2):
        ax3.text(j, i, str(C[i, j]), ha='center', va='center', fontsize=14, fontweight='bold')
plt.colorbar(im3, ax=ax3, fraction=0.046)

plt.tight_layout()
plt.savefig('images/ch02/NN02_matrix_multiplication.png', dpi=150, bbox_inches='tight')
print(f"✅ Matrix multiplication visualization saved")
