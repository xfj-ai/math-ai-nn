# Vectors and matrices
import numpy as np

# Vector dot product
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
dot = np.dot(a, b)
print(f"Dot product: {dot}")

# Matrix multiplication
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
C = A @ B
print(f"Matmul:\n{C}")

# Norms
print(f"L2 norm: {np.linalg.norm(a):.4f}")
print(f"Cosine sim: {dot/(np.linalg.norm(a)*np.linalg.norm(b)):.4f}")

# Broadcasting
X = np.array([[1,2,3],[4,5,6]])
b = np.array([10, 20, 30])
print(f"With broadcast:\n{X + b}")

