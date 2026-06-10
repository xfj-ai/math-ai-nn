import numpy as np, matplotlib.pyplot as plt
from scipy import signal

x = np.array([1, 2, 3, 4, 5])
k = np.array([0.5, 1.0, 0.5])
conv_1d = np.convolve(x, k, mode="valid")
print(f"1D input: {x}")
print(f"1D kernel: {k}")
print(f"1D conv (valid): {conv_1d}")

img = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]])
kernel = np.array([[1,0],[-1,0]])
conv_2d = signal.convolve2d(img, kernel, mode="valid")
print(f"2D conv:\n{conv_2d}")

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(img, cmap="gray"); axes[0].set_title("Input")
axes[1].imshow(kernel, cmap="gray"); axes[1].set_title("Kernel")
axes[2].imshow(conv_2d, cmap="gray"); axes[2].set_title("Output")
plt.tight_layout()
plt.savefig("images/ch06/NN06_convolution_demo.png", dpi=150)
