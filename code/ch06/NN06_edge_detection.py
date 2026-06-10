import numpy as np, matplotlib.pyplot as plt
from scipy import signal
from skimage import data

camera = data.camera().astype(float)[::2, ::2]

kernels = {
    "Sobel X": np.array([[-1,0,1],[-2,0,2],[-1,0,1]]),
    "Sobel Y": np.array([[-1,-2,-1],[0,0,0],[1,2,1]]),
    "Gaussian": np.array([[1,2,1],[2,4,2],[1,2,1]])/16,
    "Laplacian": np.array([[0,-1,0],[-1,4,-1],[0,-1,0]]),
}

fig, axes = plt.subplots(1, 5, figsize=(20, 4))
axes[0].imshow(camera, cmap="gray"); axes[0].set_title("Original")
for ax, (name, k) in zip(axes[1:], kernels.items()):
    out = signal.convolve2d(camera, k, mode="same")
    ax.imshow(out, cmap="gray"); ax.set_title(name)
plt.tight_layout()
plt.savefig("images/ch06/NN06_edge_detection.png", dpi=150)
print("Edge detection visualization saved")
