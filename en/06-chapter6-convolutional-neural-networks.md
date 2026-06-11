# Chapter 6: Deep Learning & Convolutional Neural Networks

> **Goal**: Understand why standard fully connected networks fail for images, and how CNNs solve the problem through **spatial structure** and **weight sharing**.

> **Code**: `../code/ch06/` (7 files)

> **Figures**: `../images/ch06/`

---

## 📋 Chapter Learning Objectives

- [ ] Understand why fully connected networks are inefficient for images
- [ ] Master convolution operation: kernel, stride, padding
- [ ] Understand pooling and its purpose
- [ ] Be able to build a CNN in PyTorch
- [ ] Understand CNN backpropagation
- [ ] Know classic CNN architectures

---

## 6-1 Why Do We Need CNNs?

### The Problem with Fully Connected Layers for Images

An image of 224×224×3 has **150,528** input values. A fully connected layer with 1024 neurons would have:

$$
150,528 \times 1024 = 154,140,672 \text{ parameters}
$$

That's **154 million parameters** for just the first layer!

### Three Key Ideas of CNNs

| Idea | Benefit |
|:-----|:--------|
| **Local connectivity** | Each neuron only sees a small patch of the input |
| **Weight sharing** | Same filter applied everywhere → drastically fewer parameters |
| **Hierarchical features** | Low-level edges → mid-level shapes → high-level objects |

### Parameter Comparison

| Layer Type | Parameters (224×224×3 → 64 filters) |
|:-----------|:-------------------------------------|
| Fully connected | ~150M |
| Convolutional (3×3) | ~1,700 |

That's **~100,000x fewer parameters**!

---

## 6-2 Understanding CNN with the "Little Genius" Analogy

Imagine the "little geniuses" in the hidden layer each specialize in detecting a specific pattern:

```text
Layer 1: Edge detectors
  ├── Genius A: detects horizontal edges 
  ├── Genius B: detects vertical edges
  ├── Genius C: detects 45° edges
  └── Genius D: detects corners

Layer 2: Texture detectors
  ├── Genius E: detects stripes
  ├── Genius F: detects checkerboard
  └── Genius G: detects polka dots

Layer 3: Object part detectors
  ├── Genius H: detects eyes
  ├── Genius I: detects wheels
  └── Genius J: detects windows
```

Each "filter" is a small weight matrix that slides across the image looking for its pattern.

---

## 6-3 Convolution Layer Mathematics

### The Convolution Operation

A convolution slides a **filter (kernel)** across the input:

```text
Input (5×5)        Filter (3×3)        Output (3×3)
[x₁₁ x₁₂ x₁₃ x₁₄ x₁₅]   [w₁₁ w₁₂ w₁₃]     [y₁₁ y₁₂ y₁₃]
[x₂₁ x₂₂ x₂₃ x₂₄ x₂₅]   [w₂₁ w₂₂ w₂₃]  =  [y₂₁ y₂₂ y₂₃]
[x₃₁ x₃₂ x₃₃ x₃₄ x₃₅]   [w₃₁ w₃₂ w₃₃]     [y₃₁ y₃₂ y₃₃]
[x₄₁ x₄₂ x₄₃ x₄₄ x₄₅]
[x₅₁ x₅₂ x₅₃ x₅₄ x₅₅]
```

### Mathematical Formula

$$
y_{ij} = \sum_{p=1}^{k} \sum_{q=1}^{k} x_{i+p, j+q} \cdot w_{pq} + b
$$

### Output Size Formula

$$
\text{Output size} = \frac{\text{Input size} - \text{Kernel size} + 2 \times \text{Padding}}{\text{Stride}} + 1
$$

### Padding and Stride

| | Without Padding | With Padding (same) |
|:---|:---------------|:--------------------|
| **Stride=1** | Output shrinks | Output same as input |
| **Stride=2** | Output shrinks faster | Downsampling |

### Python Demonstration

```python
import torch
import torch.nn as nn

# Simple convolution
conv = nn.Conv2d(in_channels=1, out_channels=1,
                 kernel_size=3, stride=1, padding=0)

x = torch.randn(1, 1, 5, 5)  # (batch, channels, height, width)
y = conv(x)
print(f"Input shape:  {x.shape}")
print(f"Output shape: {y.shape}")

# With padding to preserve size
conv_same = nn.Conv2d(1, 1, 3, stride=1, padding=1)
y_same = conv_same(x)
print(f"Output (same): {y_same.shape}")
```

---

## 6-4 Pooling and Fully Connected Layers

### Max Pooling

Max pooling selects the maximum value in each window:

```text
Input (4×4):
[1, 3, 2, 4]
[5, 6, 7, 8]
[9, 10, 11, 12]
[13, 14, 15, 16]

Max pooling (2×2, stride=2):
[6, 8]
[14, 16]
```

### Why Pooling?

| Benefit | Explanation |
|:--------|:------------|
| **Dimensionality reduction** | Reduces spatial size, fewer params |
| **Translation invariance** | Small shifts don't change max |
| **Receptive field growth** | Each later pixel "sees" more of the input |

### The Full CNN Pipeline

```text
Input → [Conv → ReLU → Pool] × N → Flatten → FC → Softmax → Output
```

---

## 6-5 Experiencing CNN with Python

### Edge Detection with Convolution

```python
import numpy as np
from scipy import signal

# Simple image
image = np.array([
    [10, 10, 10, 10, 10],
    [10, 10, 10, 10, 10],
    [10, 10, 10, 10, 10],
    [10, 10, 10, 10, 10],
    [10, 10, 10, 10, 10],
])
image[2, :] = 200  # horizontal line in middle

# Horizontal edge filter (Sobel)
filter_h = np.array([
    [-1, -1, -1],
    [0,  0,  0],
    [1,  1,  1],
])

edges = signal.convolve2d(image, filter_h, mode='valid')
print(f"Edges:\n{edges}")
# The horizontal line will be strongly detected!
```

---

## 6-6 CNN Backpropagation

Backpropagation in CNNs follows the same principle as fully connected networks:

- **Convolution layer backward**: $\delta^{(l)}$ is computed by **full convolution** of $\delta^{(l+1)}$ with the flipped kernel
- **Pooling layer backward**: unpooling — gradient is routed back to the max location

### Gradient Flow in a CNN

```text
Forward:  x → Conv → ReLU → Pool → FC → Loss
                                            ↓
Backward: dL/dx ← dL/dz_conv ← dL/dz_relu ← dL/dz_pool ← dL/dz_fc ← dL/dL
```

---

## 6-7 PyTorch CNN Implementation

### Complete CNN for MNIST

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # Conv block 1
        x = self.pool(torch.relu(self.conv1(x)))   # (1,28,28)→(32,14,14)
        # Conv block 2
        x = self.pool(torch.relu(self.conv2(x)))   # (32,14,14)→(64,7,7)
        # Flatten
        x = x.view(-1, 64 * 7 * 7)
        # Classifier
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Training setup
model = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_data = datasets.MNIST('./data', train=True,
                           transform=transform, download=True)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

# Training
for epoch in range(3):
    model.train()
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch}: loss = {loss.item():.4f}")

# Evaluate
model.eval()
correct = 0
test_data = datasets.MNIST('./data', train=False,
                          transform=transform, download=True)
test_loader = DataLoader(test_data, batch_size=64)
with torch.no_grad():
    for batch_x, batch_y in test_loader:
        outputs = model(batch_x)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == batch_y).sum().item()
print(f"Test accuracy: {100 * correct / len(test_data):.2f}%")
```

---

## 6-8 CNN Visualization Techniques

### Feature Map Visualization

```python
import matplotlib.pyplot as plt

# Get feature maps from first conv layer
activation = {}
def hook_fn(name):
    def hook(module, input, output):
        activation[name] = output.detach()
    return hook

model.conv1.register_forward_hook(hook_fn('conv1'))

# Forward pass
sample, _ = test_data[0]
_ = model(sample.unsqueeze(0))

# Visualize first 8 feature maps
fig, axes = plt.subplots(2, 4, figsize=(12, 6))
for i, ax in enumerate(axes.flat):
    if i < activation['conv1'].shape[1]:
        ax.imshow(activation['conv1'][0, i], cmap='gray')
        ax.axis('off')
        ax.set_title(f'Filter {i}')
plt.tight_layout()
plt.show()
```

---

## 6-9 Classic CNN Architectures

| Architecture | Year | Key Innovation | Parameters |
|:-------------|:-----|:---------------|:-----------|
| **LeNet-5** | 1998 | First CNN | 60K |
| **AlexNet** | 2012 | Deep CNN, ReLU, Dropout | 60M |
| **VGGNet** | 2014 | Very deep, small filters | 138M |
| **GoogLeNet** | 2014 | Inception modules | 5M |
| **ResNet** | 2015 | Skip connections → 152 layers | 25M |

---

## 6-10 Receptive Fields and Dilated Convolutions

### Receptive Field

The **receptive field** is the region of the input that influences a particular neuron. Stacking convolutions increases the receptive field.

### Dilated Convolution

Dilated convolution inserts gaps in the kernel to increase receptive field without adding parameters:

```python
# Standard 3×3 convolution
conv_std = nn.Conv2d(1, 1, 3, dilation=1)

# Dilated 3×3 convolution (receptive field = 5×5)
conv_dilated = nn.Conv2d(1, 1, 3, dilation=2)
```

---

## 6-11 Data Augmentation

### Why Augment?

Limited data → overfitting. Augmentation creates **new training samples** through transformations.

```python
from torchvision import transforms

train_transform = transforms.Compose([
    transforms.RandomRotation(10),        # rotate ±10°
    transforms.RandomAffine(0, shear=10), # shear
    transforms.RandomHorizontalFlip(),     # flip
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
```

### Common Augmentations

| Augmentation | Description | When to Use |
|:-------------|:------------|:------------|
| Rotation | Rotate by small angles | Objects have no "up" |
| Flip | Horizontal/vertical mirror | Symmetric objects |
| Crop | Randomly crop and resize | Partially occluded objects |
| Color jitter | Shift brightness, contrast | Varying lighting conditions |

---

## 📖 Chapter Summary

### CNN Core Concepts

```text
Convolution → Local feature detection with weight sharing
Pooling → Dimensionality reduction + translation invariance
Hierarchical features → Edges → Textures → Objects
```

### CNN vs. Fully Connected

| Aspect | Fully Connected | CNN |
|:-------|:----------------|:----|
| Parameters | ~150M (first layer) | ~1,700 (first layer) |
| Spatial info | Lost (flattened) | Preserved |
| Translation invariance | Learned | Built-in (pooling) |

### 🧪 Exercises

#### Exercise 1: Manual Convolution
Given a 4×4 input and a 2×2 kernel, manually compute the output with stride 1 and 2.

#### Exercise 2: Build a CNN
Train a CNN on CIFAR-10 (32×32×3 color images). Try to achieve >70% accuracy.

#### Exercise 3: Visualize Filters
Visualize the learned filters from the first conv layer. What patterns do they detect?

#### Exercise 4: Ablation Study
Remove the pooling layers from your CNN. How does accuracy change? Parameter count?
