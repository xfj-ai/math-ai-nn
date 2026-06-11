# Chapter 3: PyTorch Basics — Tensor & Autograd

> **Goal**: Understand the two core tools PyTorch provides — **Tensor** (efficient array computation) and **Autograd** (automatic differentiation) — and how they work together to enable neural network training.

> **Code**: `../code/ch03/` (8 files)

---

## 📋 Chapter Learning Objectives

- [ ] Understand the relationship between Tensor and NumPy array
- [ ] Master Tensor creation and basic operations
- [ ] Understand broadcasting mechanism
- [ ] Understand Autograd: how PyTorch automatically computes gradients
- [ ] Master computational graph concepts
- [ ] Be able to build network modules with nn.Module
- [ ] Understand DataLoader for batch processing

---

## 3-1 Why Do We Need PyTorch?

### 3-1-1 From Excel to NumPy to PyTorch

| Tool | Pros | Cons |
|:-----|:-----|:------|
| **Excel** | Visual, accessible | Can't scale, no GPU |
| **NumPy** | Powerful, great for prototyping | No GPU, no autograd |
| **PyTorch** | GPU + autograd + ecosystem | Learning curve |

NumPy is great for fixed computations, but neural networks need **differentiation** — and PyTorch's autograd handles this automatically.

### 3-1-2 Deep Learning Framework Comparison

| Framework | Pros | Cons |
|:----------|:-----|:------|
| **PyTorch** | Dynamic graphs, Pythonic, research-friendly | Production deployment maturing |
| **TensorFlow/Keras** | Deployment, production | Steeper learning curve for custom work |
| **JAX** | Functional, fast | Newer, smaller ecosystem |

### 3-1-3 PyTorch's Design Philosophy

1. **Pythonic**: Feels like writing NumPy
2. **Dynamic graphs**: Graphs built on-the-fly (ease debugging)
3. **Tensors first**: Same interface for CPU/GPU

---

## 3-2 PyTorch Tensor Basics

### 3-2-1 What Is a Tensor?

A **tensor** is a multi-dimensional array — a generalization of:

```text
Scalar (0D) → Vector (1D) → Matrix (2D) → Tensor (3D+)
```

In neural networks: images are 4D tensors (batch × channels × height × width).

### 3-2-2 Creating Tensors

```python
import torch

# From data
t1 = torch.tensor([1, 2, 3])
t2 = torch.tensor([[1, 2], [3, 4]])

# From shapes (useful for weights)
zeros = torch.zeros(2, 3)
ones = torch.ones(2, 3)
randn = torch.randn(2, 3)  # standard normal

# From NumPy
import numpy as np
np_array = np.array([1, 2, 3])
t3 = torch.from_numpy(np_array)

print(f"tensor: {t1}")
print(f"zeros: {zeros}")
print(f"random: {randn}")
```

### 3-2-3 Tensor Properties

```python
x = torch.randn(3, 4)
print(f"Shape: {x.shape}")
print(f"Dtype: {x.dtype}")
print(f"Device: {x.device}")
print(f"Numel: {x.numel()}")  # total elements
```

---

## 3-3 Tensor Operations and Broadcasting

### 3-3-1 Basic Operations

```python
a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])

print(f"Add: {a + b}")
print(f"Mul: {a * b}")     # element-wise
print(f"Dot: {a @ b}")     # dot product (torch.dot(a, b))
print(f"Matmul: {a @ b}")  # same for 1D; use @ for 2D+
```

### 3-3-2 Broadcasting ⭐

Broadcasting automatically expands dimensions to make shapes compatible:

```python
a = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])  # shape (2, 3)
b = torch.tensor([10, 20, 30])  # shape (3,) → broadcast to (2, 3)

c = a + b  # b is "stretched" along dimension 0
print(c)
# tensor([[11, 22, 33],
#         [14, 25, 36]])
```

### 3-3-3 Dimension Operations

```python
x = torch.randn(2, 3, 4)
print(f"Original: {x.shape}")

# Reshape
print(f"Reshaped: {x.view(6, 4).shape}")  # or x.reshape(6, 4)

# Transpose
print(f"Transposed: {x.transpose(0, 1).shape}")  # (3, 2, 4)

# Unsqueeze / Squeeze
print(f"Unsqueezed: {x.unsqueeze(0).shape}")  # (1, 2, 3, 4)
print(f"Squeezed:   {x.squeeze().shape}")     # removes dims of size 1
```

---

## 3-4 Autograd: Automatic Differentiation ⭐

### 3-4-1 What Is Autograd?

**Autograd** is PyTorch's automatic differentiation engine. It automatically computes **gradients of any computation** you define.

### 3-4-2 requires_grad: Tell PyTorch Which Tensors Need Gradients

```python
# Only tensors with requires_grad=True will accumulate gradients
w = torch.tensor([2.0], requires_grad=True)
b = torch.tensor([1.0], requires_grad=True)
x = torch.tensor([3.0])  # input (no gradient needed)
```

### 3-4-3 Forward Pass: Building the Computation Graph

```python
# Forward: PyTorch records every operation in a graph
z = w * x + b
loss = z ** 2

print(f"z = {z.item():.2f}")
print(f"loss = {loss.item():.2f}")
```

### 3-4-4 Backward Pass: One Line for All Gradients

```python
# Backward: computes ALL gradients via chain rule
loss.backward()

print(f"d(loss)/dw = {w.grad.item():.4f}")
print(f"d(loss)/db = {b.grad.item():.4f}")

# Verify manually:
# loss = (wx + b)²
# d(loss)/dw = 2(wx + b) * x = 2(2*3+1) * 3 = 42
# d(loss)/db = 2(wx + b) * 1 = 2(2*3+1) = 14
print(f"Expected: dL/dw = {2*(2*3+1)*3:.4f}, dL/db = {2*(2*3+1):.4f}")
```

### 3-4-5 How backward() Works

`loss.backward()` traces the computational graph backward:

```text
loss ──→ grad = 1
  ↑
z² ──→ d(loss)/dz = 2z
  ↑
z = wx + b ──→ dz/dw = x,  dz/db = 1
  ↑
[w, b, x] (leaf tensors)
```

### 3-4-6 Gradient Accumulation ⚠️

Gradients **accumulate** by default — you must zero them before each backward pass:

```python
# WRONG: gradients accumulate!
loss.backward()
loss.backward()  # gradients DOUBLE!
print(w.grad)  # 2x expected value!

# CORRECT: zero the gradients first
w.grad.zero_()
loss.backward()  # now correct
```

---

## 3-5 Computational Graph Deep Dive ⭐

### 3-5-1 grad_fn: Tracing Operation Sources

Every tensor tracks how it was created via `grad_fn`:

```python
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2
z = y.mean()

print(f"x.grad_fn = {x.grad_fn}")  # None (leaf tensor)
print(f"y.grad_fn = {y.grad_fn}")  # <PowBackward0>
print(f"z.grad_fn = {z.grad_fn}")  # <MeanBackward0>
```

### 3-5-2 Static vs. Dynamic Graphs

| Aspect | Static (TF1) | Dynamic (PyTorch) |
|:-------|:-------------|:------------------|
| Graph built | Before execution | During execution |
| Debugging | Hard | Easy |
| Flexibility | Limited | Full Python control |

### 3-5-3 Detaching from the Graph

Sometimes you want to **stop gradient flow** through part of the graph:

```python
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2
z = y.detach()  # z is a normal tensor, no gradient tracking
w = z ** 2      # w won't track gradients through y
```

---

## 3-6 nn.Module: Building Blocks for Networks

### 3-6-1 The nn.Module Base Class

All neural network components inherit from `nn.Module`:

```python
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

net = SimpleNet()
print(net)
```

### 3-6-2 nn.Sequential: Quick Stacking

```python
net = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10),
)
print(net)
```

### 3-6-3 nn.Parameter: Trainable Parameters

```python
# nn.Linear already registers its weights as parameters
for name, param in net.named_parameters():
    print(f"{name}: {param.shape}, requires_grad={param.requires_grad}")
```

### 3-6-4 Model Parameter Management

```python
# Total parameter count
total = sum(p.numel() for p in net.parameters())
print(f"Total parameters: {total:,}")

# Training mode vs eval mode
net.train()  # enables dropout, batchnorm updates
net.eval()   # disables dropout, fixes batchnorm
```

---

## 3-7 Datasets and DataLoaders

### 3-7-1 Dataset: Where the Data Lives

```python
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
```

### 3-7-2 DataLoader: Batching and Shuffling

```python
dataset = MyDataset(torch.randn(1000, 784),
                    torch.randint(0, 10, (1000,)))
loader = DataLoader(dataset, batch_size=32, shuffle=True)

for batch_x, batch_y in loader:
    print(f"Batch: x={batch_x.shape}, y={batch_y.shape}")
    break
```

### 3-7-3 MNIST Dataset Example

```python
from torchvision import datasets, transforms

mnist = datasets.MNIST(
    root='./data',
    train=True,
    transform=transforms.ToTensor(),
    download=True
)

loader = DataLoader(mnist, batch_size=64, shuffle=True)
images, labels = next(iter(loader))
print(f"Images: {images.shape}")  # (64, 1, 28, 28)
print(f"Labels: {labels.shape}")  # (64,)
```

---

## 3-8 Complete Training Loop

### 3-8-1 The Training Loop Template

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Model
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10),
)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training loop
for epoch in range(10):
    for batch_x, batch_y in loader:
        # 1. Flatten images
        batch_x = batch_x.view(batch_x.size(0), -1)

        # 2. Forward
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)

        # 3. Backward
        optimizer.zero_grad()  # clear old gradients
        loss.backward()        # compute new gradients

        # 4. Update
        optimizer.step()       # apply gradients

    print(f"Epoch {epoch}: loss = {loss.item():.4f}")
```

### 3-8-2 GPU Acceleration

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using: {device}")

model.to(device)  # move model to GPU

for batch_x, batch_y in loader:
    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
    # ... forward, backward, update (same as before)
```

---

## 📖 Chapter Summary

### Core Concepts

```text
Tensor ──→ Multi-dimensional array (like NumPy, but GPU-ready)
Autograd ──→ Automatic gradient computation
Computational Graph ──→ Records operations for backward pass
nn.Module ──→ Building block for network components
DataLoader ──→ Efficient batch processing
```

### 🧪 Exercises

#### Exercise 1: Tensor Practice
Create a 3×4 tensor, compute its mean, std, reshape to 2×6.

#### Exercise 2: Autograd
Define $y = x^2 + 3x + 1$, compute $dy/dx$ at $x=2$ using autograd.

#### Exercise 3: Custom Module
Create a 3-layer network with nn.Module, count its parameters.

#### Exercise 4: Full Training
Train a model on MNIST (use the template above), achieve >90% accuracy on the test set.
