# Chapter 4: Optimization in Neural Networks

> **Goal**: Understand how neural networks learn through optimization — from loss functions to gradient descent variants, with hands-on PyTorch experiments.

> **Code**: `../code/ch04/` (7 files)

---

## 📋 Chapter Learning Objectives

- [ ] Understand the formal formulation of neural network optimization
- [ ] Master the concept of loss functions (MSE, Cross-Entropy)
- [ ] Understand Softmax and its relationship to probabilities
- [ ] Master SGD, Momentum, Adam optimizers
- [ ] Understand learning rate scheduling
- [ ] Be able to train a complete classifier in PyTorch

---

## 4-1 Parameters and Variables of Neural Networks

### Parameter Categories

| Type | Examples | Characteristics |
|:-----|:---------|:----------------|
| **Trainable parameters** | $W, b$ | Learned from data via gradient descent |
| **Hyperparameters** | Learning rate, layer count | Set manually, tuned via search |
| **Input data** | $x$ (images, text) | Given, never changed |
| **Target labels** | $t$ (ground truth) | Given for supervised learning |

### Formal Definition

$$
\theta = \{W^{(1)}, b^{(1)}, W^{(2)}, b^{(2)}, \dots, W^{(L)}, b^{(L)}\}
$$

The goal of learning: find $\theta^*$ that minimizes the loss.

---

## 4-2 Variable Relationships in Neural Networks

### Computational Graph

```text
x → [W¹, b¹] → z¹ → σ → a¹ → [W², b²] → z² → σ → a² → L(a², t)
```

Where each variable's value depends on the previous ones.

### Forward Dependency Chain

- $a^{(l)}$ depends on: $a^{(l-1)}$, $W^{(l)}$, $b^{(l)}$
- $L$ depends on: $a^{(L)}$ (output) and $t$ (target)
- All parameters ultimately affect $L$

---

## 4-3 Training Data and Ground Truth

### Dataset Structure

| Dataset | Purpose | Size |
|:--------|:--------|:-----|
| **Training** | Learn parameters | ~60% |
| **Validation** | Tune hyperparameters | ~20% |
| **Test** | Final evaluation | ~20% |

### MNIST as Running Example

```python
from torchvision import datasets, transforms

mnist_train = datasets.MNIST(root='./data', train=True,
                             transform=transforms.ToTensor(), download=True)
mnist_test = datasets.MNIST(root='./data', train=False,
                            transform=transforms.ToTensor(), download=True)

print(f"Training samples: {len(mnist_train)}")
print(f"Test samples: {len(mnist_test)}")
print(f"Image shape: {mnist_train[0][0].shape}")
```

---

## 4-4 Loss (Cost) Functions

### 4-4-1 Mean Squared Error (MSE)

$$
L_{\text{MSE}} = \frac{1}{2m} \sum_{i=1}^{m} \|y_i - t_i\|^2
$$

Best for: **regression** tasks (continuous outputs).

### 4-4-2 Cross-Entropy Loss

$$
L_{\text{CE}} = -\sum_{k} t_k \log p_k
$$

Best for: **classification** tasks.

### 4-4-3 Softmax Function

Converts raw scores (logits) to probabilities:

$$
p_k = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}
$$

Properties:
- All $p_k \in (0, 1)$
- $\sum_{k} p_k = 1$
- Preserves ranking: larger $z_k$ → larger $p_k$

### 4-4-4 Why Cross-Entropy + Softmax?

The combination of Softmax + Cross-Entropy gives a particularly nice gradient:

$$
\frac{\partial L}{\partial z_k} = p_k - t_k
$$

This means: the gradient is simply the difference between prediction and target!

---

## 4-5 Experiencing Neural Networks with Python & PyTorch

### Manual Loss Computation

```python
import torch
import torch.nn.functional as F

# Simulated outputs and targets
logits = torch.tensor([[2.0, 0.5, 0.1]])  # raw scores
targets = torch.tensor([0])  # true class = class 0

# Softmax
probs = F.softmax(logits, dim=1)
print(f"Probabilities: {probs}")

# Cross-entropy loss
loss = F.cross_entropy(logits, targets)
print(f"Loss: {loss.item():.4f}")
```

### Gradient of Softmax + Cross-Entropy

```python
# Manually verify the nice gradient
probs = F.softmax(logits, dim=1)
grad = probs.clone()
grad[0, targets[0]] -= 1  # p_k - t_k
print(f"Gradient w.r.t. logits: {grad}")
```

---

## 4-6 Experiment: Training Your First Classifier

### Complete Training Pipeline

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 1. Data
transform = transforms.ToTensor()
train_data = datasets.MNIST(root='./data', train=True,
                           transform=transform, download=True)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

# 2. Model
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 10),
)

# 3. Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. Training loop
for epoch in range(5):
    total_loss = 0
    for images, labels in train_loader:
        images = images.view(images.size(0), -1)  # flatten

        # Forward
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch}: avg loss = {avg_loss:.4f}")

print("Training complete!")
```

---

## ⚠️ Common Optimization Issues

| Issue | Symptom | Solution |
|:------|:--------|:---------|
| Learning rate too high | Loss diverges (NaN) | Reduce lr by 10x |
| Learning rate too low | Loss decreases too slowly | Increase lr |
| No normalization | Slow convergence | Normalize inputs to [0,1] |
| Bad initialization | Vanishing/exploding gradients | Use xavier/kaiming init |

---

## 4-7 Optimizer Comparison ⭐

### SGD (Stochastic Gradient Descent)

$$
\theta_{t+1} = \theta_t - \eta \nabla L(\theta_t)
$$

Simple, but can be slow to converge.

### SGD with Momentum

$$
v_{t+1} = \beta v_t + \nabla L(\theta_t)
$$
$$
\theta_{t+1} = \theta_t - \eta v_{t+1}
$$

Like a ball rolling downhill — builds speed in consistent directions.

### Adam (Adaptive Moment Estimation)

Combines momentum + adaptive learning rates:

- **Momentum**: tracks direction
- **RMSprop**: adapts per-parameter learning rate
- **Bias correction**: handles initialization

```python
# PyTorch optimizer comparison
optimizers = {
    'sgd': optim.SGD(model.parameters(), lr=0.01),
    'momentum': optim.SGD(model.parameters(), lr=0.01, momentum=0.9),
    'adam': optim.Adam(model.parameters(), lr=0.001),
}

# Usage in training loop
for name, opt in optimizers.items():
    print(f"Optimizer: {name}")
    # ... standard training loop with opt.step()
```

---

## 4-8 Learning Rate Scheduling

### Common Schedulers

```python
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

# Fixed step decay
scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
# Reduce LR every 10 epochs by factor 0.1

# Adaptive reduction
scheduler = ReduceLROnPlateau(optimizer, mode='min',
                              factor=0.5, patience=3)
# Reduce LR by half when loss plateaus for 3 epochs
```

### Learning Rate Best Practices

| Strategy | When to Use |
|:---------|:------------|
| **Constant** | Simple problems, short training |
| **Step decay** | Well-understood tasks, transfer learning |
| **Cosine annealing** | Modern deep learning, CV tasks |
| **Reduce on plateau** | When you don't know the right schedule |

---

## 4-9 Loss Landscape Visualization

The loss landscape visualizes how loss changes with parameters (simplified to 2D):

```python
import numpy as np
import matplotlib.pyplot as plt

# Simulate loss landscape (simplified)
x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)

# Loss function: bowl with some ravines
Z = X**2 + Y**2 + 0.5 * X * Y

plt.figure(figsize=(8, 6))
plt.contourf(X, Y, Z, levels=50, cmap='viridis')
plt.colorbar(label='Loss')
plt.xlabel('Parameter 1')
plt.ylabel('Parameter 2')
plt.title('Loss Landscape')
plt.show()
```

---

## 📖 Chapter Summary

### Core Concepts

```text
Loss Functions → Measure prediction error
Softmax → Convert logits to probabilities
Gradient Descent → Follow negative gradient
Optimizers (SGD/Momentum/Adam) → Smart way to follow gradient
Learning Rate Schedule → Adaptive step sizes
```

### 🧪 Exercises

#### Exercise 1: Implement MSE and Cross-Entropy
Write both loss functions from scratch in NumPy. Compare their behavior on the same predictions.

#### Exercise 2: Manual Softmax
Compute Softmax of `[2.0, 1.0, 0.1]` manually, then verify with `torch.softmax()`.

#### Exercise 3: Optimizer Comparison
Train the same model with SGD, SGD+momentum, and Adam. Compare convergence speed.

#### Exercise 4: Learning Rate Search
Try learning rates `[0.1, 0.01, 0.001, 0.0001]` on MNIST. Which converges fastest?
