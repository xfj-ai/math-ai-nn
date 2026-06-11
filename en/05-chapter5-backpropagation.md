# Chapter 5: Backpropagation — The Heart of Neural Networks ⭐

> **Goal**: **Thoroughly understand** the backpropagation algorithm — how errors propagate backward through the network to compute gradients efficiently.

> **Code**: `../code/ch05/` (6 files)

---

## 📋 Chapter Learning Objectives

- [ ] Understand why gradient computation is challenging in multi-layer networks
- [ ] Master the concept of **neuron error** $\delta$
- [ ] Understand the **delta recurrence**: $\delta^{(l)} = \delta^{(l+1)} W^{(l+1)} \odot \sigma'(z^{(l)})$
- [ ] Be able to manually trace backpropagation on a simple network
- [ ] Implement backpropagation from scratch with NumPy
- [ ] Understand how PyTorch's autograd implements backpropagation

---

## 5-1 Gradient Descent Review and Multi-Layer Challenges

### The Challenge

For a single neuron, the gradient is straightforward. But for deep networks:

$$
L = L(y), \quad y = \sigma(z^{(L)}), \quad z^{(L)} = W^{(L)}a^{(L-1)} + b^{(L)}, \quad \dots
$$

The loss depends on **every** weight through a **chain** of operations.

### The Naive Approach Would Be Insanely Slow

If we computed $\partial L / \partial w_{ji}^{(l)}$ directly for each weight:

- For each weight $\partial L / \partial w_{ji}^{(l)}$, we'd need to re-compute the chain
- With $10^6+$ parameters, this is **completely impractical**

### The Breakthrough: Backpropagation

Backpropagation computes **all gradients in one backward pass** — roughly as expensive as a forward pass.

> **Core Insight**: Backpropagation = efficient application of the chain rule. It reuses intermediate computations rather than re-computing for each parameter.

---

## 5-2 Neuron Error $\delta$: The Core Concept ⭐

### Definition

The **neuron error** $\delta_j^{(l)}$ is the gradient of the loss with respect to the pre-activation $z_j^{(l)}$:

$$
\delta_j^{(l)} = \frac{\partial L}{\partial z_j^{(l)}}
$$

### Why Define $\delta$?

The gradient of the loss with respect to weight $w_{ji}^{(l)}$ is:

$$
\frac{\partial L}{\partial w_{ji}^{(l)}} = \frac{\partial L}{\partial z_j^{(l)}} \cdot \frac{\partial z_j^{(l)}}{\partial w_{ji}^{(l)}} = \delta_j^{(l)} \cdot a_i^{(l-1)}
$$

**If we know $\delta$ for each neuron, we can compute all parameter gradients easily!**

### Delta Recurrence (The Secret Sauce)

The magic of backpropagation: $\delta$ at one layer can be computed from $\delta$ at the next layer!

$$
\delta^{(l)} = (W^{(l+1)})^{\mathsf{T}} \delta^{(l+1)} \odot \sigma'(z^{(l)})
$$

Where $\odot$ is element-wise multiplication.

```text
Backward signal flow:
δ^(L) (output) → δ^(L-1) → δ^(L-2) → ... → δ^(1) (first hidden layer)
```

---

## 5-3 Mathematical Derivation of Backpropagation ⭐

### Step 1: Output Layer Error $\delta^{(L)}$

For MSE loss $L = \frac{1}{2}\|y - t\|^2$ with Sigmoid output:

$$
\delta^{(L)} = (y - t) \odot \sigma'(z^{(L)})
$$

### Step 2: Hidden Layer Error (Delta Recurrence)

$$
\delta^{(l)} = ((W^{(l+1)})^{\mathsf{T}} \delta^{(l+1)}) \odot \sigma'(z^{(l)})
$$

### Step 3: Parameter Gradients

$$
\frac{\partial L}{\partial W^{(l)}} = \delta^{(l)} (a^{(l-1)})^{\mathsf{T}}
$$

$$
\frac{\partial L}{\partial b^{(l)}} = \delta^{(l)}
$$

### The Complete Algorithm

```text
Forward pass:
  a⁰ = x
  z¹ = W¹a⁰ + b¹,  a¹ = σ(z¹)
  z² = W²a¹ + b²,  a² = σ(z²)
  ...
  zᴸ = Wᴸaᴸ⁻¹ + bᴸ,  y = σ(zᴸ)
  L = ½||y - t||²

Backward pass:
  δᴸ = (y - t) ⊙ σ'(zᴸ)
  δᴸ⁻¹ = (Wᴸ)ᵀδᴸ ⊙ σ'(zᴸ⁻¹)
  δᴸ⁻² = (Wᴸ⁻¹)ᵀδᴸ⁻¹ ⊙ σ'(zᴸ⁻²)
  ...
  δ¹ = (W²)ᵀδ² ⊙ σ'(z¹)

Parameter gradients:
  ∂L/∂Wˡ = δˡ(aˡ⁻¹)ᵀ
  ∂L/∂bˡ = δˡ
```

---

## 5-4 Computational Graph and Backpropagation

### Forward Graph (Building)

```text
x ─→ W¹x+b¹ ─→ σ ─→ a¹ ─→ W²a¹+b² ─→ σ ─→ a² ─→ ½||a²-t||² ─→ L
```

### Backward Graph (Gradient Flow)

```text
L ─→ δ² = (a²-t) ⊙ σ'(z²)
↓
δ¹ = (W²)ᵀδ² ⊙ σ'(z¹)
↓
∂L/∂W² = δ²(a¹)ᵀ,  ∂L/∂b² = δ²
∂L/∂W¹ = δ¹(x)ᵀ,   ∂L/∂b¹ = δ¹
```

---

## 5-5 Manual Backpropagation in Python ⭐

### Full Implementation from Scratch

```python
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

class NeuralNetwork:
    """2-layer neural network with manual backprop"""

    def __init__(self, input_size, hidden_size, output_size):
        self.W1 = np.random.randn(hidden_size, input_size) * 0.1
        self.b1 = np.zeros((hidden_size, 1))
        self.W2 = np.random.randn(output_size, hidden_size) * 0.1
        self.b2 = np.zeros((output_size, 1))

    def forward(self, x):
        """Forward pass (stores intermediates for backward)"""
        self.x = x.reshape(-1, 1)

        self.z1 = self.W1 @ self.x + self.b1
        self.a1 = sigmoid(self.z1)

        self.z2 = self.W2 @ self.a1 + self.b2
        self.a2 = sigmoid(self.z2)

        return self.a2

    def backward(self, target, lr=0.1):
        """Backward pass: compute gradients and update parameters"""
        target = target.reshape(-1, 1)

        # Output layer error: δ² = (a² - t) ⊙ σ'(z²)
        delta2 = (self.a2 - target) * sigmoid_derivative(self.z2)

        # Hidden layer error: δ¹ = (W²)ᵀδ² ⊙ σ'(z¹)
        delta1 = (self.W2.T @ delta2) * sigmoid_derivative(self.z1)

        # Parameter gradients
        grad_W2 = delta2 @ self.a1.T
        grad_b2 = delta2
        grad_W1 = delta1 @ self.x.T
        grad_b1 = delta1

        # Parameter updates (gradient descent)
        self.W2 -= lr * grad_W2
        self.b2 -= lr * grad_b2
        self.W1 -= lr * grad_W1
        self.b1 -= lr * grad_b1

    def train(self, X, y, epochs=1000, lr=0.1, verbose=True):
        """Full training loop"""
        for epoch in range(epochs):
            total_loss = 0
            for i in range(len(X)):
                output = self.forward(X[i])
                loss = 0.5 * np.sum((output - y[i])**2)
                total_loss += loss
                self.backward(y[i], lr)

            if verbose and epoch % 200 == 0:
                print(f"Epoch {epoch}, Loss: {total_loss/len(X):.6f}")

    def predict(self, X):
        """Predict class labels"""
        predictions = []
        for x in X:
            output = self.forward(x)
            predictions.append(np.argmax(output))
        return np.array(predictions)

# Test on a simple problem
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[1, 0], [0, 1], [0, 1], [1, 0]])  # XOR (one-hot)

nn = NeuralNetwork(2, 4, 2)
nn.train(X, y, epochs=2000, lr=0.5)

predictions = nn.predict(X)
print(f"\nPredictions: {predictions}")
print(f"Expected:    {np.argmax(y, axis=1)}")
print(f"Accuracy: {np.mean(predictions == np.argmax(y, axis=1)) * 100:.1f}%")
```

```output
Epoch 0, Loss: 0.256478
Epoch 200, Loss: 0.125638
Epoch 400, Loss: 0.024562
Epoch 600, Loss: 0.008923
Epoch 800, Loss: 0.004561
Epoch 1000, Loss: 0.002891
Epoch 1200, Loss: 0.002012
Epoch 1400, Loss: 0.001523
Epoch 1600, Loss: 0.001234
Epoch 1800, Loss: 0.001012

Predictions: [0 1 1 0]
Expected:    [0 1 1 0]
Accuracy: 100.0%
```

> **Core Insight**: Backpropagation is just the chain rule, applied efficiently. The key insight: errors at each layer can be computed **recursively** from errors at the next layer — this is the $\delta$ recurrence.

---

## 5-6 PyTorch Autograd Deep Dive

### Comparing Manual vs. Autograd

```python
import torch

# Manual backpropagation (from our NumPy implementation)
# vs. PyTorch autograd

# PyTorch version
class TorchNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.W1 = torch.nn.Parameter(torch.randn(hidden_size, input_size) * 0.1)
        self.b1 = torch.nn.Parameter(torch.zeros(hidden_size, 1))
        self.W2 = torch.nn.Parameter(torch.randn(output_size, hidden_size) * 0.1)
        self.b2 = torch.nn.Parameter(torch.zeros(output_size, 1))

    def forward(self, x):
        z1 = self.W1 @ x + self.b1
        a1 = torch.sigmoid(z1)
        z2 = self.W2 @ a1 + self.b2
        a2 = torch.sigmoid(z2)
        return a2

    def train_step(self, x, target, lr=0.1):
        x = torch.tensor(x, dtype=torch.float32).reshape(-1, 1)
        target = torch.tensor(target, dtype=torch.float32).reshape(-1, 1)

        output = self.forward(x)
        loss = 0.5 * torch.sum((output - target)**2)

        # Autograd handles everything!
        loss.backward()

        # Manually update (no optimizer)
        with torch.no_grad():
            self.W1 -= lr * self.W1.grad
            self.b1 -= lr * self.b1.grad
            self.W2 -= lr * self.W2.grad
            self.b2 -= lr * self.b2.grad

            # Zero gradients
            self.W1.grad.zero_()
            self.b1.grad.zero_()
            self.W2.grad.zero_()
            self.b2.grad.zero_()

        return loss.item()

# Same experiment, but with autograd
torch_nn = TorchNN(2, 4, 2)
for epoch in range(1000):
    total_loss = 0
    for i in range(len(X)):
        loss = torch_nn.train_step(X[i], y[i], lr=0.5)
        total_loss += loss

    if epoch % 200 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss/len(X):.6f}")
```

---

## 5-7 Building a Complete Neural Network with PyTorch

### Clean PyTorch Implementation

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class TwoLayerNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Sigmoid(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)

# Training setup
model = TwoLayerNet(2, 4, 2)
criterion = nn.MSELoss()  # for simplicity; CrossEntropyLoss is better for classification
optimizer = optim.SGD(model.parameters(), lr=0.5)

# Data
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)
dataset = TensorDataset(X_tensor, y_tensor)
loader = DataLoader(dataset, batch_size=1, shuffle=True)

# Training
for epoch in range(1000):
    total_loss = 0
    for batch_x, batch_y in loader:
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if epoch % 200 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss/len(loader):.6f}")
```

---

## 5-8 Experiment: Training an MNIST Classifier from Scratch

### Full MNIST Training

```python
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Data
transform = transforms.ToTensor()
train_data = datasets.MNIST(root='./data', train=True,
                           transform=transform, download=True)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

test_data = datasets.MNIST(root='./data', train=False,
                          transform=transform, download=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

# Model
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 10),
)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training
for epoch in range(5):
    model.train()
    for images, labels in train_loader:
        images = images.view(images.size(0), -1)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Evaluation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.view(images.size(0), -1)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Epoch {epoch}: Test accuracy = {accuracy:.2f}%")
```

---

## 5-9 Gradient Flow Analysis: Why Deep Networks Are Hard to Train

### Vanishing Gradients

In deep networks, gradients can **vanish** as they propagate backward:

$$
\delta^{(1)} \approx (W^{(2)})^{\mathsf{T}} \delta^{(2)} \odot \sigma'(z^{(1)})
$$

If $\sigma'(z)$ is small (Sigmoid/Tanh saturation), the gradient shrinks with each layer.

### Gradient Norm Across Layers

```python
# Simulate gradient magnitude across layers
import numpy as np

def simulate_gradient_flow(depth=10, init_scale=1.0):
    grad_norm = init_scale
    norms = [grad_norm]
    for l in range(depth):
        # Simulate: δ^{(l)} = (W)ᵀδ^{(l+1)} ⊙ σ'(z)
        # Effect: gradient typically shrinks or grows
        grad_norm *= np.random.uniform(0.5, 1.5)
        norms.append(grad_norm)
    return norms

norms = simulate_gradient_flow(20)
print(f"First layer gradient norm: {norms[0]:.4f}")
print(f"Last layer gradient norm:  {norms[-1]:.4f}")
print(f"Ratio: {norms[-1]/norms[0]:.4f}")
# If ratio << 1: vanishing gradients
# If ratio >> 1: exploding gradients
```

### Solutions

| Problem | Solution |
|:--------|:---------|
| Vanishing gradients | ReLU, ResNet skip connections |
| Exploding gradients | Gradient clipping, weight normalization |
| Unstable gradients | Batch normalization, careful initialization |

---

## 5-10 Practical Backpropagation Tips

### Debugging with Gradient Checks

```python
def gradient_check(model, x, y, epsilon=1e-5):
    """Numerical gradient check"""
    param = model.W1
    grad_numerical = np.zeros_like(param)

    for i in range(param.shape[0]):
        for j in range(param.shape[1]):
            param[i, j] += epsilon
            loss_plus = model.forward(x); loss_plus = 0.5*np.sum((model.a2 - y)**2)

            param[i, j] -= 2*epsilon
            loss_minus = model.forward(x); loss_minus = 0.5*np.sum((model.a2 - y)**2)

            param[i, j] += epsilon  # restore
            grad_numerical[i, j] = (loss_plus - loss_minus) / (2*epsilon)

    # Compare with analytical gradient
    print(f"Max diff: {np.max(np.abs(grad_numerical - model.grad_W1)):.2e}")
```

### Common Mistakes

| Mistake | Symptom | Fix |
|:--------|:--------|:----|
| Forgot `.zero_grad()` | Gradients explode | Call `optimizer.zero_grad()` before each step |
| Wrong loss function | Poor convergence | Use CrossEntropyLoss for classification |
| No activation on output | Unbounded predictions | Add appropriate output activation |
| Learning rate too high | NaN loss | Reduce learning rate |

---

## 📖 Chapter Summary

### The Backpropagation Algorithm

```text
Forward: x → z¹ → a¹ → z² → a² → ... → zᴸ → y → L
                         ↓
Backward: δᴸ → δᴸ⁻¹ → ... → δ¹
              ↓          ↓
           ∂L/∂Wᴸ     ∂L/∂W¹
```

### Key Equations

| Step | Equation |
|:-----|:---------|
| Output error | $\delta^{(L)} = (y - t) \odot \sigma'(z^{(L)})$ |
| Delta recurrence | $\delta^{(l)} = (W^{(l+1)})^{\mathsf{T}} \delta^{(l+1)} \odot \sigma'(z^{(l)})$ |
| Weight gradient | $\frac{\partial L}{\partial W^{(l)}} = \delta^{(l)} (a^{(l-1)})^{\mathsf{T}}$ |
| Bias gradient | $\frac{\partial L}{\partial b^{(l)}} = \delta^{(l)}$ |

### 🧪 Exercises

#### Exercise 1: Trace Backprop Manually
Given a 2-2-1 network with specific weights, manually trace one forward and backward pass.

#### Exercise 2: Implement Backprop from Scratch
Modify the manual implementation in this chapter to work with Tanh activation.

#### Exercise 3: Gradient Check
Add numerical gradient checking to your implementation and verify correctness.

#### Exercise 4: Compare Activation Functions
Train the same network with Sigmoid, Tanh, and ReLU. Compare convergence speed.
