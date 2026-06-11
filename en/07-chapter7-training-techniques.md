# Chapter 7: Training Techniques — Optimizers, Regularization & Loss Functions

> **Goal**: Master the advanced techniques that make neural network training stable, fast, and generalizable.

> **Code**: `../code/ch07/` (7 files)

---

## 📋 Chapter Learning Objectives

- [ ] Understand optimizer variants (SGD → Adam → AdamW)
- [ ] Master learning rate scheduling
- [ ] Understand regularization methods (L1/L2, dropout, early stopping)
- [ ] Understand normalization (BatchNorm, LayerNorm)
- [ ] Master weight initialization strategies
- [ ] Understand loss functions for different tasks
- [ ] Deep understanding of Softmax

---

## 7-1 Optimizer Landscape: From SGD to AdamW

### Optimizer Evolution

```text
SGD → SGD+Momentum → Nesterov → AdaGrad → RMSprop → Adam → AdamW
```

### SGD with Momentum

$$
v_{t+1} = \beta v_t + \nabla L(\theta_t)
$$
$$
\theta_{t+1} = \theta_t - \eta v_{t+1}
$$

### Adam (Adaptive Moment Estimation)

Adam combines momentum with per-parameter adaptive learning rates:

```python
import torch.optim as optim

optimizers = {
    'sgd': optim.SGD(model.parameters(), lr=0.01),
    'momentum': optim.SGD(model.parameters(), lr=0.01, momentum=0.9),
    'adam': optim.Adam(model.parameters(), lr=0.001),
    'adamw': optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01),
}
```

### Which Optimizer to Choose?

| Scenario | Recommended Optimizer |
|:---------|:---------------------|
| Quick prototyping | Adam |
| CV (ImageNet-scale) | SGD + Momentum |
| NLP / Transformers | AdamW |
| Sparse data | AdaGrad |
| RL / non-stationary | RMSprop |

---

## 7-2 Learning Rate Scheduling

### Common Schedulers

```python
from torch.optim.lr_scheduler import (
    StepLR, MultiStepLR, CosineAnnealingLR,
    ReduceLROnPlateau
)

# Step decay: reduce by 0.1 every 30 epochs
scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

# Cosine annealing: smooth decay
scheduler = CosineAnnealingLR(optimizer, T_max=100)

# Adaptive: reduce when loss plateaus
scheduler = ReduceLROnPlateau(optimizer, mode='min',
                              factor=0.5, patience=5)
```

### Learning Rate Best Practices

| Strategy | Use Case |
|:---------|:---------|
| Constant | Simple problems, short training |
| Step decay | Standard CV training |
| Cosine | Modern deep learning |
| One-cycle | Fast convergence |
| Warmup + decay | Transformers |

---

## 7-3 Regularization and Generalization

### L2 Regularization (Weight Decay)

Adds penalty for large weights:

$$
L_{\text{total}} = L_{\text{original}} + \frac{\lambda}{2} \sum \|w\|^2
$$

```python
# Option 1: Add to loss manually
l2_lambda = 0.001
l2_loss = sum(p.pow(2).sum() for p in model.parameters())
total_loss = original_loss + l2_lambda * l2_loss

# Option 2: weight_decay in optimizer (AdamW does this properly)
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
```

### Dropout

Randomly "drops" neurons during training:

```python
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Dropout(0.5),  # 50% probability
    nn.Linear(256, 10),
)
```

### Early Stopping

Stop training when validation loss stops improving:

```python
best_loss = float('inf')
patience = 5
wait = 0

for epoch in range(100):
    train_loss = train_epoch()
    val_loss = validate()

    if val_loss < best_loss:
        best_loss = val_loss
        wait = 0
        torch.save(model.state_dict(), 'best_model.pt')
    else:
        wait += 1
        if wait >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
```

---

## 7-4 Normalization Methods

### Batch Normalization

Normalizes each batch to have mean 0, variance 1:

```python
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.BatchNorm1d(256),
    nn.ReLU(),
    nn.Linear(256, 10),
)
```

### Comparison

| Method | Normalizes Across | Best For |
|:-------|:------------------|:---------|
| BatchNorm | Batch dimension | CV (fixed-size batches) |
| LayerNorm | Feature dimension | NLP / Transformers |
| InstanceNorm | Single sample | Style transfer |
| GroupNorm | Groups of channels | Small batch sizes |

---

## 7-5 Weight Initialization

### Why Initialization Matters

Bad initialization → vanishing/exploding gradients. Good initialization → fast convergence.

### Common Strategies

```python
# Default PyTorch (Kaiming Uniform for ReLU)
layer = nn.Linear(784, 256)  # automatically initialized

# Manual initialization
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)  # for tanh/sigmoid
        # nn.init.kaiming_uniform_(m.weight)  # for ReLU
        nn.init.zeros_(m.bias)

model.apply(init_weights)
```

---

## 7-6 Loss Functions Overview

### Classification

| Loss | Formula | Use Case |
|:-----|:--------|:---------|
| Cross-Entropy | $-\sum t_k \log p_k$ | Multi-class |
| Binary CE | $-(t\log p + (1-t)\log(1-p))$ | Binary |
| Focal Loss | Weighted CE | Imbalanced classes |

### Regression

| Loss | Formula | Robustness |
|:-----|:--------|:-----------|
| MSE | $\frac{1}{2}(y-t)^2$ | Sensitive to outliers |
| MAE | $\|y-t\|$ | Robust to outliers |
| Huber | MSE for small error, MAE for large | Best of both |

---

## 7-7 Deep Understanding of Softmax ⭐

### Beyond the Basic Definition

Softmax converts logits to probabilities, but its behavior depends on **temperature**:

$$
p_k = \frac{e^{z_k / T}}{\sum_{j} e^{z_j / T}}
$$

| Temperature | Effect |
|:------------|:-------|
| $T \to 0$ | Becomes argmax (hard assignment) |
| $T = 1$ | Standard Softmax |
| $T \to \infty$ | Becomes uniform distribution |

### Temperature in Practice

```python
def softmax_with_temperature(logits, temperature=1.0):
    """Softmax with temperature scaling"""
    scaled_logits = logits / temperature
    exp_logits = torch.exp(scaled_logits - torch.max(scaled_logits))
    return exp_logits / exp_logits.sum(dim=-1, keepdim=True)

logits = torch.tensor([2.0, 1.0, 0.1])
for T in [0.5, 1.0, 2.0, 5.0]:
    probs = softmax_with_temperature(logits, T)
    print(f"T={T:.1f}: {probs.numpy().round(3)}")
```

---

## 7-8 Regularization Deep Dive

### Which Regularization When?

| Method | Effect | When to Add |
|:-------|:-------|:------------|
| L2 | Small weights | Always (mild) |
| Dropout | Prevents co-adaptation | Large networks, overfitting |
| Early stopping | Limits training time | Always |
| Data augmentation | More data implicitly | CV tasks |
| Label smoothing | Softens targets | Classification |

---

## 7-9 Hyperparameter Tuning

### Key Hyperparameters

```text
Learning rate:      Most important → tune first!
Batch size:         Memory vs. convergence
Hidden units:       Capacity
Number of layers:   Depth
Dropout rate:       Regularization strength
Weight decay:       L2 strength
```

### Tuning Strategies

1. **Grid search**: Systematic but expensive
2. **Random search**: Better for high-dimensional spaces
3. **Bayesian optimization**: Efficient but complex
4. **Learning rate finder**: Find optimal lr quickly

---

## 7-10 Gradient Clipping

Prevents exploding gradients by scaling down large gradients:

```python
# Before stepping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

---

## 📖 Chapter Summary

```text
Optimizers → How to follow the gradient (SGD → Adam → AdamW)
Scheduling → When to change step size
Regularization → Prevent overfitting (L2, dropout, early stopping)
Normalization → Stable training (BatchNorm, LayerNorm)
Initialization → Start in the right region
Loss functions → Define the goal
```

### 🧪 Exercises

#### Exercise 1: Optimizer Comparison
Train MNIST with SGD, SGD+Momentum, Adam. Compare epochs to 95% accuracy.

#### Exercise 2: Regularization
Train a network with and without dropout. Compare train vs. test accuracy.

#### Exercise 3: BatchNorm
Add BatchNorm to a deep network (5+ layers). Does it help convergence?

#### Exercise 4: Learning Rate Finder
Implement a learning rate finder: start lr=1e-5, increase exponentially each batch, plot loss vs. lr.
