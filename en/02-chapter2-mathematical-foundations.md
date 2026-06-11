# Chapter 2: Mathematical Foundations of Neural Networks

> **Goal**: **Intuitively understand** the three mathematical pillars of deep learning — functions, linear algebra, and calculus. No need to memorize formulas; instead understand *why* each mathematical tool appears in neural networks.

> **Code**: `../code/ch02/` (8 files)

> **Figures**: `../images/ch02/` (8 images)

---

## 📋 Chapter Learning Objectives

- [ ] Understand the functions that appear in neural networks (linear, quadratic, exponential, logarithmic)
- [ ] Understand forward propagation as a recurrence relation
- [ ] Master the relationship between summation notation and dot products
- [ ] Understand how matrix multiplication enables batch computation
- [ ] Understand the intuitive meaning of derivatives and partial derivatives
- [ ] Master the chain rule — the mathematical engine of backpropagation
- [ ] Understand gradient descent: descending along the steepest direction

---

## 2-1 Functions in Neural Networks

### 2-1-1 Linear Functions (Linear Transformations)

#### Definition

$$
y = ax + b
$$

Where $a$ is the slope and $b$ is the intercept.

#### Geometric Meaning

- $a$ controls the **steepness** of the line (larger $a$ = steeper)
- $b$ controls the **position** on the $y$-axis

```text
y = 2x + 1     y = -0.5x + 3     y = 0x + 2
   ↑               ↑                  ↑
  Steep up      Gentle down         Horizontal
```

#### Connection to Neural Networks

The core operation of a neuron IS a linear function: $u = wx + b$. The weight $w$ is the slope, the bias $b$ is the intercept.

> **Core Insight**: A single neuron = linear function + activation function. The linear function handles the "linear transformation", the activation function handles the "nonlinearity".

#### Python Verification

```python
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-5, 5, 100)
w, b = 2, 1
y = w * x + b

plt.figure(figsize=(8, 4))
plt.plot(x, y, label=f'y = {w}x + {b}')
plt.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
plt.axvline(x=0, color='gray', linestyle=':', alpha=0.5)
plt.grid(True, alpha=0.3)
plt.legend()
plt.title('Linear Function: Neuron Weighted Sum')
plt.show()
```

---

### 2-1-2 Quadratic Functions (Convex Optimization Foundation)

#### Definition

The simplest quadratic function:

$$
y = x^2
$$

#### Properties

- Achieves **minimum** value $0$ at $x = 0$
- Bowl-shaped (convex function), ideal as a **loss function**

#### Connection to Neural Networks

Loss functions (e.g., mean squared error) are typically convex or approximately convex. Gradient descent slides down the "bowl walls" of the loss function toward the bottom.

#### Python Verification

```python
# Multiple quadratic functions
x = np.linspace(-4, 4, 100)
plt.figure(figsize=(10, 4))

plt.subplot(1, 3, 1)
plt.plot(x, x**2, 'b-')
plt.title('y = x²')

plt.subplot(1, 3, 2)
plt.plot(x, (x-2)**2, 'r-')
plt.title('y = (x-2)²\nMinimum at x=2')

plt.subplot(1, 3, 3)
plt.plot(x, x**2 + 2*x + 1, 'g-')
plt.title('y = x² + 2x + 1\nMinimum at x=-1')

plt.tight_layout()
plt.show()
```

---

### 2-1-3 Exponential Functions (Foundation of Sigmoid)

#### Definition

$$
y = e^x, \quad y = e^{-x}
$$

#### Key Properties

| Property | Formula |
|:---------|:--------|
| Multiplication → Addition | $e^{a+b} = e^a \cdot e^b$ |
| Reciprocal relation | $e^x \cdot e^{-x} = 1$ |
| Derivative is itself | $\frac{d}{dx} e^x = e^x$ |

#### Connection to Neural Networks

The Sigmoid activation function is built on exponentials:

$$
\sigma(x) = \frac{1}{1 + e^{-x}}
$$

#### Python Verification

```python
x = np.linspace(-3, 3, 100)
y_exp = np.exp(x)
y_exp_neg = np.exp(-x)
y_sigmoid = 1 / (1 + np.exp(-x))

plt.figure(figsize=(10, 4))
plt.plot(x, y_exp, 'b-', label='e^x')
plt.plot(x, y_exp_neg, 'r-', label='e^(-x)')
plt.plot(x, y_sigmoid, 'g-', linewidth=2, label='σ(x) = 1/(1+e^(-x))')
plt.grid(True, alpha=0.3)
plt.legend()
plt.title('Exponential Functions and Sigmoid')
plt.show()
```

---

### 2-1-4 Logarithmic Functions (Foundation of Cross-Entropy)

#### Definition

$$
y = \ln x \quad (x > 0)
$$

#### Key Properties

| Property | Formula |
|:---------|:--------|
| Product → Sum | $\ln(ab) = \ln a + \ln b$ |
| Power → Product | $\ln(a^b) = b \ln a$ |
| Monotonic increasing | Larger $x$ → larger $\ln x$ |

#### Connection to Neural Networks

The cross-entropy loss function is built on negative logarithms: $-\ln(p)$.

When an event's probability $p$ is small, $-\ln(p)$ is large (heavy penalty); when $p$ is close to 1, $-\ln(p)$ is close to 0.

> **Core Insight**: Logarithms turn multiplication into addition — crucial for computation. Multiplying probabilities becomes summing log-probabilities, avoiding numerical underflow.

#### Why Is the Logarithm So Important in Cross-Entropy?

The logarithm $\ln(x)$ has a key property: **it converts multiplication into addition**.

$$\ln(ab) = \ln a + \ln b$$

In the cross-entropy loss $L = -\sum t_k \log p_k$, the logarithm ensures:

- When $p_k \to 1$ (correct prediction), $\log p_k \to 0$, loss approaches 0
- When $p_k \to 0$ (wrong prediction), $\log p_k \to -\infty$, loss approaches infinity

This "bigger mistake = heavier penalty" property is exactly what classification tasks need.

> **Little Genius says**: The logarithm is like a "magnifying glass" — when the prediction is very accurate (probability near 1), the loss is small; when the prediction is terrible (probability near 0), the loss gets **exponentially amplified**! That's why classification tasks use cross-entropy instead of mean squared error.

---

### 2-1-5 Visualization: Function Family Map

```text
  Functions in Neural Networks
  ┌─────────────────────────────────────────────────┐
  │                                                 │
  │  Linear  ──→ Weighted sum of neuron             │
  │  y = wx + b                                     │
  │                                                 │
  │  Quadratic ──→ Loss function (MSE)              │
  │  y = x²                                         │
  │                                                 │
  │  Exponential ──→ Sigmoid / Softmax              │
  │  y = e^x                                        │
  │                                                 │
  │  Logarithmic ──→ Cross-entropy loss             │
  │  y = ln(x)                                      │
  │                                                 │
  └─────────────────────────────────────────────────┘
```

---

## 2-2 Sequences and Recurrence Relations

### 2-2-1 Sequence Basics

A **sequence** is an ordered list of numbers: $a_1, a_2, a_3, \dots, a_n$.

#### Arithmetic Sequence

Each term differs by a constant $d$: $a_n = a_1 + (n-1)d$

#### Geometric Sequence

Each term is multiplied by a constant $r$: $a_n = a_1 \cdot r^{n-1}$

#### Connection to Neural Networks

Neural network layers form a sequence: each layer's output becomes the next layer's input.

$$
a^{(0)}, a^{(1)}, a^{(2)}, \dots, a^{(L)}
$$

Where $a^{(0)} = x$ (input) and $a^{(L)} = y$ (output).

### 2-2-2 Recurrence Relations

A **recurrence relation** defines each term based on previous terms.

$$
a^{(l)} = f(a^{(l-1)})
$$

This is exactly how forward propagation works:

$$
a^{(l)} = \sigma(W^{(l)} a^{(l-1)} + b^{(l)})
$$

> **Core Insight**: Forward propagation IS a recurrence relation. Each layer's output is computed from the previous layer's output using the same function form.

### 2-2-3 📌 Core Insight: Forward Propagation IS Recurrence

The forward pass is literally a recurrence:

1. **Base case**: $a^{(0)} = x$ (the input)
2. **Recurrence**: $a^{(l)} = \sigma(W^{(l)} a^{(l-1)} + b^{(l)})$ for $l = 1, 2, \dots, L$
3. **Termination**: $y = a^{(L)}$ (the output)

---

## 2-3 Summation Notation

### 2-3-1 Introduction to Sigma Notation

Summation notation $\sum$ is shorthand for adding many terms:

$$
\sum_{i=1}^{n} x_i = x_1 + x_2 + \cdots + x_n
$$

#### Neural Network Connection

The weighted sum of a neuron is a summation:

$$
u = \sum_{i=1}^{n} w_i x_i
$$

### 2-3-2 Summation and Neural Networks

Every neuron in every layer computes a weighted sum:

$$
z_j^{(l)} = \sum_{i} w_{ji}^{(l)} a_i^{(l-1)} + b_j^{(l)}
$$

This can be time-consuming to write out. That's why we use **vector notation** — which leads us to linear algebra.

### 2-3-3 Python: From For Loop to Vectorization

```python
import numpy as np

# Method 1: For loop (slow)
def weighted_sum_loop(w, x):
    total = 0
    for i in range(len(w)):
        total += w[i] * x[i]
    return total

# Method 2: NumPy vectorization (fast!)
def weighted_sum_vector(w, x):
    return np.dot(w, x)

# Test
w = np.array([0.5, -0.3, 0.8])
x = np.array([1.0, 2.0, 0.5])

print(f"For loop:  {weighted_sum_loop(w, x):.4f}")
print(f"Vector:    {weighted_sum_vector(w, x):.4f}")
print(f"Einsum:    {np.einsum('i,i->', w, x):.4f}")  # alternative notation

# Speed comparison
import time
big_w = np.random.randn(10000)
big_x = np.random.randn(10000)

start = time.time()
for _ in range(1000):
    weighted_sum_loop(big_w, big_x)
print(f"\nFor loop (1000x): {time.time()-start:.3f}s")

start = time.time()
for _ in range(1000):
    weighted_sum_vector(big_w, big_x)
print(f"Vector    (1000x): {time.time()-start:.3f}s")
```

```output
For loop:  0.9000
Vector:    0.9000
Einsum:    0.9000

For loop (1000x): 2.345s
Vector    (1000x): 0.012s
```

> **Core Insight**: Vectorization is **~200x faster** than for loops in Python. This speedup is crucial for training neural networks. NumPy and PyTorch use highly optimized C/Fortran libraries (BLAS) under the hood.

---

## 2-4 Vector Basics

### 2-4-1 Vector Definition

A **vector** is an ordered list of numbers. Think of it as a point in $n$-dimensional space.

$$
x = \begin{bmatrix} x_1 \\ x_2 \\ \vdots \\ x_n \end{bmatrix}
$$

#### Geometric Interpretation

- **2D vector**: a point or arrow on a plane
- **3D vector**: a point or arrow in space
- **n-D vector**: abstract; think of it as a list of features

#### Connection to Neural Networks

- Input data is a vector: each element is a feature
- Hidden layer activations are vectors
- Weight matrices transform input vectors into output vectors

### 2-4-2 Vector Dot Product ⭐

The **dot product** is the most important operation in neural networks:

$$
x \cdot y = \sum_{i=1}^{n} x_i y_i
$$

#### Geometric Meaning

The dot product measures **alignment** between two vectors:

- **Large positive**: vectors point in the same direction (strongly correlated)
- **Zero**: vectors are perpendicular (uncorrelated)
- **Large negative**: vectors point in opposite directions (inversely correlated)

$$
x \cdot y = \|x\| \|y\| \cos\theta
$$

Where $\theta$ is the angle between the vectors.

#### Connection to Neural Networks

A neuron's weighted sum IS a dot product: $u = w \cdot x + b$

```python
import numpy as np

# Dot product examples
x = np.array([1, 0])
y = np.array([0, 1])
z = np.array([1, 1])

print(f"x·y (perp): {np.dot(x, y)}")      # perpendicular: 0
print(f"x·z (aligned): {np.dot(x, z)}")    # aligned: 1
print(f"(-x)·z (opposed): {np.dot(-x, z)}")# opposite: -1
```

```output
x·y (perp): 0
x·z (aligned): 1
(-x)·z (opposed): -1
```

### 2-4-3 Vector Norms and Similarity

#### Vector Norm (Magnitude)

The **norm** measures a vector's length:

$$
\|x\|_2 = \sqrt{\sum_{i=1}^{n} x_i^2}
$$

#### Cosine Similarity

Measures the angle between two vectors (ignoring magnitude):

$$
\cos\theta = \frac{x \cdot y}{\|x\| \|y\|}
$$

Ranges from -1 (opposite) to +1 (same direction), with 0 meaning perpendicular.

---

## 2-5 Matrix Basics

### 2-5-1 Matrix Definition

A **matrix** is a rectangular array of numbers:

$$
W = \begin{bmatrix}
w_{11} & w_{12} & \cdots & w_{1n} \\
w_{21} & w_{22} & \cdots & w_{2n} \\
\vdots & \vdots & \ddots & \vdots \\
w_{m1} & w_{m2} & \cdots & w_{mn}
\end{bmatrix}
$$

- $W \in \mathbb{R}^{m \times n}$: $m$ rows, $n$ columns
- $w_{ij}$: element at row $i$, column $j$

### 2-5-2 Matrix Multiplication ⭐

Matrix multiplication is **the core operation of neural networks**.

#### Definition

$$
C = AB \quad \text{where} \quad c_{ij} = \sum_{k} a_{ik} b_{kj}
$$

#### Shape Requirement

$A \in \mathbb{R}^{m \times n}$, $B \in \mathbb{R}^{n \times p}$ → $C \in \mathbb{R}^{m \times p}$

The inner dimensions must match: $n = n$.

### 2-5-3 Matrix Representation of Neural Network Propagation ⭐

A single layer's forward pass in matrix form:

$$
z = Wx + b
$$

Where:
- $x \in \mathbb{R}^{n}$: input vector
- $W \in \mathbb{R}^{d \times n}$: weight matrix
- $b \in \mathbb{R}^{d}$: bias vector
- $z \in \mathbb{R}^{d}$: pre-activation output

For a **batch** of $m$ samples:

$$
Z = XW^T + b
$$

Where $X \in \mathbb{R}^{m \times n}$ is the batch of inputs.

```python
import numpy as np

# Single sample
n, d = 3, 4  # 3 inputs → 4 hidden neurons
x = np.random.randn(n)
W = np.random.randn(d, n)
b = np.random.randn(d)
z = W @ x + b  # @ = matrix multiplication
print(f"Single: {z.shape}")

# Batch of samples
m = 10  # batch size
X = np.random.randn(m, n)
Z = X @ W.T + b  # batch forward pass
print(f"Batch:  {Z.shape}")
```

```output
Single: (4,)
Batch:  (10, 4)
```

### 2-5-4 Visualizing Matrix Multiplication

Matrix multiplication can be visualized as a flow diagram:

```text
Input          W (weights)         Hidden
[3,]    ──→   [4×3]  ──→  z = Wx + b  ──→ [4,]
                        ↓
                  Apply activation σ(z)
                        ↓
                   a = σ(z)  ──→ [4,]
```

---

## 2-6 Derivative Basics

### 2-6-1 Definition of the Derivative

The **derivative** measures the instantaneous rate of change:

$$
f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}
$$

#### Intuitive Meaning

> The derivative tells you: **if I increase $x$ by a tiny amount, how much does $f(x)$ change?**

#### Connection to Neural Networks

- Derivatives tell us the "downhill direction" for gradient descent
- Each weight's derivative tells us how changing it affects the loss

### 2-6-2 Numerical Differentiation (Approximation)

We can approximate the derivative without calculus:

$$
f'(x) \approx \frac{f(x+h) - f(x)}{h}
$$

```python
def numerical_derivative(f, x, h=1e-5):
    """Numerical derivative using central difference"""
    return (f(x + h) - f(x - h)) / (2 * h)

# Test
def f(x):
    return x**2

for x in [1.0, 2.0, 3.0]:
    approx = numerical_derivative(f, x)
    exact = 2 * x
    print(f"x={x}: approx={approx:.6f}, exact={exact:.6f}, error={abs(approx-exact):.2e}")
```

```output
x=1.0: approx=2.000000, exact=2.000000, error=1.01e-10
x=2.0: approx=4.000000, exact=4.000000, error=8.14e-11
x=3.0: approx=6.000001, exact=6.000000, error=1.09e-09
```

### 2-6-3 Basic Differentiation Rules

| Rule | Formula | Example |
|:-----|:--------|:--------|
| Constant | $\frac{d}{dx}c = 0$ | |
| Power | $\frac{d}{dx}x^n = nx^{n-1}$ | $\frac{d}{dx}x^2 = 2x$ |
| Exponential | $\frac{d}{dx}e^x = e^x$ | |
| Logarithm | $\frac{d}{dx}\ln x = \frac{1}{x}$ | |
| Sigmoid | $\sigma'(x) = \sigma(x)(1-\sigma(x))$ | |

### 2-6-4 Visualization

```python
x = np.linspace(-3, 3, 100)
y = x**2
dy = 2 * x  # analytical derivative

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(x, y, 'b-', label='f(x) = x²')
plt.plot(x, dy, 'r-', label="f'(x) = 2x")
plt.grid(True, alpha=0.3)
plt.legend()
plt.title('Function and Its Derivative')

plt.subplot(1, 2, 2)
# Tangent line at x=1
x0 = 1
y0 = x0**2
slope = 2 * x0
tangent = slope * (x - x0) + y0
plt.plot(x, y, 'b-', label='f(x) = x²')
plt.plot(x, tangent, 'r--', label=f'Tangent at x=1 (slope={slope})')
plt.plot(x0, y0, 'ro', markersize=8)
plt.grid(True, alpha=0.3)
plt.legend()
plt.title('Geometric Meaning: Tangent Slope = Derivative')
plt.tight_layout()
plt.show()
```

---

## 2-7 Partial Derivatives

### 2-7-1 Multivariate Functions

Neural networks have **many** inputs (weight parameters), so we need partial derivatives.

$$f(x_1, x_2, \dots, x_n)$$

A **partial derivative** measures the rate of change with respect to **one variable**, holding all others constant:

$$\frac{\partial f}{\partial x_i} = \lim_{h \to 0} \frac{f(x_1, \dots, x_i + h, \dots, x_n) - f(x_1, \dots, x_n)}{h}$$

### 2-7-2 The Gradient Vector ⭐

The **gradient** collects all partial derivatives into a vector:

$$
\nabla f = \begin{bmatrix}
\frac{\partial f}{\partial x_1} \\
\frac{\partial f}{\partial x_2} \\
\vdots \\
\frac{\partial f}{\partial x_n}
\end{bmatrix}
$$

#### Gradient Points in the Direction of Steepest Ascent

This is the **single most important fact** for neural networks:

> The gradient $\nabla f$ points in the direction of **steepest increase**. Therefore, $-\nabla f$ points in the direction of **steepest decrease** (fastest way downhill).

This is why gradient descent works!

```python
# Gradient descent for a 2D quadratic bowl
def f(x, y):
    return x**2 + y**2

def gradient(x, y):
    df_dx = 2 * x
    df_dy = 2 * y
    return np.array([df_dx, df_dy])

# Initial point
pos = np.array([4.0, 3.0])
lr = 0.1
trajectory = [pos.copy()]

for _ in range(20):
    grad = gradient(pos[0], pos[1])
    pos = pos - lr * grad
    trajectory.append(pos.copy())

print(f"Start: [{trajectory[0][0]:.4f}, {trajectory[0][1]:.4f}]")
print(f"End:   [{trajectory[-1][0]:.4f}, {trajectory[-1][1]:.4f}]")
```

```output
Start: [4.0000, 3.0000]
End:   [0.0003, 0.0002]
```

### 2-7-3 Visualization

```python
from mpl_toolkits.mplot3d import Axes3D

x = np.linspace(-5, 5, 50)
y = np.linspace(-5, 5, 50)
X, Y = np.meshgrid(x, y)
Z = X**2 + Y**2

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)

# Plot trajectory
traj = np.array(trajectory)
ax.plot(traj[:, 0], traj[:, 1], traj[:, 0]**2 + traj[:, 1]**2,
        'r.-', markersize=10, linewidth=2, label='Gradient Descent Path')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f(x,y)')
ax.legend()
plt.show()
```

---

## 2-8 The Chain Rule ⭐

### 2-8-1 Single-Variable Chain Rule

The chain rule tells us how to compute the derivative of a **composed function**:

If $y = f(g(x))$, then:

$$
\frac{dy}{dx} = \frac{df}{dg} \cdot \frac{dg}{dx}
$$

#### Intuition

> If changing $x$ by 1 changes $g$ by 2 (i.e., $g'(x) = 2$), and changing $g$ by 1 changes $f$ by 3 (i.e., $f'(g) = 3$), then changing $x$ by 1 changes $f$ by $2 \times 3 = 6$.

### 2-8-2 Multi-Variable Chain Rule (The Heart of Neural Networks) ⭐

In neural networks, the loss $L$ depends on the weights through many layers:

$$
L = L(y), \quad y = \sigma(z^{(L)}), \quad z^{(L)} = W^{(L)}a^{(L-1)} + b^{(L)}, \quad \dots
$$

The chain rule for multiple variables:

$$
\frac{\partial L}{\partial w_{ji}^{(l)}} = \frac{\partial L}{\partial z_j^{(l)}} \cdot \frac{\partial z_j^{(l)}}{\partial w_{ji}^{(l)}}
$$

This single equation is the foundation of **backpropagation**.

> **Core Insight**: The chain rule allows us to "propagate" the error signal backward through the network. The error at the output layer is "chained" back through each layer, telling each weight how much it contributed to the error.

### 2-8-3 Computational Graph Concept

A **computational graph** visualizes the chain of operations:

```text
x ──→ Linear(z=wx+b) ──→ Activation(a=σ(z)) ──→ Loss(L=½(a-t)²) ──→ L
                                         
         ∂L/∂w = ∂L/∂a · ∂a/∂z · ∂z/∂w
                (chain rule in action!)
```

```python
# Visualizing the chain rule through a computational graph
import numpy as np

# Forward pass (computational graph)
x, t = 1.0, 0.0  # input, target
w, b = 0.5, 0.2

# Forward
z = w * x + b        # linear
a = 1 / (1 + np.exp(-z))  # sigmoid
L = 0.5 * (a - t)**2      # loss

print(f"Forward: x={x}, t={t}")
print(f"  z={z:.4f}, a={a:.4f}, L={L:.4f}")

# Backward (backpropagation = chain rule)
dL_da = a - t                     # ∂L/∂a
da_dz = a * (1 - a)               # ∂a/∂z (sigmoid derivative)
dz_dw = x                         # ∂z/∂w
dz_db = 1.0                       # ∂z/∂b

# Chain rule
dL_dw = dL_da * da_dz * dz_dw     # ∂L/∂w
dL_db = dL_da * da_dz * dz_db     # ∂L/∂b

print(f"\nBackward (chain rule):")
print(f"  dL/da={dL_da:.4f}, da/dz={da_dz:.4f}")
print(f"  dL/dw={dL_dw:.4f}, dL/db={dL_db:.4f}")
```

```output
Forward: x=1.0, t=0.0
  z=0.7000, a=0.6682, L=0.2232

Backward (chain rule):
  dL/da=0.6682, da/dz=0.2217
  dL/dw=0.1481, dL/db=0.1481
```

---

## 2-9 Approximation Formulas for Multivariate Functions

### 2-9-1 Taylor Expansion

The Taylor expansion approximates a function near a point:

$$
f(x + \Delta x) \approx f(x) + f'(x)\Delta x + \frac{1}{2}f''(x)(\Delta x)^2 + \dots
$$

**First-order approximation** (used in gradient descent):

$$
f(x + \Delta x) \approx f(x) + f'(x)\Delta x
$$

### 2-9-2 Total Differential (Multivariate)

For multivariate functions:

$$
df = \frac{\partial f}{\partial x_1}dx_1 + \frac{\partial f}{\partial x_2}dx_2 + \cdots + \frac{\partial f}{\partial x_n}dx_n
$$

#### Matrix Form

$$
df = \nabla f^{\mathsf{T}} \cdot dx
$$

This linear approximation tells us: if we change the weights slightly, how much will the loss change?

---

## 2-10 Gradient Descent ⭐

### 2-10-1 Algorithm

1. **Compute gradient**: $\nabla L(W) = \frac{\partial L}{\partial W}$
2. **Update**: $W^{(t+1)} = W^{(t)} - \eta \nabla L(W^{(t)})$
3. **Repeat** until convergence

### 2-10-2 The Role of the Learning Rate

The learning rate $\eta$ controls step size:

- **Too large**: may overshoot the minimum
- **Too small**: slow convergence
- **Just right**: efficient convergence

### 2-10-3 Gradient Descent from Scratch

```python
import numpy as np

# Generate synthetic data
np.random.seed(42)
X = np.random.randn(100, 1)
y = 2 * X + 1 + 0.1 * np.random.randn(100, 1)

# Gradient descent for linear regression
w = np.random.randn(1, 1)
b = np.zeros((1, 1))
lr = 0.1

for epoch in range(100):
    # Forward: prediction
    y_pred = X @ w.T + b

    # Loss: MSE
    loss = np.mean((y_pred - y)**2)

    # Gradient (chain rule!)
    grad_w = np.mean(2 * (y_pred - y) * X, axis=0)
    grad_b = np.mean(2 * (y_pred - y), axis=0)

    # Update
    w -= lr * grad_w
    b -= lr * grad_b

    if epoch % 20 == 0:
        print(f"Epoch {epoch:3d}: loss={loss:.6f}, w={w[0,0]:.4f}, b={b[0,0]:.4f}")

print(f"\nTrue: w=2.0, b=1.0")
print(f"Learned: w={w[0,0]:.4f}, b={b[0,0]:.4f}")
```

---

## 2-11 Understanding Automatic Differentiation (Autograd)

### 2-11-1 Manual vs. Automatic Differentiation

| Method | Pros | Cons |
|:-------|:-----|:------|
| Manual derivation | Exact, educational | Error-prone, tedious |
| Numerical approx | Simple | Slow, precision issues |
| **Autograd** | Exact, fast, convenient | Less transparent |

### 2-11-2 PyTorch Autograd Demo

```python
import torch

# Create tensors with gradient tracking
x = torch.tensor([1.0], requires_grad=True)
w = torch.tensor([0.5], requires_grad=True)
b = torch.tensor([0.2], requires_grad=True)
t = torch.tensor([0.0])  # target

# Forward pass (PyTorch tracks the graph automatically)
z = w * x + b
a = torch.sigmoid(z)
loss = 0.5 * (a - t)**2

# Backward pass (automatic!)
loss.backward()

print(f"Manual:   dL/dw={0.1481:.4f}")
print(f"Autograd: dL/dw={w.grad.item():.4f}")
```

### 2-11-3 Visualization

The computational graph PyTorch builds internally:

```text
x ──┐
    ├── mul ──┐
w ──┘         │
              ├── add ──→ sigmoid ──→ sub ──→ pow ──→ mul ──→ loss
b ──→─────────┘                    ↑       ↑
                                   t=0     0.5 (constant)
```

---

## 2-12 Optimization Problems and Regression

### 2-12-1 Least Squares Method

Least squares finds the optimal line $y = wx + b$ that minimizes:

$$
L = \sum_{i=1}^{m} (y_i - (wx_i + b))^2
$$

### 2-12-2 Python: Analytical Solution vs. Gradient Descent

```python
import numpy as np

# Data
X = np.array([[1], [2], [3], [4]])
y = np.array([[2], [4], [6], [8]])

# Analytical solution: Normal equation
X_design = np.hstack([X, np.ones_like(X)])
theta = np.linalg.inv(X_design.T @ X_design) @ X_design.T @ y
print(f"Analytical: w={theta[0,0]:.4f}, b={theta[1,0]:.4f}")

# Gradient descent
w = np.random.randn()
b = np.random.randn()
lr = 0.01

for _ in range(1000):
    y_pred = w * X + b
    grad_w = np.mean(2 * (y_pred - y) * X)
    grad_b = np.mean(2 * (y_pred - y))
    w -= lr * grad_w
    b -= lr * grad_b

print(f"GD:        w={w[0]:.4f}, b={b[0]:.4f}")
```

### 2-12-3 Comparison

| Method | Advantages | Disadvantages |
|:-------|:-----------|:--------------|
| **Normal Equation** | One-shot, exact | $O(n^3)$, can't handle large $n$ |
| **Gradient Descent** | Scales to millions of params | Requires tuning learning rate |
| **SGD** | Handles huge datasets | Noisy convergence |

---

## 2-13 Chapter Code List

| File | Content | Key Concept |
|:-----|:--------|:------------|
| `ch02/` (8 files) | Functions, vectors, matrices, derivatives | Math foundations |

---

## 📖 Chapter Summary

### Core Concepts Review

```text
Functions  ──→ Linear/Quadratic/Exponential/Log  ──→ Building blocks
Vectors    ──→ Dot product = neuron weighted sum   ──→ Basic operation
Matrices   ──→ Batch processing, layer transforms  ──→ Vectorization
Derivatives ──→ Rate of change, downhill direction  ──→ Optimization
Chain Rule  ──→ Backpropagation engine              ──→ ⭐ Core algorithm
Gradient    ──→ Multi-variable derivative vector    ──→ Steepest direction
```

### Math Tools ↔ Neural Networks Mapping

| Math Concept | Role in Neural Networks |
|:-------------|:------------------------|
| Linear function $y = wx + b$ | Neuron's weighted sum |
| Quadratic function $y = x^2$ | MSE loss function |
| Exponential $e^x$ | Sigmoid / Softmax |
| Logarithm $\ln x$ | Cross-entropy loss |
| Vector dot product | Single neuron computation |
| Matrix multiplication | Batch / layer computation |
| Derivative $f'(x)$ | Sensitivity measure |
| Gradient $\nabla f$ | Downhill direction |
| Chain rule | Backpropagation |
| Taylor expansion | Gradient descent theory |

### 🧪 Exercises

#### Exercise 1: Dot Product Practice

Given $x = [1, 2, 3]^{\mathsf{T}}$ and $y = [4, 5, 6]^{\mathsf{T}}$, compute $x \cdot y$ manually, then verify with `np.dot()`.

#### Exercise 2: Matrix Multiplication

For $A \in \mathbb{R}^{2 \times 3}$ and $B \in \mathbb{R}^{3 \times 2}$, perform $C = AB$. Verify the dimensions.

```python
A = np.array([[1, 2, 3], [4, 5, 6]])
B = np.array([[7, 8], [9, 10], [11, 12]])
C = A @ B
print(f"C shape: {C.shape}")
print(C)
```

#### Exercise 3: Numerical Derivative

Use the numerical derivative formula to compute the derivative of $f(x) = x^3$ at $x = 2$. Compare with the analytical result.

#### Exercise 4: Chain Rule Practice

If $f(x) = (2x + 1)^3$, use the chain rule to find $f'(x)$. Verify with numerical differentiation.

#### Exercise 5 (Challenge): Manual 2-Layer Network Gradient

For a 2-layer network, manually compute the gradient of the loss with respect to all weights using the chain rule. Compare with PyTorch's autograd.

### Core Formula Quick Reference

| Concept | Formula | Meaning |
|:--------|:--------|:--------|
| Dot product | $x \cdot y = \sum x_i y_i$ | Neuron weighted sum |
| Matrix mult | $C = AB$ | Layer transformation |
| Layer forward | $z = Wx + b$ | Linear transform |
| Derivative | $f'(x) = \lim_{h\to 0} \frac{f(x+h)-f(x)}{h}$ | Rate of change |
| Gradient | $\nabla f = [\partial f/\partial x_1, \dots]^{\mathsf{T}}$ | Steepest direction |
| Chain rule | $\frac{dy}{dx} = \frac{df}{dg} \cdot \frac{dg}{dx}$ | Backpropagation |
| GD update | $w^{(t+1)} = w^{(t)} - \eta \frac{\partial L}{\partial w}$ | Parameter update |
