# Appendix C: Notation Conventions

> This appendix defines all mathematical symbols used in this book. The same symbol may have different meanings in different contexts; see chapter-specific explanations.

---

## C-1 Basic Symbols

| Symbol | Meaning | Example |
|:-------|:--------|:--------|
| $x, y, z$ | Scalar (lowercase italic) | $x = 3.14$ |
| $\mathbf{x}, \mathbf{w}$ | Vector (lowercase bold) | $\mathbf{x} \in \mathbb{R}^d$ |
| $\mathbf{W}, \mathbf{X}$ | Matrix (uppercase bold) | $\mathbf{W} \in \mathbb{R}^{d \times h}$ |
| $\mathbb{R}$ | Set of real numbers | $\mathbb{R}^n$ denotes $n$-dim real space |

---

## C-2 Neural Network Specific Symbols

### C-2-1 Network Structure

| Symbol | Meaning |
|:-------|:--------|
| $L$ | Number of layers |
| $n^{(l)}$ | Number of neurons in layer $l$ |
| $m$ | Batch size (number of samples) |
| $d$ | Input dimension |

### C-2-2 Layer Parameters

| Symbol | Meaning | Shape |
|:-------|:--------|:------|
| $W^{(l)}$ | Weight matrix of layer $l$ | $(n^{(l)}, n^{(l-1)})$ |
| $b^{(l)}$ | Bias vector of layer $l$ | $(n^{(l)}, 1)$ |
| $z^{(l)}$ | Pre-activation (linear output) | $(n^{(l)}, 1)$ |
| $a^{(l)}$ | Post-activation (after nonlinearity) | $(n^{(l)}, 1)$ |

### C-2-3 Learning

| Symbol | Meaning |
|:-------|:--------|
| $\eta$ | Learning rate |
| $\delta_j^{(l)}$ | Error of neuron $j$ in layer $l$ |
| $L$ or $\mathcal{L}$ | Loss/cost function |
| $\nabla L$ | Gradient of loss |
| $\odot$ | Element-wise multiplication |
| $\frac{\partial L}{\partial W}$ | Gradient of loss w.r.t. weights |
