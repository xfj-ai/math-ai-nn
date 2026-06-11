# Appendix E: Exercise Reference Answers

> This appendix provides reference answers and explanations for chapter exercises.

---

## Chapter 1: The Idea of Neural Networks

### Exercise 1: M-P Neuron for NAND Gate

Set threshold = 2, weights = [-1, -1]:

```python
def mp_neuron(x1, x2, w1=-1, w2=-1, threshold=-1.5):
    """NAND gate: negated AND"""
    total = w1 * x1 + w2 * x2
    return 1 if total >= threshold else 0

# Test
for x1, x2 in [(0,0), (0,1), (1,0), (1,1)]:
    print(f"NAND({x1},{x2}) = {mp_neuron(x1, x2)}")
# Output: 1, 1, 1, 0
```

### Exercise 4: Parameter Count

- Layer 1: 784 × 256 + 256 = 200,960
- Layer 2: 256 × 10 + 10 = 2,570
- Total: 203,530

If hidden = 512: 784 × 512 + 512 = 401,920; 512 × 10 + 10 = 5,130; Total = 407,050 (2×)

---

## Chapter 2: Mathematical Foundations

### Exercise 1: Dot Product

$x \cdot y = 1 \times 4 + 2 \times 5 + 3 \times 6 = 32$

### Exercise 3: Numerical Derivative

$f(x) = x^3$, $f'(x) = 3x^2$, $f'(2) = 12$

```python
def numerical_derivative(f, x, h=1e-5):
    return (f(x+h) - f(x-h)) / (2*h)
print(numerical_derivative(lambda x: x**3, 2))  # ~12.0
```

---

## Chapter 3: PyTorch Basics

### Exercise 2: Autograd

```python
x = torch.tensor([2.0], requires_grad=True)
y = x**2 + 3*x + 1
y.backward()
print(x.grad)  # 2*2 + 3 = 7
```
