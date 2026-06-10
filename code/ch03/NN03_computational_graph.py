# Computational graph
import torch

def trace_graph(x, w, b):
    u = w * x + b
    y = torch.sigmoid(u)
    L = 0.5 * (y - 1.0)**2

    print(f"L.grad_fn: {L.grad_fn}")
    print(f"y.grad_fn: {y.grad_fn}")
    print(f"u.grad_fn: {u.grad_fn}")

    L.backward()
    print(f"grad w: {w.grad.item():.4f}")
    print(f"grad b: {b.grad.item():.4f}")

x = torch.tensor([2.0], requires_grad=True)
w = torch.tensor([0.5], requires_grad=True)
b = torch.tensor([0.1], requires_grad=True)
trace_graph(x, w, b)

# Gradient accumulation demo
x = torch.tensor([2.0], requires_grad=True)
y = x**2
y.backward()
print(f"\n1st backward: x.grad = {x.grad}")

y2 = x**2
y2.backward()
print(f"2nd backward (accumulated): x.grad = {x.grad}")

x.grad.zero_()
y3 = x**2
y3.backward()
print(f"After zero: x.grad = {x.grad}")

