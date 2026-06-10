# Gradient checking
import torch
import numpy as np

def grad_check(model, x, t, eps=1e-5):
    losses_at = lambda theta: model(*theta) if hasattr(model, __call__) else None

    params = list(model.parameters())
    for i, p in enumerate(params):
        p_flat = p.data.flatten()
        for j in range(min(3, len(p_flat))):
            # Compute numerical gradient
            p.data.flatten()[j] += eps
            L_plus = compute_loss(model, x, t)
            p.data.flatten()[j] -= 2*eps
            L_minus = compute_loss(model, x, t)
            p.data.flatten()[j] += eps
            num_grad = (L_plus - L_minus)/(2*eps)
            print(f"  Param[{i}][{j}]: num={num_grad:.6f}")

def compute_loss(model, x, t):
    y = model(x)
    return 0.5*(y-t)**2

# Simple test
w = torch.tensor([[0.5]], requires_grad=True)
b = torch.tensor([0.1], requires_grad=True)
x = torch.tensor([[2.0]])
t = torch.tensor([1.0])

# Forward
u = x @ w + b
y = torch.sigmoid(u)
L = 0.5*(y-t)**2
L.backward()

# Manual check
w_val, b_val = 0.5, 0.1
eps = 1e-5
L_plus = 0.5*(torch.sigmoid(torch.tensor(x.item()*(w_val+eps)+b_val))-t)**2
L_minus = 0.5*(torch.sigmoid(torch.tensor(x.item()*(w_val-eps)+b_val))-t)**2
num_grad = (L_plus-L_minus)/(2*eps)
print(f"Auto grad w: {w.grad.item():.6f}")
print(f"Numerical w: {num_grad.item():.6f}")
print(f"Diff: {abs(w.grad.item()-num_grad.item()):.2e}")

