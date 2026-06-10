# Autograd demo
import torch

# Manual forward + Auto backward
x = torch.tensor([2.0], requires_grad=True)
w = torch.tensor([0.5], requires_grad=True)
b = torch.tensor([0.1], requires_grad=True)
t = torch.tensor([1.0])

u = w * x + b
y = torch.sigmoid(u)
L = 0.5 * (y - t)**2

L.backward()
print(f"dL/dw = {w.grad.item():.4f}")
print(f"dL/db = {b.grad.item():.4f}")
print(f"dL/dx = {x.grad.item():.4f}")

# Verify with manual chain rule
def sig(x): return 1/(1+torch.exp(-x))
u_val = 0.5*2.0 + 0.1
y_val = sig(torch.tensor(u_val))
dL_dy = y_val - 1.0
dy_du = y_val * (1 - y_val)
du_dw = 2.0
dL_dw_manual = dL_dy * dy_du * du_dw
print(f"Manual check dL/dw = {dL_dw_manual.item():.4f}")

