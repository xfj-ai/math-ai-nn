# Autograd vs manual check
import torch
import numpy as np

# Manual backprop for 2-layer net
def sig(z): return 1/(1+np.exp(-z))
def sig_deriv(y): return y*(1-y)

x_np = np.array([0.5, -0.3])
t_np = np.array([0.8])
W1_np = np.array([[0.1, 0.3], [0.2, 0.4]])
b1_np = np.array([0.1, 0.2])
W2_np = np.array([[0.5], [0.6]])
b2_np = np.array([0.3])

# Manual
u1 = x_np @ W1_np + b1_np; z1 = sig(u1)
u2 = z1 @ W2_np + b2_np; y = sig(u2)
L = 0.5*(y-t_np)**2
dL_dy = y - t_np
dW2 = z1.reshape(-1,1) @ (dL_dy * sig_deriv(y)).reshape(1,-1)
dz1 = (dL_dy * sig_deriv(y)) * W2_np.squeeze()
dW1 = x_np.reshape(-1,1) @ (dz1 * sig_deriv(z1)).reshape(1,-1)

# Autograd
x_t = torch.tensor(x_np, requires_grad=True)
t_t = torch.tensor(t_np)
W1_t = torch.tensor(W1_np, requires_grad=True)
b1_t = torch.tensor(b1_np, requires_grad=True)
W2_t = torch.tensor(W2_np, requires_grad=True)
b2_t = torch.tensor(b2_np, requires_grad=True)

u1_t = x_t @ W1_t + b1_t
z1_t = torch.sigmoid(u1_t)
u2_t = z1_t @ W2_t + b2_t
y_t = torch.sigmoid(u2_t)
L_t = 0.5*(y_t - t_t)**2
L_t.backward()

print(f"Manual dW1:\n{dW1.round(decimals=6)}")
print(f"Auto  dW1:\n{W1_t.grad.round(decimals=6)}")
print(f"\nMatch: {np.allclose(dW1, W1_t.grad.numpy(), atol=1e-6)}")
print(f"\nManual dW2:\n{dW2.round(decimals=6)}")
print(f"Auto  dW2:\n{W2_t.grad.round(decimals=6)}")
print(f"Match: {np.allclose(dW2, W2_t.grad.numpy(), atol=1e-6)}")

