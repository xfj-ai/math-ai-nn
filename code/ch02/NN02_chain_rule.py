# Chain rule demo
import numpy as np

# Simple chain: z = (x+1)^2
x = 2.0
# dz/dx = 2*(x+1)
print(f"Manual chain: dz/dx = {2*(x+1):.4f}")

# Multi-variable chain
x, y = 2.0, 3.0
# u = x + y, v = u^3
# dv/dx = dv/du * du/dx = 3u^2 * 1
u = x + y
dv_du = 3*u**2
dv_dx = dv_du * 1
print(f"Multi-var chain: dv/dx = {dv_dx:.4f}")

# Neural network chain (one neuron)
w, b, x_in = 0.5, 0.1, 2.0
# u = wx+b, y = sigmoid(u), L = 0.5*(y-t)^2
def sig(x): return 1/(1+np.exp(-x))
t = 1.0
u = w*x_in + b
y = sig(u)
L = 0.5*(y-t)**2
# dL/dw = (y-t) * y*(1-y) * x
dL_dw = (y-t) * y*(1-y) * x_in
print(f"NN chain: dL/dw = {dL_dw:.4f}")

