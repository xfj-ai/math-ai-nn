# Manual backprop (single neuron)
import numpy as np

# Forward: x -> u=wx+b -> y=sigma(u) -> L=0.5*(y-t)^2
x, w, b, t = 2.0, 0.5, 0.1, 1.0
def sig(z): return 1/(1+np.exp(-z))

# Forward
u = w*x + b
y = sig(u)
L = 0.5*(y-t)**2
print(f"Forward: u={u:.4f}, y={y:.4f}, L={L:.4f}")

# Backward (manual chain rule)
dL_dy = y - t
dy_du = y*(1-y)
du_dw = x
dL_dw = dL_dy * dy_du * du_dw
dL_db = dL_dy * dy_du * 1
print(f"Backward: dL/dw={dL_dw:.4f}, dL/db={dL_db:.4f}")

# Numerical check
h = 1e-6
dL_dw_num = (0.5*(sig((w+h)*x+b)-t)**2 - 0.5*(sig((w-h)*x+b)-t)**2)/(2*h)
print(f"Numerical dL/dw={dL_dw_num:.4f}, diff={abs(dL_dw-dL_dw_num):.2e}")

