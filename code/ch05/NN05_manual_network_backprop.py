# Full network backprop (manual)
import numpy as np

def sig(z): return 1/(1+np.exp(-z))
def sig_deriv(y): return y*(1-y)

# 2-2-1 network
class Net:
    def __init__(self):
        self.W1 = np.array([[0.1, 0.3], [0.2, 0.4]])
        self.b1 = np.array([0.1, 0.2])
        self.W2 = np.array([[0.5], [0.6]])
        self.b2 = np.array([0.3])

    def forward(self, x):
        self.x = x
        self.u1 = x @ self.W1 + self.b1
        self.z1 = sig(self.u1)
        self.u2 = self.z1 @ self.W2 + self.b2
        self.y = sig(self.u2)
        return self.y

    def backward(self, t):
        # Output layer
        dL_dy = self.y - t
        dy_du2 = sig_deriv(self.y)
        dL_du2 = dL_dy * dy_du2  # (1,1)
        dL_dW2 = self.z1.reshape(-1,1) @ dL_du2.reshape(1,-1)
        dL_db2 = dL_du2.copy()
        dL_dz1 = dL_du2 * self.W2.squeeze()

        # Hidden layer
        dz1_du1 = sig_deriv(self.z1)
        dL_du1 = dL_dz1 * dz1_du1
        dL_dW1 = self.x.reshape(-1,1) @ dL_du1.reshape(1,-1)
        dL_db1 = dL_du1.copy()

        return dL_dW1, dL_db1, dL_dW2, dL_db2

net = Net()
x = np.array([0.5, -0.3])
t = np.array([0.8])
y = net.forward(x)
L = 0.5*(y-t)**2
dW1, db1, dW2, db2 = net.backward(t)
print(f"Forward: y={y.item():.4f}, L={L.item():.4f}")
print(f"dW1:\n{dW1.round(4)}\ndb1: {db1.round(4)}")
print(f"dW2:\n{dW2.round(4)}\ndb2: {db2.round(4)}")

