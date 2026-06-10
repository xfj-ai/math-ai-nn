# Layer-wise backprop
import numpy as np

def sig(z): return 1/(1+np.exp(-z))
def sig_deriv(y): return y*(1-y)

class Linear:
    def __init__(self, w, b):
        self.w, self.b = w, b
    def forward(self, x):
        self.x = x
        self.u = x @ self.w + self.b
        return self.u
    def backward(self, du):
        self.dw = self.x.reshape(-1,1) @ du.reshape(1,-1)
        self.db = du.copy()
        return du @ self.w.T

class Sigmoid:
    def forward(self, u):
        self.y = sig(u)
        return self.y
    def backward(self, dy):
        return dy * sig_deriv(self.y)

class MSELoss:
    def forward(self, y, t):
        self.y, self.t = y, t
        self.L = 0.5*(y-t)**2
        return self.L
    def backward(self):
        return self.y - self.t

# Build net: linear1 -> sigmoid -> linear2 -> sigmoid -> loss
net = [
    Linear(np.array([[0.1,0.3],[0.2,0.4]]), np.array([0.1,0.2])),
    Sigmoid(),
    Linear(np.array([[0.5],[0.6]]), np.array([0.3])),
    Sigmoid(),
]
loss_fn = MSELoss()

x = np.array([0.5, -0.3])
t = np.array([0.8])

# Forward
h = x
for layer in net:
    h = layer.forward(h)
L = loss_fn.forward(h, t)
print(f"Loss: {L.item():.4f}")

# Backward
d = loss_fn.backward()
for layer in reversed(net):
    d = layer.backward(d)

print(f"\ndW1:\n{net[0].dw.round(4)}")
print(f"dW2:\n{net[2].dw.round(4)}")

