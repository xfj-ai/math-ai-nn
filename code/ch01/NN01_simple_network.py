# 2-layer network forward pass
import numpy as np
def sigmoid(x): return 1/(1+np.exp(-x))

class TwoLayerNet:
    def __init__(self):
        np.random.seed(42)
        self.W1 = np.random.randn(2,4)*0.1
        self.b1 = np.zeros(4)
        self.W2 = np.random.randn(4,1)*0.1
        self.b2 = np.zeros(1)
    def forward(self, x):
        u1 = x @ self.W1 + self.b1
        z1 = sigmoid(u1)
        print("h1:", u1.round(4), z1.round(4))
        u2 = z1 @ self.W2 + self.b2
        y = sigmoid(u2)
        print("out:", u2.round(4), y.round(4))
        return y

if __name__ == "__main__":
    net = TwoLayerNet()
    y = net.forward(np.array([0.5, -0.3]))
    print("output:", y.item())

