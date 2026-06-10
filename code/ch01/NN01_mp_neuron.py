# M-P Neuron test
import numpy as np

class MPNeuron:
    def __init__(self, weights, threshold=0):
        self.weights = weights
        self.threshold = threshold
    def forward(self, x):
        z = np.dot(self.weights, x)
        return 1 if z >= self.threshold else 0

def test():
    n = MPNeuron(np.array([0.5,0.5]), 1.0)
    print(n.forward(np.array([1,1])))

if __name__ == "__main__":
    test()

