# Adam optimizer
import numpy as np, matplotlib.pyplot as plt

def f(x): return x[0]**2 + 5*x[1]**2
def grad(x): return np.array([2*x[0], 10*x[1]])

x0 = np.array([1.5, 1.0])
x = x0.copy()
lr = 0.1; beta1, beta2 = 0.9, 0.999
m, v = np.zeros(2), np.zeros(2)
eps = 1e-8; epochs = 50

loss_adam = [f(x)]
for t in range(1, epochs+1):
    g = grad(x)
    m = beta1*m + (1-beta1)*g
    v = beta2*v + (1-beta2)*g**2
    m_hat = m/(1-beta1**t)
    v_hat = v/(1-beta2**t)
    x -= lr * m_hat / (np.sqrt(v_hat) + eps)
    loss_adam.append(f(x))

plt.plot(loss_adam, label="Adam")
plt.yscale("log"); plt.legend(); plt.grid()
plt.title("Adam Convergence")
plt.xlabel("Step"); plt.ylabel("Loss")
plt.savefig("images/ch04/NN04_adam_convergence.png", dpi=150)
print(f"Adam final trajectory: x={x.round(4)}")

