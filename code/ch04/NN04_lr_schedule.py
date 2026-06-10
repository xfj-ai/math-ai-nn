# LR scheduling
import numpy as np, matplotlib.pyplot as plt

epochs = 100
lrs_step = [0.1 * (0.5 ** (e//20)) for e in range(epochs)]
lrs_cos = [0.01 + 0.09*(1+np.cos(np.pi*e/epochs))/2 for e in range(epochs)]
lrs_exp = [0.1 * np.exp(-0.03*e) for e in range(epochs)]

plt.figure(figsize=(10, 4))
plt.plot(lrs_step, label="StepDecay (*0.5 every 20)")
plt.plot(lrs_cos, label="CosineAnnealing")
plt.plot(lrs_exp, label="ExponentialDecay")
plt.legend(); plt.xlabel("Epoch"); plt.ylabel("Learning Rate")
plt.title("Learning Rate Schedules")
plt.savefig("images/ch04/NN04_lr_schedule.png", dpi=150)

