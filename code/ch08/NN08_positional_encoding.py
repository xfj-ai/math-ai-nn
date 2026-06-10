import torch
import numpy as np
import matplotlib.pyplot as plt

def positional_encoding(seq_len, d_model):
    pe = torch.zeros(seq_len, d_model)
    pos = torch.arange(seq_len).unsqueeze(1).float()
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0)/d_model))
    pe[:, 0::2] = torch.sin(pos * div_term)
    pe[:, 1::2] = torch.cos(pos * div_term)
    return pe

pe = positional_encoding(100, 32)
plt.figure(figsize=(12, 4))
plt.subplot(121)
plt.imshow(pe.T, cmap="viridis", aspect="auto")
plt.xlabel("Position"); plt.ylabel("Dimension")
plt.title("Positional Encoding Matrix")
plt.colorbar()

plt.subplot(122)
for dim in [0, 1, 2, 3]:
    plt.plot(pe[:, dim].numpy(), label=f"dim {dim}")
plt.legend(); plt.title("Encodings for first 4 dims")
plt.xlabel("Position")
plt.tight_layout()
plt.savefig("images/ch08/NN08_positional_encoding.png", dpi=150)
print("Positional encoding saved")
