# Tensor basics
import torch

# Creation
a = torch.tensor([[1,2],[3,4]])
b = torch.zeros(2,3)
c = torch.randn(2,3)
print(f"From list:\n{a}")
print(f"Zeros:\n{b}")

# Properties
print(f"Shape: {a.shape}, dtype: {a.dtype}, device: {a.device}")

# Type conversion
f = a.float()
print(f"Float:\n{f}")

# To numpy
import numpy as np
n = a.numpy()
print(f"Back to numpy:\n{n}")

# GPU check
if torch.cuda.is_available():
    g = a.cuda()
    print(f"On GPU: {g.device}")

