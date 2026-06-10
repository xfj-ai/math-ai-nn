# Broadcasting demo
import torch

# Basic broadcasting
a = torch.tensor([[1, 2, 3], [4, 5, 6]])  # (2,3)
b = torch.tensor([10, 20, 30])             # (3,) -> broadcasts to (2,3)
print(f"a + b:\n{a + b}")

# Column broadcast
c = torch.tensor([[10], [20]])             # (2,1) -> broadcasts to (2,3)
print(f"a + c:\n{a + c}")

# Without broadcasting (loop)
result = torch.zeros_like(a)
for i in range(2):
    for j in range(3):
        result[i,j] = a[i,j] + b[j]
print(f"Loop version:\n{result}")

print("\nBroadcasting rules: trailing dims match, 1 expands to fit")

