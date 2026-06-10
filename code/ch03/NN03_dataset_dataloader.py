# Dataset and DataLoader
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Custom dataset
class MyDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

X = np.random.randn(1000, 784)
y = np.random.randint(0, 10, 1000)
dataset = MyDataset(X, y)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

for batch_x, batch_y in loader:
    print(f"Batch: X={batch_x.shape}, y={batch_y.shape}")
    break

# MNIST
from torchvision import datasets, transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
mnist = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
mnist_loader = DataLoader(mnist, batch_size=64, shuffle=True)
print(f"MNIST: {len(mnist)} samples, batches={len(mnist_loader)}")

