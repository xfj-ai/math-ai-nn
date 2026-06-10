# Training loop template
import torch
import torch.nn as nn
import torch.optim as optim

model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 10),
)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Dummy data
X = torch.randn(256, 784)
y = torch.randint(0, 10, (256,))

print("Training loop:")
for epoch in range(5):
    # Forward
    outputs = model(X)
    loss = criterion(outputs, y)

    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    _, predicted = torch.max(outputs, 1)
    acc = (predicted == y).float().mean()
    print(f"  Epoch {epoch}: loss={loss.item():.4f}, acc={acc:.2%}")

print("\nThe 3-step pattern: zero_grad() -> backward() -> step()")

