import torch, torch.nn as nn

x = torch.randn(1, 1, 8, 8)
max_pool = nn.MaxPool2d(2)
avg_pool = nn.AvgPool2d(2)
print(f"MaxPool(2x2): {max_pool(x).shape}")
print(f"AvgPool(2x2): {avg_pool(x).shape}")

conv_s1 = nn.Conv2d(1, 1, 3, stride=1, padding=1)
conv_s2 = nn.Conv2d(1, 1, 3, stride=2, padding=1)
print(f"Conv stride=1: {conv_s1(x).shape}")
print(f"Conv stride=2: {conv_s2(x).shape}")

print("Receptive field grows with layers:")
print("  1 conv(3x3) -> 3x3 field")
print("  2 convs(3x3) -> 5x5 field")
print("  3 convs(3x3) -> 7x7 field")
