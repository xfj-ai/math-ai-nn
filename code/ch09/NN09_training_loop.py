import torch
import torch.nn as nn
import torch.optim as optim

# GPT-like model training loop
class TinyGPT(nn.Module):
    def __init__(self, vocab_size=50):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, 32)
        self.pos = nn.Embedding(20, 32)
        self.attn = nn.MultiheadAttention(32, 4, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 32))
        self.norm1 = nn.LayerNorm(32)
        self.norm2 = nn.LayerNorm(32)
        self.head = nn.Linear(32, vocab_size)

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        h = self.embed(x) + self.pos(pos)
        causal_mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        h2, _ = self.attn(h, h, h, attn_mask=causal_mask)
        h = self.norm1(h + h2)
        h2 = self.ffn(h)
        h = self.norm2(h + h2)
        return self.head(h)

model = TinyGPT()
optimizer = optim.AdamW(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Dummy training data
data = torch.randint(0, 50, (32, 15))
labels = torch.randint(0, 50, (32, 15))

print("Training TinyGPT (5 epochs)...")
for epoch in range(5):
    logits = model(data)
    loss = criterion(logits.reshape(-1, 50), labels.reshape(-1))
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    print(f"  Epoch {epoch}: loss={loss.item():.4f}")

print("TinyGPT training complete!")
