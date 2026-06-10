import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleLM(nn.Module):
    def __init__(self, vocab_size=50, d_model=32, num_layers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.rnn = nn.LSTM(d_model, d_model, num_layers, batch_first=True)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.embed(x)
        x, _ = self.rnn(x)
        return self.head(x)

model = SimpleLM()
x = torch.randint(0, 50, (2, 10))
logits = model(x)
print(f"Autoregressive LM: {x.shape} -> {logits.shape}")

# Greedy decoding
def generate(model, start_token=0, max_len=20):
    model.eval()
    x = torch.tensor([[start_token]])
    generated = [start_token]
    for _ in range(max_len):
        with torch.no_grad():
            logits = model(x)
            next_token = logits[0, -1].argmax().item()
            generated.append(next_token)
            x = torch.cat([x, torch.tensor([[next_token]])], dim=1)
    return generated

print(f"Generated: {generate(model)[:10]}...")
