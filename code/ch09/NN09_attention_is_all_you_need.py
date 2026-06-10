import torch
import torch.nn as nn

# Simplified Transformer (decoder-only)
class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_qkv = nn.Linear(d_model, 3*d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, T, D = x.shape
        qkv = self.W_qkv(x).reshape(B, T, 3, self.num_heads, self.d_k)
        q, k, v = qkv[:,:,0], qkv[:,:,1], qkv[:,:,2]
        q, k, v = q.transpose(1,2), k.transpose(1,2), v.transpose(1,2)

        scores = q @ k.transpose(-2, -1) / (self.d_k**0.5)
        causal_mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        scores.masked_fill_(causal_mask, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        out = attn @ v
        out = out.transpose(1,2).reshape(B, T, D)
        return self.W_o(out)

class DecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.attn = CausalSelfAttention(d_model, num_heads)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.norm1(x + self.attn(x))
        x = self.norm2(x + self.ffn(x))
        return x

class DecoderOnlyTransformer(nn.Module):
    def __init__(self, vocab_size=100, d_model=32, num_heads=4, d_ff=64, num_layers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(100, d_model)
        self.blocks = nn.ModuleList([DecoderBlock(d_model, num_heads, d_ff) for _ in range(num_layers)])
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        x = self.embed(x) + self.pos(pos)
        for block in self.blocks:
            x = block(x)
        return self.head(x)

model = DecoderOnlyTransformer()
x = torch.randint(0, 100, (2, 20))
logits = model(x)
params = sum(p.numel() for p in model.parameters())
print(f"GPT-like DecoderOnlyTransformer: {params:,} params")
print(f"Input: {x.shape} -> Output: {logits.shape}")
