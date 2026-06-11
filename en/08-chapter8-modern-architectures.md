# Chapter 8: Modern Architectures — From ResNet to Transformer

> **Goal**: Understand the key architectural innovations that enable modern deep learning — skip connections, attention mechanisms, and the Transformer.

> **Code**: `../code/ch08/` (5 files)

---

## 📋 Chapter Learning Objectives

- [ ] Understand ResNet and why skip connections solve vanishing gradients
- [ ] Understand RNN basics and the vanishing gradient problem in sequences
- [ ] Master the attention mechanism: Query, Key, Value
- [ ] Understand the complete Transformer architecture
- [ ] Be able to implement a minimal Transformer block

---

## 8-1 Residual Networks (ResNet) ⭐

### The Degradation Problem

Adding more layers should not hurt performance, but in practice it does — even on training data. This is the **degradation problem**.

### Skip Connections (Residual Learning)

Instead of learning $H(x)$, learn the residual $F(x) = H(x) - x$:

$$
H(x) = F(x) + x
$$

```text
    x
    │
    ├──→ [Weight] → [ReLU] → [Weight] → + → ReLU → output
    │                                        ↑
    └────────────────── shortcut ────────────┘
```

### PyTorch ResNet Block

```python
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3,
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3,
                              padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut: adjust dimensions if needed
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual  # Skip connection!
        return torch.relu(out)
```

### Why Skip Connections Work

| Benefit | Explanation |
|:--------|:------------|
| Gradient highway | Gradients can flow directly through skip connections |
| Identity mapping | Network can choose to ignore extra layers |
| Ensemble effect | Behaves like an ensemble of shallow networks |

---

## 8-2 Recurrent Neural Networks & Sequence Modeling

### The RNN Cell

$$
h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b_h)
$$

### Vanishing Gradient in RNNs

RNNs also suffer from vanishing gradients — gradients must flow through many time steps.

### LSTM & GRU

LSTM adds **gates** to control information flow:

- **Forget gate**: what to discard
- **Input gate**: what to store
- **Output gate**: what to output

---

## 8-3 Attention Mechanism ⭐

### The Core Idea

Attention allows the model to **focus on relevant parts** of the input.

### Query, Key, Value

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^{\mathsf{T}}}{\sqrt{d_k}}\right) V
$$

| Component | Analogy | Role |
|:----------|:--------|:-----|
| **Query** | What you're looking for | Current focus |
| **Key** | What's available | What each position offers |
| **Value** | The actual content | What to extract if matched |

### Scaled Dot-Product Attention

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

def scaled_dot_product_attention(Q, K, V):
    """Compute attention scores and weighted values"""
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)
    attention_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attention_weights, V)
    return output, attention_weights

# Example
batch, seq_len, d_k = 2, 4, 8
Q = torch.randn(batch, seq_len, d_k)
K = torch.randn(batch, seq_len, d_k)
V = torch.randn(batch, seq_len, d_k)

output, weights = scaled_dot_product_attention(Q, K, V)
print(f"Output shape: {output.shape}")     # (2, 4, 8)
print(f"Weights shape: {weights.shape}")    # (2, 4, 4)
```

### Multi-Head Attention

Instead of one attention, use **multiple heads** in parallel:

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, Q, K, V):
        batch_size = Q.size(0)

        # Linear projections + split into heads
        Q = self.W_q(Q).view(batch_size, -1, self.num_heads, self.d_head).transpose(1, 2)
        K = self.W_k(K).view(batch_size, -1, self.num_heads, self.d_head).transpose(1, 2)
        V = self.W_v(V).view(batch_size, -1, self.num_heads, self.d_head).transpose(1, 2)

        # Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_head ** 0.5)
        attn = F.softmax(scores, dim=-1)
        context = torch.matmul(attn, V)

        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, self.num_heads * self.d_head)
        return self.W_o(context)
```

---

## 8-4 Complete Transformer Architecture ⭐

### Encoder Block

```python
class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Multi-head attention + residual + layernorm
        attn_out = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))

        # FFN + residual + layernorm
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x
```

### Overall Architecture

```text
Output ← Linear ← LayerNorm ← Decoder × N ← Encoder × N
                                  ↑              ↑
                              (shifted)      Positional
                              output          Encoding
                              tokens            ↓
                                            Input tokens
```

### Positional Encoding

Since attention has no inherent notion of order, we add positional information:

$$
PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d_{\text{model}}})
$$
$$
PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d_{\text{model}}})
$$

---

## 8-5 Transformer Visualization

### Attention Patterns

Attention weights reveal what the model focuses on. For example, in a translation task:

```text
"I love neural networks"
    ↓    ↓       ↓
"J'aime les réseaux de neurones"
```

The attention matrix would show which English words align with which French words.

---

## 8-6 From RNN to LSTM/GRU

| Architecture | Key Innovation | When to Use |
|:-------------|:---------------|:------------|
| RNN | Sequential processing | Simple sequences |
| LSTM | Gating mechanism | Long sequences |
| GRU | Simplified LSTM | Medium sequences |
| Transformer | Full attention | Most tasks (modern default) |

---

## 8-7 BERT vs. GPT: Pre-training Paradigms

| Aspect | BERT | GPT |
|:-------|:-----|:----|
| Architecture | Encoder-only | Decoder-only |
| Training | Masked LM (bidirectional) | Autoregressive (left-to-right) |
| Best for | Understanding (classification, QA) | Generation (text, code) |
| Examples | BERT, RoBERTa, ALBERT | GPT-2, GPT-3, GPT-4 |

---

## 8-8 Modern Architecture Design Patterns

| Pattern | Example | Benefit |
|:--------|:--------|:--------|
| Skip connections | ResNet | Gradient flow |
| LayerNorm | Transformer | Stable training |
| Pre-norm | GPT | Better for deep models |
| GELU activation | GPT/BERT | Smooth ReLU variant |

---

## 8-9 Vision Transformer (ViT)

ViT applies the Transformer directly to image patches:

```python
class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_ch=3, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # (B, embed_dim, H/p, W/p)
        x = x.flatten(2)  # (B, embed_dim, num_patches)
        return x.transpose(1, 2)  # (B, num_patches, embed_dim)
```

---

## 8-10 Modern Architecture Design Principles

1. **Scale**: More data + more parameters + more compute
2. **Normalization**: Every block should normalize
3. **Residual connections**: Essential for deep models
4. **Attention is universal**: Works for text, images, audio, video

---

## 📖 Chapter Summary

### Architecture Evolution

```text
FC → CNN → ResNet → RNN/LSTM → Attention → Transformer → ViT → GPT
```

### 🧪 Exercises

#### Exercise 1: Implement a ResNet Block
Build a ResidualBlock, verify gradient flow by comparing with a plain block.

#### Exercise 2: Attention Visualization
Implement attention and visualize the weight matrix. Which tokens attend to which?

#### Exercise 3: Minimal Transformer
Build a 2-layer Transformer encoder and train it on a simple task.

#### Exercise 4: BERT vs. GPT
Implement both masked LM and autoregressive LM objectives. Compare their behavior.
