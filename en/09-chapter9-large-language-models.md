# Chapter 9: Large Language Models — Training, Sampling & Inference

> **Goal**: Understand how large language models (LLMs) are trained, how they generate text, and the techniques that make them practical.

> **Code**: `../code/ch09/` (4 files)

---

## 📋 Chapter Learning Objectives

- [ ] Understand the language modeling objective (next-token prediction)
- [ ] Master autoregressive generation
- [ ] Understand sampling strategies (greedy, top-k, top-p)
- [ ] Understand KV Cache for inference acceleration
- [ ] Master LoRA for efficient fine-tuning
- [ ] Understand quantization fundamentals

---

## 9-1 From Language Model to Large Language Model

### What Is a Language Model?

A language model assigns a probability to a sequence of tokens:

$$
P(w_1, w_2, \dots, w_n) = \prod_{t=1}^{n} P(w_t | w_{<t})
$$

### Scaling Hypothesis

As models get larger (more parameters, more data, more compute), capabilities **emerge**:

```text
Model Size:     100M → 1B → 10B → 100B → 1T
Capabilities:   Grammar → Facts → Reasoning → Expertise
```

### Key LLMs

| Model | Parameters | Year | Innovation |
|:------|:-----------|:-----|:-----------|
| GPT-2 | 1.5B | 2019 | Zero-shot learning |
| GPT-3 | 175B | 2020 | In-context learning |
| LLaMA | 65B | 2023 | Open-source, efficient |
| GPT-4 | ~1.8T | 2023 | Multi-modal, reasoning |

---

## 9-2 Autoregressive Generation & Training

### Training Objective

Maximize the likelihood of the next token given previous tokens:

$$
L = -\sum_{t=1}^{T} \log P(w_t | w_{<t}; \theta)
$$

### Generation Loop

```python
def generate_autoregressive(model, prompt, max_tokens=100):
    """Autoregressive text generation"""
    tokens = tokenize(prompt)

    for _ in range(max_tokens):
        # Forward pass
        logits = model(tokens)

        # Get prediction for the next token
        next_token_logits = logits[:, -1, :]

        # Sample
        next_token = sample_from_logits(next_token_logits)

        # Append to sequence
        tokens = torch.cat([tokens, next_token], dim=1)

        # Stop if EOS token
        if next_token.item() == EOS_TOKEN_ID:
            break

    return detokenize(tokens)
```

---

## 9-3 Sampling Strategies ⭐

### Greedy Decoding

Always pick the most likely token:

$$w_t = \arg\max P(w | w_{<t})$$

✅ Simple, deterministic ❌ Repetitive, boring

### Temperature Sampling

Controls randomness:

$$P(w) \propto \exp(\text{logit}_w / T)$$

| Temperature | Effect |
|:------------|:-------|
| $T \to 0$ | Greedy (deterministic) |
| $T = 1$ | Standard softmax |
| $T > 1$ | More random, creative |

### Top-k Sampling

Only sample from the $k$ most likely tokens:

```python
def top_k_sampling(logits, k=50):
    """Sample from top-k tokens only"""
    values, indices = torch.topk(logits, k)
    probs = F.softmax(values / temperature, dim=-1)
    chosen = torch.multinomial(probs, 1)
    return indices[0, chosen]
```

### Top-p (Nucleus) Sampling

Sample from the smallest set of tokens whose cumulative probability exceeds $p$:

```python
def top_p_sampling(logits, p=0.9):
    """Nucleus sampling"""
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    sorted_probs = F.softmax(sorted_logits / temperature, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Remove tokens with cumulative probability above p
    sorted_indices_to_remove = cumulative_probs > p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    indices_to_remove = sorted_indices_to_remove.scatter(
        1, sorted_indices, sorted_indices_to_remove)
    logits[indices_to_remove] = float('-inf')
    probs = F.softmax(logits / temperature, dim=-1)
    return torch.multinomial(probs, 1)
```

### Sampling Comparison

| Strategy | Diversity | Quality | Use Case |
|:---------|:----------|:--------|:---------|
| Greedy | Low | High (first try) | Factual answers |
| Temperature | Medium | Medium | Creative writing |
| Top-k (k=50) | Medium | High | General purpose |
| Top-p (p=0.9) | High | High | Balanced |
| Top-k + Top-p | High | Highest | Production |

---

## 9-4 KV Cache: Inference Acceleration ⭐

### The Problem

At each generation step, the model re-computes attention for **all** previous tokens — $O(n^2)$ complexity.

### The Solution: KV Cache

Cache the Key and Value matrices from previous steps:

```python
class CachedAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

    def forward(self, x, kv_cache=None):
        batch, seq_len = x.shape[:2]

        Q = self.W_q(x).view(batch, seq_len, self.num_heads, self.d_head)
        K = self.W_k(x).view(batch, seq_len, self.num_heads, self.d_head)
        V = self.W_v(x).view(batch, seq_len, self.num_heads, self.d_head)

        # Concatenate with cache
        if kv_cache is not None:
            K_cache, V_cache = kv_cache
            K = torch.cat([K_cache, K], dim=1)
            V = torch.cat([V_cache, V], dim=1)

        # Update cache
        new_kv_cache = (K, V)

        # Attention (only need last query for generation)
        Q_last = Q[:, -1:] if kv_cache is not None else Q
        scores = torch.matmul(Q_last, K.transpose(-2, -1)) / (self.d_head ** 0.5)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)
        return out, new_kv_cache
```

### Speedup

| Without KV Cache | With KV Cache |
|:-----------------|:--------------|
| $O(n^2)$ per step | $O(n)$ per step |
| ~3× slower for 100 tokens | Baseline |
| ~10× slower for 1000 tokens | Baseline |

---

## 9-5 Efficient Fine-Tuning: LoRA

### The Problem

Full fine-tuning of a 175B model is **prohibitively expensive** ($> $1M per run).

### LoRA: Low-Rank Adaptation

LoRA freezes the original weights and adds **small rank decomposition matrices**:

$$
W' = W + BA
$$

Where $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times k}$, and $r \ll \min(d, k)$.

```python
class LoRALayer(nn.Module):
    """Low-Rank Adaptation layer"""
    def __init__(self, original_layer, rank=8, alpha=16):
        super().__init__()
        self.original = original_layer
        self.original.requires_grad_(False)  # freeze

        d, k = original_layer.weight.shape
        self.A = nn.Parameter(torch.randn(rank, k) / rank)
        self.B = nn.Parameter(torch.zeros(d, rank))
        self.scale = alpha / rank

    def forward(self, x):
        # Original (frozen) + LoRA (trainable)
        return self.original(x) + (x @ self.A.T @ self.B.T) * self.scale
```

### Why LoRA Works

- Pre-trained models have **low intrinsic rank**
- A small number of parameters can capture task-specific adaptations
- Can swap LoRA modules for different tasks without loading the base model

### Memory Comparison

| Method | Trainable Params | Memory |
|:-------|:-----------------|:-------|
| Full fine-tune | 175B | > 350GB |
| LoRA (r=8) | ~0.3B | < 1GB |

---

## 9-6 Quantization Basics

### Why Quantize?

Reducing precision from FP32 → INT8:
- **4× smaller memory**
- **2-4× faster inference**
- Minimal accuracy loss

### Quantization Types

| Type | Description | Accuracy Loss |
|:-----|:------------|:--------------|
| **Post-training** (PTQ) | Quantize after training | Small |
| **Quantization-aware** (QAT) | Train with simulated quant | Minimal |
| **GPTQ** | Weight-only, one-shot | Very small |
| **GGML/GGUF** | CPU-optimized | Small |

### INT8 Quantization

```python
def quantize_int8(tensor):
    """Quantize FP32 tensor to INT8"""
    scale = tensor.abs().max() / 127.0
    quantized = (tensor / scale).round().char()
    return quantized, scale

def dequantize(quantized, scale):
    """Restore from INT8 to FP32"""
    return quantized.float() * scale
```

---

## 9-7 RLHF: Reinforcement Learning from Human Feedback

### The RLHF Pipeline

```text
1. Supervised fine-tuning (SFT) on human demonstrations
2. Train a reward model on human preferences
3. Optimize the policy (LLM) using PPO against the reward model
```

### Why RLHF?

- Language modeling objective ($P(w_t | w_{<t})$) ≠ helpful/honest/harmless
- Human feedback directly optimizes for what we want

---

## 9-8 RAG: Retrieval-Augmented Generation

### The Problem

LLMs have **knowledge cutoffs** and can **hallucinate** facts.

### Solution

RAG retrieves relevant documents from a knowledge base before generating:

```text
Query → Retrieve relevant docs → Augment prompt → Generate answer
```

### Benefits

| Benefit | Description |
|:--------|:------------|
| Up-to-date knowledge | No retraining needed |
| Verifiable | Can check source documents |
| Reduces hallucinations | Grounded in retrieved text |

---

## 9-9 Practical Model Quantization

```python
# Using bitsandbytes for quantization
import torch
import transformers

model = transformers.AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b",
    load_in_4bit=True,  # 4-bit quantization
    bnb_4bit_compute_dtype=torch.float16,
)

# Memory: ~4GB instead of ~14GB for FP16
```

---

## 9-10 Prompt Engineering

### Key Techniques

| Technique | Description | Example |
|:----------|:------------|:--------|
| **Zero-shot** | No examples | "Translate to French: Hello" |
| **Few-shot** | A few examples | "English: Hello → French: Bonjour" |
| **Chain-of-thought** | Step-by-step reasoning | "Let's think step by step..." |
| **System prompt** | Role setting | "You are a helpful assistant" |

---

## 9-11 Evaluating LLMs

### Metrics

| Metric | What It Measures |
|:-------|:-----------------|
| **Perplexity** | How well the model predicts (lower is better) |
| **BLEU** | N-gram overlap with reference |
| **ROUGE** | Recall-oriented overlap |
| **Human eval** | Helpfulness, harmlessness |

---

## 📖 Chapter Summary

### LLM Technology Stack

```text
Pre-training (next-token prediction)
    ↓
Fine-tuning (SFT, RLHF)
    ↓
Inference (autoregressive + sampling)
    ↓
Optimization (KV Cache, quantization, LoRA)
    ↓
Augmentation (RAG, prompting)
```

### 🧪 Exercises

#### Exercise 1: Implement Autoregressive Generation
Write a simple autoregressive generation loop with greedy decoding.

#### Exercise 2: Compare Sampling Strategies
Generate text with greedy, top-k (k=10), and top-p (p=0.9). Compare diversity.

#### Exercise 3: Implement LoRA
Add LoRA to a linear layer and verify only the LoRA parameters are trainable.

#### Exercise 4: KV Cache
Implement generation with and without KV cache. Measure the speedup.
