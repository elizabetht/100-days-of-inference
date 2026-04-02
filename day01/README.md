# Day 01

**Topic:** LLM Inference Mechanics: Tokenization, KV Cache, Autoregressive Decoding

**Date:** 2026-03-31

**Layer:** Runtime

## What I explored

How LLMs actually generate text: subword tokenization maps strings to integer token IDs without any neural network, the prefill phase processes the full prompt in parallel to populate the KV cache, and the decode phase generates one token per forward pass by sampling from a softmax distribution over the vocabulary. The KV cache is the key optimization that keeps decode attention linear in sequence length rather than quadratic.

## Key insight

The KV cache converts attention from O(n^2) to O(n) per decode step — without it, every token generation would require recomputing attention over the entire prior sequence from scratch. This single data structure is why long-context inference is possible at all, and why KV cache memory management (PagedAttention, prefix caching) is a central topic in inference engineering.

## Code / experiment

Notebook: [`llm-inference-mechanics.ipynb`](./llm-inference-mechanics.ipynb)

Key demos:
- Step-by-step attention walkthrough: query scores each key, softmax weights, weighted sum of values
- With-cache vs without-cache ops comparison showing where the savings come from
- sqrt(d_k) scaling visualization: why unscaled attention collapses to one-hot
- KV cache memory calculator for real model configs (Llama-3-8B/70B, Qwen2.5-7B)
- Full sampling pipeline: temperature + top-k + top-p with side-by-side distribution plots
- Autoregressive decode loop with KV cache growth tracking
- Attention time vs context length benchmark (confirms linear scaling with KV cache)

## References

- *Inference Engineering* Ch 2.2 (pp. 46-54) - Philip Kiely, Baseten Books 2026
- Vaswani et al. (2017), "Attention Is All You Need"
- Dao et al. (2022), "FlashAttention"
