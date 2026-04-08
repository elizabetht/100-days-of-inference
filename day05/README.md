# Day 05

**Topic:** KV Cache — The Most Important Optimization in Inference
**Date:** 2026-04-08
**Layer:** Runtime

## What I explored

Autoregressive generation without a KV cache is O(n²): at decode step N, the model re-runs attention over the full N-token sequence from scratch. Every past token's K and V projections get recomputed even though those tokens haven't changed.

The fix is mechanical: cache K and V for all past tokens. On each new decode step, compute Q/K/V for the single new token only, concatenate the new K/V with the cached K/V, then run attention. The query still attends over the full history — we just don't recompute K and V for positions that are already done.

Built this from scratch using GPT-2 weights (pure numpy, no torch):
- `prefill()` — processes the full prompt in one pass and populates the KVCache
- `KVCache` class — per-layer K/V store with append and retrieval
- `decode_step()` — single-token forward pass that reads from and updates the cache
- Verified that cached and no-cache generation produce identical output

## Key insight

KV cache is the canonical time-space tradeoff in inference. It converts O(n²) compute into O(n) memory growth. The memory formula is `2 × n_layers × n_embd × seq_len × bytes_per_element`. For a 70B model at 32K context in float16, that's ~160 GB per request — more than a single H100's memory. This is why every production serving system (vLLM, TGI, TensorRT-LLM) is, in significant part, a KV cache management system. PagedAttention, prefix caching, and KV quantization all exist to manage this one data structure.

## Code / experiment

Notebook: [`kv-cache.ipynb`](./kv-cache.ipynb)

Key demo: benchmark comparing per-token decode latency with vs without KV cache across prompt lengths 10–200 tokens. The no-cache latency grows linearly with sequence length; the cached decode step stays near-constant (only one new token's projections are computed). Speedup grows with context length. The notebook also computes the exact KV cache memory footprint for models from GPT-2 to GPT-3 scale at various sequence lengths, showing why memory — not compute — is the binding constraint for long-context serving.

## References

- *Inference Engineering* Ch 5.3 (pp. 136-141) — Philip Kiely, Baseten Books 2026
- [PagedAttention](https://arxiv.org/abs/2309.06180) — Kwon et al., SOSP 2023
- [FlashAttention-2](https://arxiv.org/abs/2307.08691) — Dao 2023
