# Day 05 — KV Cache

Autoregressive generation has a hidden cost: at each decode step, the model re-runs attention over the entire sequence from scratch. Token 0's Key and Value vectors at step 5 are identical to step 500 — but they get recomputed every time. Total work across N steps: 1 + 2 + 3 + ... + N = O(N²).

The fix is the KV cache. Cache every past token's K and V after computing them once. On each new step, compute Q/K/V for the single new token only, concatenate with the cache, and run attention. The query still sees the full history — we just don't redo work for positions that are already done. The optimization is mathematically lossless.

Benchmarked this with real GPT-2 weights in pure NumPy. Without cache, per-token latency grows linearly with context length. With cache, it stays near-constant. At 200 tokens of context, the cached path is already 2.6x faster — and the gap keeps widening.

The tradeoff is memory. The formula: 2 x n_layers x n_embd x seq_len x bytes_per_element. For a 70B model at 32K context in float16, that's ~160 GB per request — more than a single H100. Multiply by concurrent users and GPU memory, not compute, becomes the binding constraint.

This one data structure is why PagedAttention (vLLM), prefix caching, and KV quantization exist. Every major inference serving system is, in significant part, a KV cache management system.

The notebook (https://github.com/elizabetht/100-days-of-inference/blob/main/day05/kv-cache.ipynb) builds the cache from scratch, verifies identical output, benchmarks the speedup, and computes memory costs from GPT-2 to GPT-3 scale.

#LLM #Inference #KVCache #GPT2 #PagedAttention #vLLM #DeepLearning #AI #MLEngineering #100DaysOfInference
