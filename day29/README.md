# Day 29

**Topic:** Autoregressive Decoder Loop
**Date:** 2026-04-06
**Layer:** Implementation

## What I explored
Built a complete autoregressive decoder loop with KV cache from scratch. Measured TTFT vs TPOT, tracked KV cache memory growth with sequence length, and implemented top-k sampling.

## Key insight
The KV cache converts attention from O(n²) recomputation to O(n) memory — without it, each decode step requires full attention over all prior tokens.

## Code / experiment
Notebook: [`autoregressive-decoder.ipynb`](./autoregressive-decoder.ipynb)
Key demo: KV cache memory growth visualization + TTFT/TPOT measurement

## References
- *Inference Engineering* (Philip Kiely, Baseten Books 2026)
