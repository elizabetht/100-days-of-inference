# Day 32

**Topic:** Profile Attention Memory Growth
**Date:** 2026-04-06
**Layer:** Implementation

## What I explored
Systematically profiled attention memory growth: analytical breakdown by component (weights, KV cache, attention scores), empirical GPU measurement using CUDA memory stats, and visualization of the O(n²) vs O(n) scaling curves.

## Key insight
The attention score matrix crosses the KV cache in memory at around seq=2K-4K — beyond that, FlashAttention's O(n) HBM access is not just faster but the only way to fit in VRAM.

## Code / experiment
Notebook: [`attention-memory-profiling.ipynb`](./attention-memory-profiling.ipynb)
Key demo: Memory breakdown table + attention scores vs KV cache scaling curves

## References
- *Inference Engineering* (Philip Kiely, Baseten Books 2026)
