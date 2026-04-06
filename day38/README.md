# Day 38

**Topic:** Prefix Caching with Hash-Based Deduplication
**Date:** 2026-04-06
**Layer:** Implementation

## What I explored
Built a hash-based prefix cache with block-aligned deduplication. Simulated 1000 requests with an 80% system prompt hit rate, measuring total tokens saved from recomputation.

## Key insight
Block-aligned caching is key: caching arbitrary-length prefixes is expensive in storage. Aligning to block_size=16 means a cache entry covers exactly 16 tokens, enabling O(1) lookup per block.

## Code / experiment
Notebook: [`prefix-cache-deduplication.ipynb`](./prefix-cache-deduplication.ipynb)
Key demo: Prefix cache hit rate on 1000-request simulation + hash collision probability analysis

## References
- *Inference Engineering* (Philip Kiely, Baseten Books 2026)
