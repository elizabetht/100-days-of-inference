# Day 14

**Topic:** KV Cache: Prefix Caching & Cache-Aware Routing
**Date:** 2026-04-06
**Layer:** Runtime

## What I explored
Built a hash-based prefix cache with LRU eviction and simulated cache-aware routing across 4 workers. Measured compute savings from shared system prompt caching and analyzed KV cache memory requirements for production model deployments.

## Key insight
Cache-aware routing is essential: routing requests randomly across workers with prefix caches is almost as bad as no cache at all. Routing by prefix hash converts a cache from per-worker to effectively cluster-wide.

## Code / experiment
Notebook: [`kv-cache-prefix-caching.ipynb`](./kv-cache-prefix-caching.ipynb)
Key demo: Prefix cache hit rate simulation + cache-aware routing accuracy

## References
- *Inference Engineering* Ch 5.3 (Philip Kiely, Baseten Books 2026)
- vLLM prefix caching implementation
