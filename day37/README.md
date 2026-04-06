# Day 37

**Topic:** Build a KV Cache Manager
**Date:** 2026-04-06
**Layer:** Implementation

## What I explored
Built a full KV cache block allocator with LRU eviction. Simulated a serving session with allocation, extension, and eviction events, measuring utilization at each step.

## Key insight
Block-based KV cache management eliminates internal fragmentation: padding to max_seq_len is replaced by on-demand block extension, allowing 100% utilization in theory.

## Code / experiment
Notebook: [`kv-cache-manager.ipynb`](./kv-cache-manager.ipynb)
Key demo: Block allocator simulation with LRU eviction + utilization tracking

## References
- *Inference Engineering* (Philip Kiely, Baseten Books 2026)
