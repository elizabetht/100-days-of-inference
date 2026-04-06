# Day 07

**Topic:** vLLM: PagedAttention & Continuous Batching
**Date:** 2026-04-06
**Layer:** Runtime

## What I explored
Implemented PagedAttention block allocation and continuous batching simulation from scratch. Visualized KV cache fragmentation under naive vs paged allocation, and benchmarked GPU utilization under static vs continuous batching with realistic request arrival patterns.

## Key insight
Continuous batching fills GPU slots at the iteration level — not the batch level — converting GPU idle time from O(max_request_length) to O(1 decode step). Combined with PagedAttention, this is why vLLM achieves near-100% GPU utilization.

## Code / experiment
Notebook: [`vllm-paged-attention.ipynb`](./vllm-paged-attention.ipynb)
Key demo: BlockAllocator simulation + continuous vs static batching utilization comparison

## References
- *Inference Engineering* Ch 4.3.1 (Philip Kiely, Baseten Books 2026)
- Kwon et al. (2023), "Efficient Memory Management for LLM Serving with PagedAttention"
