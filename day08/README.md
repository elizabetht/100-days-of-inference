# Day 08

**Topic:** SGLang: RadixAttention & Structured Outputs
**Date:** 2026-04-06
**Layer:** Runtime

## What I explored
Implemented a RadixCache tree for automatic KV prefix sharing and simulated FSM-based constrained decoding for structured JSON output. Measured prefix cache hit rates and quantified latency savings from speculative prefill on fixed output patterns.

## Key insight
RadixAttention's tree structure enables KV cache reuse even across non-concurrent requests — a system prompt cached from yesterday's traffic is reused today without any explicit management.

## Code / experiment
Notebook: [`sglang-radix-attention.ipynb`](./sglang-radix-attention.ipynb)
Key demo: RadixCache prefix sharing simulation + FSM-constrained JSON generation walkthrough

## References
- *Inference Engineering* Ch 4.3.2 (Philip Kiely, Baseten Books 2026)
- Zheng et al. (2023), "SGLang: Efficient Execution of Structured Language Model Programs"
