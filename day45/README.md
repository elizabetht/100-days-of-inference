# Day 45

**Topic:** SGLang: Structured Output Latency Benchmark
**Date:** 2026-04-06
**Layer:** Implementation

## What I explored
Benchmarked SGLang structured output configuration and simulated constrained vs unconstrained decode latency.

## Key insight
Constrained decoding overhead is proportional to FSM complexity: a simple JSON schema with 2 fields adds <1ms per token; a complex nested schema may add 5-10ms.

## Code / experiment
Notebook: [`sglang-structured-output.ipynb`](./sglang-structured-output.ipynb)
Key demo: Structured output latency vs unconstrained baseline

## References
- *Inference Engineering* (Philip Kiely, Baseten Books 2026)
