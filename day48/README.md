# Day 48

**Topic:** Benchmark TTFT vs Throughput Across Batch Sizes
**Date:** 2026-04-06
**Layer:** Implementation

## What I explored
Benchmarked TTFT and throughput across batch sizes 1-64, plotted the opposing curves, and implemented an SLO-based optimizer to find the maximum throughput batch size within a latency budget.

## Key insight
TTFT and throughput have exactly opposite batch size sensitivities — this is the defining tradeoff in inference serving and why continuous batching uses small dynamic batch sizes for interactive traffic.

## Code / experiment
Notebook: [`ttft-throughput-tradeoff.ipynb`](./ttft-throughput-tradeoff.ipynb)
Key demo: TTFT vs batch size and throughput vs batch size curves

## References
- *Inference Engineering* (Philip Kiely, Baseten Books 2026)
