# Day 25

**Topic:** Performance Benchmarking: Tooling & Profiling
**Date:** 2026-04-06
**Layer:** Tooling

## What I explored
Built a benchmarking toolkit covering TTFT/TPOT measurement, torch.profiler kernel tracing, and throughput-latency tradeoff analysis across batch sizes. Demonstrated why warm-up is essential and how batch size sweeps reveal the GPU's performance envelope.

## Key insight
The throughput-latency Pareto curve is the fundamental tool for capacity planning: find the batch size where throughput flattens (memory-bound plateau) and latency begins to grow — that's your operating point.

## Code / experiment
Notebook: [`performance-benchmarking.ipynb`](./performance-benchmarking.ipynb)
Key demo: Latency and throughput vs batch size curves + torch.profiler output

## References
- *Inference Engineering* Ch 4.5 (Philip Kiely, Baseten Books 2026)
- PyTorch Profiler documentation
