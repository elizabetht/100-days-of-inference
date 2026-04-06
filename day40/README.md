# Day 40

**Topic:** Benchmark Ops:Byte Ratio in Practice
**Date:** 2026-04-06
**Layer:** Implementation

## What I explored
Benchmarked matmul arithmetic intensity and throughput across batch sizes, overlaying on the GPU roofline model. Confirmed that batch=1 decode is 100x below the ridge point while batch=4096 prefill approaches compute peak.

## Key insight
Empirical roofline validation confirms the theory: batch=1 decode achieves ~1 TFLOP/s (memory-bound), while large square GEMMs achieve 50+ TFLOP/s (compute-bound).

## Code / experiment
Notebook: [`ops-byte-ratio-benchmark.ipynb`](./ops-byte-ratio-benchmark.ipynb)
Key demo: Empirical roofline plot with measured matmul performance across batch sizes

## References
- *Inference Engineering* (Philip Kiely, Baseten Books 2026)
