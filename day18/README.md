# Day 18

**Topic:** GPU Generations: Hopper, Ada, Blackwell, Rubin
**Date:** 2026-04-06
**Layer:** Infrastructure

## What I explored
Surveyed NVIDIA GPU generations from A100 through B200, comparing FP8/FP4 throughput, HBM bandwidth, NVLink speed, and new architectural features. Analyzed cost-efficiency for decode vs prefill workloads.

## Key insight
H100 was the first GPU architecturally designed for LLM inference (FP8 Tensor Cores + Transformer Engine). Blackwell doubles it. For pure decode workloads, HBM bandwidth per dollar matters more than peak TFLOP/s.

## Code / experiment
Notebook: [`gpu-generations.ipynb`](./gpu-generations.ipynb)
Key demo: Generation-over-generation scaling charts (TFLOP/s and bandwidth) + cost-efficiency analysis

## References
- *Inference Engineering* Ch 3.2 (Philip Kiely, Baseten Books 2026)
- NVIDIA H100 and Blackwell Architecture White Papers
