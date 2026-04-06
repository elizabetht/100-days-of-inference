# Day 42

**Topic:** Custom Elementwise CUDA Kernel via Triton
**Date:** 2026-04-06
**Layer:** Implementation

## What I explored
Wrote a SiLU activation Triton kernel, verified correctness against PyTorch, and benchmarked bandwidth utilization.

## Key insight
Triton's tile-level abstraction means you never write explicit thread indices — you write programs over tiles, and the compiler maps tiles to warps/blocks automatically.

## Code / experiment
Notebook: [`custom-triton-kernel.ipynb`](./custom-triton-kernel.ipynb)
Key demo: Triton SiLU kernel correctness + bandwidth benchmark

## References
- *Inference Engineering* (Philip Kiely, Baseten Books 2026)
