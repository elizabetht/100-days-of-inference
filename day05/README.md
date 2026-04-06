# Day 05

**Topic:** CUDA Kernels, Kernel Selection & Kernel Fusion
**Date:** 2026-04-06
**Layer:** Runtime

## What I explored
Explored CUDA kernel dispatch, launch overhead, and the mechanics of kernel fusion. Benchmarked unfused vs compiled SwiGLU, quantified memory access reduction from FlashAttention-style fusion, and analyzed why fusion is transformative for memory-bound operations.

## Key insight
Kernel fusion eliminates intermediate HBM writes. FlashAttention reduces attention memory accesses by 10-100x for long sequences by fusing QK^T, softmax, and AV into a single tiled kernel pass.

## Code / experiment
Notebook: [`cuda-kernels-fusion.ipynb`](./cuda-kernels-fusion.ipynb)
Key demo: Memory access comparison: standard attention vs FlashAttention across sequence lengths

## References
- *Inference Engineering* Ch 4.1 (Philip Kiely, Baseten Books 2026)
- Dao et al. (2022), "FlashAttention"
