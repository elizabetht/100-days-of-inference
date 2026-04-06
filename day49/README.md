# Day 49

**Topic:** TensorRT-LLM: Compile a Model and Compare
**Date:** 2026-04-06
**Layer:** Implementation

## What I explored
Mapped the TRT-LLM compilation pipeline and analyzed cumulative speedup sources from eager PyTorch to fully optimized TRT-LLM engine.

## Key insight
TRT-LLM's kernel selection (tactic profiling) alone gives 1.5-2x speedup over PyTorch eager — it benchmarks multiple GEMM implementations and picks the fastest for each shape on the target GPU.

## Code / experiment
Notebook: [`tensorrt-llm-compile.ipynb`](./tensorrt-llm-compile.ipynb)
Key demo: TRT-LLM compilation pipeline + cumulative speedup waterfall

## References
- *Inference Engineering* (Philip Kiely, Baseten Books 2026)
