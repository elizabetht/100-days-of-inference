# Day 09

**Topic:** TensorRT-LLM: Compilation & Plugin System
**Date:** 2026-04-06
**Layer:** Runtime

## What I explored
Mapped the TensorRT-LLM build pipeline from Python model definition to GPU-specific compiled engine. Analyzed the plugin system for LLM-specific kernels, simulated tactic selection, and quantified the cumulative speedup from each optimization layer.

## Key insight
TRT-LLM's plugin system is what separates it from generic TensorRT: by understanding LLM structure (attention patterns, KV cache, sampling), it can apply optimizations that a shape-agnostic compiler cannot.

## Code / experiment
Notebook: [`tensorrt-llm.ipynb`](./tensorrt-llm.ipynb)
Key demo: Cumulative optimization speedup waterfall chart across all TRT-LLM techniques

## References
- *Inference Engineering* Ch 4.3.3 (Philip Kiely, Baseten Books 2026)
- NVIDIA TensorRT-LLM documentation
