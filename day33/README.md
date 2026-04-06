# Day 33

**Topic:** INT8 Quantization Pipeline
**Date:** 2026-04-06
**Layer:** Implementation

## What I explored
Built a complete INT8 quantization pipeline: symmetric and asymmetric quantization, per-tensor vs per-channel scale calibration, and a QuantizedLinear layer. Measured error reduction from per-channel quantization on weight matrices with outlier channels.

## Key insight
Per-channel quantization gives 10-100x lower error than per-tensor for LLM weights — the key insight is that outlier channels in transformer weights need their own scale to avoid dominating the quantization range.

## Code / experiment
Notebook: [`int8-quantization-pipeline.ipynb`](./int8-quantization-pipeline.ipynb)
Key demo: Per-channel vs per-tensor error comparison + quantized linear layer implementation

## References
- *Inference Engineering* (Philip Kiely, Baseten Books 2026)
