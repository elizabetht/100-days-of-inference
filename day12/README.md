# Day 12

**Topic:** Quantization: GPTQ, AWQ, SmoothQuant
**Date:** 2026-04-06
**Layer:** Runtime

## What I explored
Implemented three PTQ algorithms — SmoothQuant, AWQ, and a conceptual GPTQ — to handle LLM activation outliers. Measured the quantization error reduction of each approach and visualized the per-channel scale distributions they learn.

## Key insight
Activation outliers in ~1% of channels account for the majority of INT8 quantization error. SmoothQuant and AWQ both isolate and protect these channels, explaining why they achieve near-FP16 accuracy at INT4-INT8 precision.

## Code / experiment
Notebook: [`quantization-gptq-awq-smoothquant.ipynb`](./quantization-gptq-awq-smoothquant.ipynb)
Key demo: Activation outlier visualization + SmoothQuant vs AWQ error comparison

## References
- *Inference Engineering* Ch 5.1.2 (Philip Kiely, Baseten Books 2026)
- Xiao et al. (2022), "SmoothQuant"
- Lin et al. (2023), "AWQ"
