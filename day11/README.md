# Day 11

**Topic:** Quantization: Number Formats (FP8, INT8, INT4)
**Date:** 2026-04-06
**Layer:** Runtime

## What I explored
Explored quantization number formats from FP32 down to INT4. Implemented symmetric quantization, measured MSE and SNR across bit widths, and computed memory savings for real model sizes across formats.

## Key insight
Each bit halves quantization error (SNR increases ~6dB per bit) — the diminishing returns above INT8 explain why INT4 is the practical lower bound for acceptable weight-only quantization quality.

## Code / experiment
Notebook: [`quantization-number-formats.ipynb`](./quantization-number-formats.ipynb)
Key demo: Quantization error (MSE/SNR) vs bit width + model memory footprint across formats

## References
- *Inference Engineering* Ch 5.1.1 (Philip Kiely, Baseten Books 2026)
- Dettmers et al. (2022), "LLM.int8()"
