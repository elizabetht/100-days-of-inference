# Day 34

**Topic:** GPTQ-Style Round-to-Nearest with Hessian
**Date:** 2026-04-06
**Layer:** Implementation

## What I explored
Implemented GPTQ-style weight quantization with Hessian-weighted error compensation. Compared naive round-to-nearest INT4 against GPTQ on a calibration set, demonstrating output error reduction.

## Key insight
GPTQ's Hessian-based compensation turns INT4 quantization into a second-order optimization — each quantized weight's error is partially absorbed by compensating the remaining weights in the block.

## Code / experiment
Notebook: [`gptq-implementation.ipynb`](./gptq-implementation.ipynb)
Key demo: GPTQ vs naive INT4 output error comparison

## References
- *Inference Engineering* (Philip Kiely, Baseten Books 2026)
