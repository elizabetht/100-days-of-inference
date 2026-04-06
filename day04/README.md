# Day 04

**Topic:** Ops:Byte Ratio & Arithmetic Intensity
**Date:** 2026-04-06
**Layer:** Runtime

## What I explored
Explored the roofline model and arithmetic intensity as tools for understanding whether LLM inference is compute-bound or memory-bound. Built an interactive roofline plot for A100/H100/DGX Spark GPUs and computed arithmetic intensity across batch sizes for a Llama-3-8B FFN layer.

## Key insight
LLM decode at batch=1 has ~1 FLOP/byte arithmetic intensity — 100x below the A100 ridge point — making it purely memory-bound. Batching is the primary lever to improve GPU utilization.

## Code / experiment
Notebook: [`ops-byte-ratio.ipynb`](./ops-byte-ratio.ipynb)
Key demo: Roofline model plot with real GPU specs and LLM operation intensity markers

## References
- *Inference Engineering* Ch 2.4 (Philip Kiely, Baseten Books 2026)
- Williams et al. (2009), "Roofline: An Insightful Visual Performance Model"
