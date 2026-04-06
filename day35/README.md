# Day 35

**Topic:** Quantization Bit Width Sweep: Quality vs Compression
**Date:** 2026-04-06
**Layer:** Implementation

## What I explored
Swept quantization from INT2 to INT8, measuring relative output error and memory savings at each precision. Plotted the Pareto curve showing INT4 as the quality-compression sweet spot.

## Key insight
The Pareto curve inflects sharply between INT3 and INT4 — adding the 4th bit nearly eliminates the quantization error that makes INT3 unusable.

## Code / experiment
Notebook: [`quantization-bitwidth-sweep.ipynb`](./quantization-bitwidth-sweep.ipynb)
Key demo: Quality vs compression Pareto curve across bit widths

## References
- *Inference Engineering* (Philip Kiely, Baseten Books 2026)
