# Day 31

**Topic:** Flash Attention (Simplified Tiling in Python)
**Date:** 2026-04-06
**Layer:** Implementation

## What I explored
Implemented the online softmax primitive and tiled FlashAttention in pure Python/PyTorch. Verified correctness against standard SDPA and quantified HBM access reduction from O(n²) to O(n) as sequence length scales.

## Key insight
The online softmax trick is the key algorithmic insight: it allows computing exact softmax incrementally without materializing all scores, enabling tile-by-tile computation that fits in SRAM.

## Code / experiment
Notebook: [`flash-attention-tiling.ipynb`](./flash-attention-tiling.ipynb)
Key demo: Tiled Flash Attention implementation with correctness verification + HBM access comparison

## References
- *Inference Engineering* (Philip Kiely, Baseten Books 2026)
