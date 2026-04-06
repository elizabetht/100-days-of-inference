# Day 30

**Topic:** Scaled Dot-Product Attention with Masking
**Date:** 2026-04-06
**Layer:** Implementation

## What I explored
Implemented SDPA from scratch with causal masking. Visualized attention patterns, measured O(n²) memory and compute scaling, and compared against PyTorch's F.scaled_dot_product_attention() which automatically dispatches to FlashAttention.

## Key insight
The attention score matrix is O(n²) — 4096² × 32 heads × FP16 = 1GB just for scores. This is the exact bottleneck FlashAttention eliminates by never materializing the full matrix in HBM.

## Code / experiment
Notebook: [`scaled-dot-product-attention.ipynb`](./scaled-dot-product-attention.ipynb)
Key demo: Attention pattern visualization + seq² memory scaling

## References
- *Inference Engineering* (Philip Kiely, Baseten Books 2026)
