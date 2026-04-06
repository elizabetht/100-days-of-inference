# Day 47

**Topic:** Visualize PagedAttention Block Layout
**Date:** 2026-04-06
**Layer:** Implementation

## What I explored
Visualized PagedAttention physical memory layout and logical block table for 5 concurrent sequences, analyzed fragmentation at different block sizes.

## Key insight
Block size is a key PagedAttention hyperparameter: small blocks (8 tokens) minimize fragmentation but add metadata overhead; large blocks (128 tokens) are efficient for long sequences but waste memory for short ones.

## Code / experiment
Notebook: [`paged-attention-visualization.ipynb`](./paged-attention-visualization.ipynb)
Key demo: Physical memory block layout + logical block table visualization

## References
- *Inference Engineering* (Philip Kiely, Baseten Books 2026)
