# Day 03

**Topic:** Transformer Blocks & Attention Deep Dive

**Date:** 2026-04-01

**Layer:** Runtime

## What I explored

Opened up the transformer black box: embedding layer, multi-head attention, feed-forward networks, residual connections, and the output layer (LMHead). Built each component from scratch in PyTorch and traced data shapes through a real GPT-2 model.

## Key insight

The FFN holds ~2/3 of the parameters per block, but attention is the more complex operation. Multi-head attention lets the model track different kinds of token relationships in parallel.

## Code / experiment

Notebook: [`transformer-blocks-attention.ipynb`](./transformer-blocks-attention.ipynb)

Key demo: traces hidden state shapes through every sub-layer of a transformer block, and visualizes how different attention heads learn different patterns.

## References

- *Inference Engineering* Ch 2.2.2–2.2.3 (pp. 50–53)
