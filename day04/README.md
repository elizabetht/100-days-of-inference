# Day 04

**Topic:** Mixture of Experts (MoE) Routing

**Date:** 2026-04-01

**Layer:** Runtime

## What I explored

Built MoE routing from scratch: dense vs sparse FFN comparison, router implementation, expert selection and weighted combination. Simulated how sparsity vanishes under batched serving and compared memory vs compute costs for real MoE models (Mixtral, DeepSeek-V3, Qwen3-235B).

## Key insight

MoE models have far more total parameters than active parameters per token. This means they need more GPU memory (all experts loaded) but less compute per request. The sparsity benefit shrinks with batch size as different requests activate different experts.

## Code / experiment

Notebook: [`moe-routing.ipynb`](./moe-routing.ipynb)

Key demo: implements MoE routing from scratch, visualizes expert selection heatmaps and load balance, and simulates expert utilization vs batch size.

## References

- *Inference Engineering* Ch 2.2.4 (pp. 53–55)
