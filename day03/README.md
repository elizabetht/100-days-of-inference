# Day 03

**Topic:** Mixture of Experts (MoE) Routing
**Date:** 2026-04-06
**Layer:** Runtime

## What I explored
Studied how MoE layers replace dense FFNs with N sparse experts, routing each token to only top-K of them via a learned gating network. Implemented a complete MoE layer with top-K routing, load balancing loss, and parameter efficiency analysis comparing real model configs (Llama-3-8B vs Mixtral 8x7B).

## Key insight
MoE achieves 4x more parameters than a dense model while using the same FLOPs per token — quality scales with total parameters, but inference cost only scales with active parameters.

## Code / experiment
Notebook: [`moe-routing.ipynb`](./moe-routing.ipynb)
Key demo: Expert load distribution visualization under balanced vs collapsed routing

## References
- *Inference Engineering* Ch 2.2.4 (Philip Kiely, Baseten Books 2026)
- Shazeer et al. (2017), "Outrageously Large Neural Networks"
- Fedus et al. (2021), "Switch Transformers"
