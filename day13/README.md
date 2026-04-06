# Day 13

**Topic:** Speculative Decoding: Draft-Target, Medusa, EAGLE
**Date:** 2026-04-06
**Layer:** Runtime

## What I explored
Implemented speculative decoding acceptance sampling from scratch and derived the speedup formula analytically. Simulated acceptance rates under different draft quality levels, plotted optimal K vs acceptance rate, and built a Medusa multi-head prediction module.

## Key insight
The key insight: the speculative decoding output distribution is provably identical to the target model alone — speedup comes for free from the acceptance/rejection mechanism, not from approximation.

## Code / experiment
Notebook: [`speculative-decoding.ipynb`](./speculative-decoding.ipynb)
Key demo: Speedup curves vs acceptance rate and K + Medusa head implementation

## References
- *Inference Engineering* Ch 5.2 (Philip Kiely, Baseten Books 2026)
- Leviathan et al. (2022), "Fast Inference from Transformers via Speculative Decoding"
- Cai et al. (2023), "Medusa"
