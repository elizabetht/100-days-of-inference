# Day 36

**Topic:** Simulate Draft-Target Speculative Decoding
**Date:** 2026-04-06
**Layer:** Implementation

## What I explored
Built a complete speculative decoding simulator with draft and target language models. Measured acceptance rates by draft quality and position, confirming that early positions accept more often than later ones.

## Key insight
The first draft token has acceptance probability = draft_quality, but the Kth token has probability = draft_quality^K — this is why acceptance rate falls with K and why K=4-8 is typical.

## Code / experiment
Notebook: [`speculative-decoding-simulation.ipynb`](./speculative-decoding-simulation.ipynb)
Key demo: Acceptance rate by draft quality + position-wise acceptance distribution

## References
- *Inference Engineering* (Philip Kiely, Baseten Books 2026)
