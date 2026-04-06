# Day 28

**Topic:** BPE Tokenizer from Scratch
**Date:** 2026-04-06
**Layer:** Implementation

## What I explored
Implemented a full BPE tokenizer from scratch: training the merge table from a corpus, encoding text by applying merges in priority order, and measuring compression ratio vs number of merges.

## Key insight
BPE is a greedy frequency-based algorithm — each merge is locally optimal, but the resulting vocabulary is globally near-optimal for the training corpus.

## Code / experiment
Notebook: [`bpe-tokenizer.ipynb`](./bpe-tokenizer.ipynb)
Key demo: BPE merge table training + token count compression curve

## References
- *Inference Engineering* (Philip Kiely, Baseten Books 2026)
