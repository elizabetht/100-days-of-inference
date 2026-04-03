# Day 02

**Topic:** Inference from Scratch — No Libraries

**Date:** 2026-04-02

**Layer:** Runtime

## Notebooks

Two ~20-minute sessions. Each one focuses on a single concept, uses real GPT-2 weights, and builds on the previous.

| # | Notebook | What You Build |
|---|----------|---------------|
| 2a | [`01-whats-inside-a-model.ipynb`](./01-whats-inside-a-model.ipynb) | Download GPT-2, parse SafeTensors by hand, explore every weight tensor |
| 2b | [`02-tokenization.ipynb`](./02-tokenization.ipynb) | BPE tokenizer from scratch — encoding, decoding, merge tracing |

## Setup

Run notebook 01 first — it downloads GPT-2 weights into `gpt2_weights/`. All subsequent notebooks read from that directory. No pip installs beyond numpy.

## Key insight

A model is just a container of numbered arrays. Inference is indexing into those arrays, multiplying them together, and applying simple math functions. There is no magic — just matmuls, softmax, and a loop.

## References

- *Inference Engineering* Ch 2.1–2.2 (pp. 42–54) — Philip Kiely, Baseten Books 2026
