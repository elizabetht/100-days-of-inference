# Day 03

**Topic:** Embeddings — From Integers to Vectors

**Date:** 2026-04-04

**Layer:** Runtime

## What I explored

Traced the full embedding pipeline in GPT-2 using raw numpy and real model weights. Loaded the two embedding tables (`wte` and `wpe`) from `model.safetensors`, looked up token vectors by ID, added position vectors, and verified that cosine similarity captures semantic relationships (king/queen: 0.66, GPU/CPU: 0.68) and positional proximity.

## Key insight

Token embedding is a table lookup, not a matrix multiplication — `wte[token_id]` gives you the 768-dim vector. Position embedding (`wpe`) adds location awareness so the model can distinguish word order. The input to the first transformer block is simply `wte[token_ids] + wpe[:seq_len]`.

## Code / experiment

Notebook: [`embeddings.ipynb`](./embeddings.ipynb)

Key demo: loads GPT-2's wte (50257 × 768) and wpe (1024 × 768) tables, embeds a sentence end-to-end, measures cosine similarity between word pairs, and shows position similarity decay with distance.

## References

- *Inference Engineering* Ch 2.2.1 (pp. 49–50) — Model architecture, config.json
- *Inference Engineering* Ch 2.2.2 (pp. 50–51) — Embedding layer, transformer block structure
