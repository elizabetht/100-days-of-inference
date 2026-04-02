# Day 01

**Topic:** What Is LLM Inference? End-to-end text generation without optimizations

**Date:** 2026-03-31

**Layer:** Runtime

## What I explored

What inference actually is and what happens end-to-end when an LLM generates text — no optimizations, just the baseline mechanics. Tokenization converts text to integer IDs (like a protobuf serializer). A forward pass through the model produces logits — a probability score for every token in the vocabulary. Decoding picks one token from those scores. The autoregressive loop repeats forward pass + decode for each output token, re-reading the entire sequence every time.

## Key insight

The model generates text one token at a time, and each step requires a full forward pass over the entire sequence so far. This means generation time grows with sequence length — later tokens are more expensive than earlier ones. Every optimization in inference engineering (KV caching, batching, speculative decoding) exists to reduce this fundamental cost.

## Code / experiment

Notebook: [`llm-inference-mechanics.ipynb`](./llm-inference-mechanics.ipynb)

Key demos (all using real GPT-2):
- Tokenization: encode/decode with a real tokenizer, token splitting on infra terms (Kubernetes, NVIDIA, IP addresses)
- Forward pass: one pass through the model → logits shape, top-10 next-token probabilities
- Decoding strategies: greedy (argmax), temperature scaling effect on probability distribution
- Hand-rolled autoregressive loop: tokenize → forward → decode → append, step by step
- Timing benchmark: bar chart showing per-token latency growing with sequence length
- Library comparison: `model.generate()` produces identical output to the manual loop

## Next up

Day 02: Building the entire inference pipeline from scratch — tokenization, forward pass, and autoregressive decoding — without any libraries, to understand what the model actually computes inside each step.

## References

- *Inference Engineering* Ch 2.2 (pp. 46-54) - Philip Kiely, Baseten Books 2026
