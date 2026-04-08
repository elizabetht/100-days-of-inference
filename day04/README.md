# Day 04

**Topic:** Transformer Blocks & Attention Deep Dive

**Date:** 2026-04-06

**Layer:** Runtime

## What I explored

Built attention from scratch using real GPT-2 weights: Q/K/V projections, scaled dot-product attention, causal masking, multi-head splitting, output projection with residual connections. Also implemented the MLP (feed-forward) sublayer — expand to 4x, GELU activation, compress back — to run complete transformer blocks through all 12 layers and trace how attention patterns evolve from early to late layers.

Visualized GELU vs ReLU to understand why non-linearity matters: without it, stacked linear layers collapse into a single matrix multiply and depth adds no capacity.

Compared attention variants used in modern LLMs: MHA (GPT-2/3), MQA (PaLM, Falcon), and GQA (Llama 2/3, Mistral) — same core equation, different KV sharing strategies that reduce cache size during inference.

## Key insight

Attention is a learned weighted lookup — each token queries all prior tokens, and the causal mask enforces autoregressive generation. The quadratic O(n²) cost in sequence length is the fundamental bottleneck that drives every major inference optimization (FlashAttention, KV cache, prefix caching).

Tested coreference resolution by probing whether "it" attends to "dog" in *"The dog chased the ball because it was excited."* GPT-2 doesn't solve it — attention collapses into the BOS attention sink (first token absorbs 55–70% of weight). The attention mechanism is identical across model sizes; the difference is whether the learned weights are rich enough for semantic reasoning vs. falling back on positional shortcuts.

## Code / experiment

Notebook: [`attention.ipynb`](./attention.ipynb)

Key demos:
- Multi-head attention with real GPT-2 weights — visualizes all 12 heads and their entropy-based specialization (focused vs. spread)
- Complete transformer blocks (attention + MLP) through all 12 layers
- GELU vs ReLU visualization with annotated comparison
- Coreference probe across layers showing the BOS attention sink pattern

## References

- *Inference Engineering* Ch 2.2.2–2.2.3 (pp. 50–53)
- Vaswani et al., "Attention Is All You Need" (2017)
- Dao et al., "FlashAttention" (2022)
