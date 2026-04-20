# Day 15 — Speculative Decoding: Draft, Verify, Repeat

Autoregressive decoding is sequential: each token depends on all previous tokens, and at batch size 1 the entire model gets read from High Bandwidth Memory (HBM) for every token generated. A 70B model on an H100 does ~30 tokens/sec not because the silicon is too slow, but because the compute units sit idle waiting on memory.

Speculative decoding breaks this. A cheap "draft" model proposes K tokens at once, and the expensive "target" model verifies all K in a single forward pass — verification is parallel even though generation is not. Think writer and editor: the editor drafts several tokens ahead, the writer double-checks and corrects where needed, and together they emit 2-4 tokens in the time the writer alone would have produced one. Output quality is identical because verification uses rejection sampling — a mathematical guarantee, not a heuristic.

The mechanism has three steps per round. **Token speculation**: the draft emits K candidates with per-token probabilities. **Parallel verification**: the target runs one forward pass over all K, returning its own probabilities for each candidate plus a prediction for token K+1. **Rejection sampling**: walk the candidates in order, accept each with probability min(1, TP/DP); on the first rejection, discard everything after and resample from the renormalized difference of the two distributions. A perfect round emits K+1 tokens per target call; a worst-case round still emits 1.

Five mechanisms dominate production. **Draft-target** pairs two models — strongest acceptance, highest setup cost. **Medusa** bolts parallel prediction heads onto the target, no second model needed. **EAGLE / EAGLE-2 / EAGLE-3** drafts from the target's hidden states with tree expansion and multi-layer drafting, pushing acceptance 10-15% higher. **N-gram speculation** uses no model — match the last N tokens to earlier in the sequence and propose whatever followed, free on Retrieval-Augmented Generation (RAG) and code. **MLP speculator** (IBM Granite) is a small trained Multi-Layer Perceptron on hidden states, drop-in 2-3× decode speedup.

vLLM V1 supports n-gram and EAGLE only; draft-target, Medusa, and MLP speculator are V0-only and deprecating. New vLLM deployments pick n-gram (RAG-heavy) or EAGLE (everything else).

The notebook ([speculative-decoding.ipynb](./speculative-decoding.ipynb)) implements the full three-step loop with rejection sampling, sweeps acceptance rates and proposal lengths, builds an n-gram proposer from scratch, and shows why the same algorithm delivers 1.5× on RAG-like traffic and 1.0× on novel generation.

#LLM #Inference #SpeculativeDecoding #Medusa #EAGLE #NgramSpeculation #IBMGranite #vLLM #DeepLearning #AI #MLEngineering #100DaysOfInference
