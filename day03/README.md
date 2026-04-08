# Day 03 — Embeddings

Every transformer based model (GPT, LLaMA, BERT, etc) starts the same way: a lookup table. The flow is always: text → tokens (integers) → embeddings (vectors) → transformer layers.

Tokenization splits text into integer IDs, but those IDs carry zero semantic information. For example, token 4821 ("cat") is not mathematically closer to token 2955 ("dog") than to token 9173 ("airplane"). The embedding layer maps each integer to a learned dense vector: 768 dimensions in GPT-2, 4096 in LLaMA 7B, where semantic proximity is real and measurable.

GPT-2 for instance stores two embedding matrices:
- wte (50257 x 768) for token identity
- wpe (1024 x 768) for position

Without position encoding, "the cat sat on the mat" and "mat the on sat cat the" produce identical representations. The input to the first transformer block is simply wte[token_id] + wpe[position]: two table lookups and one addition.

Modern architectures like LLaMA replace the stored wpe with rotary position embeddings (RoPE), computing positions on the fly and removing the hard sequence length ceiling.

The notebook (https://github.com/elizabetht/100-days-of-inference/blob/main/day03/embeddings.ipynb) walks through this with real GPT-2 weights, loading the embedding tables, inspecting shapes, and visualizing how token vectors cluster by meaning.

#LLM #Inference #Embeddings #GPT2 #Transformers #RoPE #DeepLearning #AI #MLEngineering #100DaysOfInference
