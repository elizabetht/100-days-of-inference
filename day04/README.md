# Day 04 — Attention

Embeddings turn tokens into vectors, but each token is still computed in isolation. Attention is what lets tokens relate to one another in context.

Each token creates a Q (Query), K (Key), and V (Value). Query-Key scores decide relevance, softmax turns those into probabilities, and the model uses them to mix the right Values. In GPT-style models, a causal mask ensures each token can only attend to earlier tokens.

GPT-2 does this through multi-head attention: several attention patterns running in parallel, each learning different roles. Some track local relationships, others gather broader context.

The catch is cost: attention scales as O(n²). Double the sequence length, and compute grows 4x. That bottleneck is why optimizations like FlashAttention, KV caching, and prefix caching matter so much.

Tried a coreference example: "The dog chased the ball because it was excited." In small GPT-2, attention struggles to link "it" back to "dog." Same mechanism, just less capacity. Larger models show much stronger signals.

The notebook (https://github.com/elizabetht/100-days-of-inference/blob/main/day04/attention.ipynb) builds this from scratch using real GPT-2 weights in pure NumPy.

#LLM #Inference #Attention #GPT2 #Transformers #FlashAttention #DeepLearning #AI #MLEngineering #100DaysOfInference
