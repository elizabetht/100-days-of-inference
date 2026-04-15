# Day 10 — SGLang: RadixAttention and Structured Outputs

Day 09 covered how vLLM's PagedAttention eliminates per-request memory waste. But PagedAttention still recomputes the KV (Key-Value) cache from scratch for every request — even when thousands of requests start with the same 1000-token system prompt. That's redundant prefill compute, and it scales linearly with request volume.

SGLang (Structured Generation Language) fixes this with RadixAttention, which uses a radix tree to store KV cache blocks by shared token prefixes. Two requests sharing a common prefix reuse the same physical KV blocks — no recomputation. The tree tracks reference counts so no block is evicted while in use. For a chatbot with a 500-token system prompt, cache hit rate after warmup approaches 100% for that prefix. The first request pays full prefill cost; subsequent requests pay near-zero for the shared portion. SGLang is the inference engine of choice at xAI partly because its prefix caching handles the high-reuse patterns common in production chatbots.

Cache hit rates are workload-dependent. Shared system prompts produce very high rates. Diverse RAG (Retrieval Augmented Generation) retrievals produce low rates. Multi-turn conversation falls somewhere in between, high for session context. If the hit rate is under 20%, prefix caching is wasting memory that could hold more KV blocks. Above 60%, it's the single biggest throughput win in the system.

SGLang's second contribution is constrained decoding for structured outputs. Instead of generating tokens freely and validating afterward, the sampler masks invalid tokens at each step against a grammar or JSON schema. The output is valid by construction — no retry loops, no post-processing. SGLang is the engine of choice for MoE (Mixture of Experts) models (DeepSeek, Qwen) and for structured output use cases at scale.

The notebook ([sglang-radix-attention.ipynb](https://github.com/elizabetht/100-days-of-inference/tree/main/day10/sglang-radix-attention.ipynb)) implements a RadixAttention trie in Python, simulates cache hit rates under different workload distributions, and models the throughput gain from prefix reuse.

#LLM #Inference #SGLang #RadixAttention #PrefixCaching #StructuredOutput #DeepLearning #AI #MLEngineering #100DaysOfInference
