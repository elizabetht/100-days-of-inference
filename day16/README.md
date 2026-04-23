# Day 16 — Prefix caching is a prompt design problem

Prefix caching sounds like a runtime trick — a hash table inside vLLM that skips prefill when two requests share tokens. It is. But the production lesson is that it's really a constraint on how prompts get built: where the novel tokens sit decides whether time-to-first-token (TTFT) drops 100x or doesn't move at all.

The mechanic falls out of autoregression. Every token's key-value (KV) entry depends on every token before it, so two requests sharing the first k tokens have bit-identical KV entries for positions [0, k). "Weather in SF?" and "Weather in NYC?" share two tokens; flip to "SF weather today?" vs "NYC weather today?" and the savings vanish. The rule: variable parts go last. A 10K-token system prompt with a 50-token user question ships TTFTs 200x faster with good templating than bad.

The KV cache is big — 320 KB/token for Llama-3 70B, 40 GB for a 128K context — so it falls down a memory hierarchy (G1 video RAM → G2 host RAM → G3 local SSD → G4 networked SSD). The offload-vs-re-prefill decision is a bandwidth race: whichever is faster between the engine's prefill throughput and the interconnect wins. On PCIe hosts it's a live call; on GB200 with NVLink-C2C, G2 is so fast that offloading always wins.

The third piece is cache-aware routing. Production runs N replicas; replica 1's cache is invisible to replica 2, so a round-robin router pays full TTFT every time a user returns. Cache-aware routing hashes the prefix and pins it to the replica that served it. The notebook simulation — three replicas, eight system prompts, Zipfian traffic, 200 requests — went from 55% hit rate and 73 ms TTFT under round-robin to 96% and 7.4 ms under cache-aware. A 10x speedup from routing logic alone.

Templating, memory tiering, and routing are one problem viewed from three layers, all hinging on the autoregressive constraint that makes token position load-bearing. Production inference is the art of arranging prompts, hardware, and routers so the prefix is as long, warm, and close to the right GPU as possible.

Full walkthrough: [`kv-cache-prefix-caching.ipynb`](./kv-cache-prefix-caching.ipynb). Next up: model parallelism.

Inspired by *Inference Engineering* (Philip Kiely, Baseten Books, 2026). Deep-dives: vLLM [automatic prefix caching](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching.html) and SGLang [RadixAttention](https://arxiv.org/abs/2312.07104).

#LLM #Inference #PrefixCaching #KVCache #vLLM #SGLang #CacheAwareRouting #PromptDesign #TTFT #DeepLearning #AI #MLEngineering #100DaysOfInference
