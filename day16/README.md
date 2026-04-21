# Day 16 — Prefix caching is a prompt design problem

Prefix caching sounds like a runtime trick — a hash table inside vLLM that skips prefill when two requests share tokens. It is. But the production lesson is that it's really a constraint on how prompts get built: where the novel tokens sit decides whether time-to-first-token (TTFT) drops 100x or doesn't move at all.

The mechanic falls out of autoregression. Every token's key-value (KV) entry depends on every token before it, so two requests sharing the first k tokens have bit-identical KV entries for positions [0, k). The prefix ends at the first non-matching token and cannot resume — one changed token at position 0 wipes the whole cache. A concrete example makes it stick: "Weather in SF?" and "Weather in NYC?" share two tokens. Flip it to "SF weather today?" versus "NYC weather today?" and the savings vanish even though the suffix is identical. The rule: variable parts go last. A 10,000-token system prompt with a 50-token user question ships TTFTs 200x faster with good templating than bad.

The KV cache size math sets the budget. Per-token bytes = num_layers × 2 × num_kv_heads × head_dim × dtype_bytes. For Llama-3 70B with grouped-query attention, that's 320 KB per token — 40 GB for a 128K context. When video RAM (VRAM) fills, blocks fall down the hierarchy: G1 device memory (terabytes/s), G2 host RAM (tens to hundreds of GB/s), G3 local SSD (5–10 GB/s), G4 networked SSD. The offload-vs-re-prefill decision comes down to a bandwidth race: re-prefill produces KV at a rate set by the engine's prefill throughput; the interconnect delivers cached KV at its own rate. Whichever is faster wins. On PCIe-attached hosts this is a live call — the answer depends on model size, batch shape, and prefill pipeline. On SKUs like the GB200, NVLink-C2C makes G2 bandwidth so much higher than any plausible prefill rate that offloading wins cleanly, which is why it's the natural offloading platform.

The third piece is cache-aware routing. Production runs N replicas behind a load balancer; replica 1's cache is invisible to replica 2, so a round-robin router sends returning users to whichever replica is free and pays full TTFT every time. Cache-aware routing hashes the prefix and sticks it to the replica that served it before. The notebook simulation — three replicas, eight system prompts, Zipfian traffic, 200 requests — landed at 55% hit rate and 73 ms mean TTFT under round-robin, 96% and 7.4 ms under cache-aware. A 10x speedup from routing logic alone.

Templating, memory tiering, and router design are one problem viewed from three layers. All three hinge on the autoregressive constraint that makes token position load-bearing. The KV cache is extremely valuable and extremely fragile to changes at low positions. Production inference is the art of arranging prompts, hardware, and routers so the prefix is as long, warm, and close to the right GPU as possible.

The notebook ([`kv-cache-prefix-caching.ipynb`](./kv-cache-prefix-caching.ipynb)) tokenizes the Weather example, computes KV sizes across GPT-2, three Llama-3 sizes, and DeepSeek-V3, models templating savings on a 10K-token document, tabulates the G1–G4 hierarchy with breakeven analysis, and runs the routing simulation end-to-end. Next up: model parallelism — when the model stops fitting on one GPU.

Inspired by *Inference Engineering* (Philip Kiely, Baseten Books, 2026). Public deep-dives worth reading:
- vLLM Automatic Prefix Caching — [docs.vllm.ai/en/latest/features/automatic_prefix_caching.html](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching.html)
- SGLang RadixAttention (Zheng et al., 2024) — [arxiv.org/abs/2312.07104](https://arxiv.org/abs/2312.07104)
- PagedAttention / vLLM (Kwon et al., 2023) — [arxiv.org/abs/2309.06180](https://arxiv.org/abs/2309.06180)
- Anthropic prompt caching — [docs.claude.com/en/docs/build-with-claude/prompt-caching](https://docs.claude.com/en/docs/build-with-claude/prompt-caching)

#LLM #Inference #PrefixCaching #KVCache #vLLM #SGLang #CacheAwareRouting #PromptDesign #TTFT #DeepLearning #AI #MLEngineering #100DaysOfInference
