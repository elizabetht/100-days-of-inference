# Day 17 — Model Parallelism: Tensor, Expert, Pipeline and Data Parallelism

Llama 3 70B in FP16 is 140 GB — bigger than one H100. Llama 405B is 800 GB. DeepSeek-V3 671B in FP8 is 671 GB — bigger than a full 8×H200 node. At this scale model parallelism isn't an optimization, it's the only way the model fits. Tensor and Expert parallelism are the inference-specific axes; production combines four.

Tensor Parallelism (TP) slices each weight matrix vertically across GPUs. On 8 GPUs, each holds 1/8 of the weights and does 1/8 of the math, then the group syncs and moves to the next layer. Attention heads split the same way — 64 heads becomes 8 per GPU. Memory and latency both drop ~8×. The cost: every transformer block ends with a sync. NVLink at 900 GB/s makes that ~2ms; InfiniBand at 50 GB/s makes it ~36ms — 18× slower, enough to kill the gain. So TP almost always stops at 8 GPUs, one NVLink-connected node.

Expert Parallelism (EP) puts each MoE expert on a different GPU. The router sends tokens to their top-2 experts via all-to-all communication. Because the sync only happens at the MoE layer, not every matmul, EP scales to hundreds of GPUs. It's how DeepSeek-V3 runs across large clusters.

Pipeline Parallelism (PP) assigns contiguous layer blocks to different GPUs — GPU 0 holds layers 0–15, GPU 1 holds 16–31, etc. Activations hop between stages, and per-stage communication is tiny, so PP works across nodes over InfiniBand. The catch is the pipeline bubble: with one request in flight on a 4-stage pipeline, three GPUs sit idle. PP=8 needs ~16 concurrent micro-batches to keep the bubble under 30% — that concurrency has to come from real user traffic.

Data Parallelism (DP) is the outer loop: replicate the whole model and route independent requests to different replicas. Zero cross-replica communication. Throughput scales linearly, but VRAM and dollar cost multiply — DP is a capacity dial, not an efficiency dial.

Production stacks combine them: `DP × (TP[node] × EP[moe] × PP[multi-node])`. TP+EP+PP fit the model and hit latency; DP scales QPS on top.

Walkthrough with runnable bubble + throughput sims: [`model-parallelism.ipynb`](./model-parallelism.ipynb). Next: prefill/decode disaggregation.

Inspired by *Inference Engineering* (Philip Kiely, Baseten Books, 2026). PP and DP draw on Narayanan et al. 2021 (Megatron-LM).

#LLM #Inference #TensorParallelism #ExpertParallelism #PipelineParallelism #DataParallelism #ModelParallelism #NVLink #DeepLearning #AI #MLEngineering #100DaysOfInference
