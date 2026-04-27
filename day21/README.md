# Day 21 — Multi-GPU Instances and Multi-Instance GPU (MIG)

Picking the right GPU instance for a model is a capacity planning decision with significant cost implications. Too small and the model won't fit. Too large and VRAM and compute sit idle. Multi-Instance GPU (MIG) adds a third lever: slice one GPU into isolated hardware instances for right-sized multi-tenant serving.

The GPU count on an instance determines which models fit and what parallelism is available. A single H100 (80 GB of High-Bandwidth Memory, HBM) handles ~30B parameters in FP16 with Key-Value (KV) cache headroom, or 70B at INT4. Four GPUs (320 GB) covers 70B in FP16. Eight GPUs (640 GB) can serve 405B. GB200 NVL72, with 13.5 TB across 72 Blackwell GPUs, handles any open model available today, and the forthcoming Vera Rubin NVL72 lifts that ceiling further with 288 GB of HBM4 per GPU.

Interconnect bandwidth — not GPU count — determines whether Tensor Parallelism (TP) is viable. NVLink inside a node runs at 900 GB/s on H100 and 1.8 TB/s on B200, completing the per-layer all-reduce in microseconds. InfiniBand between nodes tops out around 50 GB/s, an order of magnitude slower. For a Llama 70B layer at batch 16, the all-reduce costs ~4 μs over NVLink vs ~80 μs over InfiniBand — across 80 layers, 6.4 ms per token on communication alone. TP beyond one NVLink domain (typically 8 GPUs) rarely pays off; use pipeline parallelism or replicas instead.

MIG partitions one GPU into isolated hardware slices, each with a dedicated share of SMs, HBM capacity, and L2 cache. An H100 can be split into up to 7 independent `1g.10gb` instances, 3 `2g.20gb` slices, 2 `3g.40gb` slices, or left as a single `7g.80gb` full GPU. Processes on one MIG instance cannot touch the memory or compute of another — it is hardware isolation, not software namespacing. That makes MIG ideal for multi-tenant inference with per-tenant performance guarantees, or for packing several small models (7B at FP16 ≈ 14 GB) onto one physical GPU without stepping on each other.

The choice is usually a layering exercise: pick the instance size that holds the model and its KV cache with headroom; use NVLink domains for TP and cross-node links for replicas or pipeline stages; use MIG when one GPU holds more capacity than any single tenant needs.

Walkthrough with an instance-fitting table, a TP-vs-interconnect all-reduce model, and H100 MIG partition sizing: [`multi-gpu-instances-mig.ipynb`](./multi-gpu-instances-mig.ipynb). Next up: Containerization — Docker and NVIDIA NIMs (Inference Microservices).

Inspired by *Inference Engineering* (Philip Kiely, Baseten Books, 2026). Further reading: [NVIDIA Multi-Instance GPU User Guide](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/) and [NVIDIA GB200 NVL72 architecture](https://www.nvidia.com/en-us/data-center/gb200-nvl72/).

#LLM #Inference #MIG #MultiGPU #TensorParallelism #NVLink #H100 #DeepLearning #AI #MLEngineering #100DaysOfInference
