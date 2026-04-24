# Day 18 — Disaggregation: Prefill/Decode Split

Prefill and decode are fundamentally different workloads. Prefill processes the entire prompt in one forward pass — it's compute-bound, high arithmetic intensity, GPU utilization near 100% for the duration. Decode generates one token at a time — it's memory-bound, low batch size, bottlenecked by High-Bandwidth Memory (HBM) bandwidth not FLOPS (floating-point operations per second). Running both on the same GPU means each one compromises the other's optimal operating point.

Disaggregation splits them onto separate hardware. A prefill worker receives a new request, runs the forward pass over the prompt, produces the key-value (KV) cache. It then transfers the KV cache to a decode worker via a high-bandwidth interconnect (NVIDIA Inference Xfer Library, NIXL, in Dynamo's case — NVLink or InfiniBand depending on topology). The decode worker holds the KV in HBM and generates tokens autoregressively until the sequence completes.

The KV transfer is the hard part. For a 70B model at 32K context in FP16: 2 × 80 layers × 8 KV heads × 128 head_dim × 32768 tokens × 2 bytes ≈ 10 GB per request. At NVLink bandwidth of 900 GB/s, this takes ~11ms. At InfiniBand (50 GB/s), it takes ~200ms. The architecture only makes sense if the compute time saved on the decode worker exceeds the transfer cost.

The economics work when prefill requests are long (high compute load that stalls decode), decode runs at scale (enough concurrent sequences to keep decode workers fully saturated), and the interconnect is fast enough that transfer cost is dominated by prefill time. GB200 NVL72 rack systems are designed precisely for this split.

Full walkthrough with runnable prefill/decode latency models and xPyD traffic simulation: [`disaggregation-prefill-decode.ipynb`](./disaggregation-prefill-decode.ipynb). Next up: GPU architecture — Streaming Multiprocessors, memory hierarchy, and how HBM bandwidth shapes every decision above.

Inspired by *Inference Engineering* (Philip Kiely, Baseten Books, 2026). Further reading on disaggregated serving: [NVIDIA Dynamo](https://docs.nvidia.com/dynamo/) and DistServe (Zhong et al. 2024).

#LLM #Inference #DisaggregatedServing #PrefillDecode #KVTransfer #NVIDIADynamo #NIXL #DeepLearning #AI #MLEngineering #100DaysOfInference
