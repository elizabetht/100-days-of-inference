# 100 Days of LLM Inference

A structured deep-dive into inference engineering — from CUDA kernels to multi-cloud autoscaling — built around *[Inference Engineering](https://www.baseten.co/library/inference-engineering/)* by Philip Kiely (Baseten Books, 2026).

Each entry is a runnable script. All experiments run on a home-lab cluster of two NVIDIA DGX Sparks.

---

## What is Inference Engineering?

> *"Doing inference well requires three layers: Runtime, Infrastructure, and Tooling."*
> — Philip Kiely, Inference Engineering

Inference engineering is the discipline of serving generative AI models in production — faster, cheaper, and more reliably. It spans the full stack from CUDA memory layouts to Kubernetes autoscaling policies. This challenge covers all three layers systematically.

---

## The Plan

### Phase 1 — Runtime: Single-Instance Optimization

Getting the most out of one GPU. This is where most of the leverage lives.

| Day | Topic | Book |
|-----|-------|------|
| [01](./day01/) | LLM Inference Mechanics: End-to-end text generation | Ch 2.2 |
| [02](./day02/) | Inference from Scratch: Model internals & tokenization | Ch 2.2 |
| [03](./day03/) | Embeddings: From integers to vectors | Ch 2.2.1 |
| [04](./day04/) | Transformer Blocks & Attention Deep Dive | Ch 2.2.2–2.2.3 |
| [05](./day05/) | KV Cache | Ch 2.2 |
| [06](./day06/) | Ops:Byte Ratio & Arithmetic Intensity | Ch 2.4 |
| [07](./day07/) | CUDA Kernels, Kernel Selection & Kernel Fusion | Ch 4.1 |
| [08](./day08/) | PyTorch, Model File Formats, ONNX & TensorRT | Ch 4.2 |
| [09](./day09/) | vLLM: PagedAttention & Continuous Batching | Ch 4.3.1 |
| [10](./day10/) | SGLang: RadixAttention & Structured Outputs | Ch 4.3.2 |
| [11](./day11/) | TensorRT-LLM: Compilation & Plugin System | Ch 4.3.3 |
| 12 | NVIDIA Dynamo: Disaggregated Serving | Ch 4.4 |
| 13 | Quantization: Number Formats (FP8, INT8, INT4) | Ch 5.1.1 |
| 14 | Quantization: GPTQ, AWQ, SmoothQuant | Ch 5.1.2 |
| 15 | Speculative Decoding: Draft-Target, Medusa, EAGLE | Ch 5.2 |
| 16 | KV Cache: Prefix Caching & Cache-Aware Routing | Ch 5.3 |
| 17 | Model Parallelism: Tensor & Expert | Ch 5.4 |
| 18 | Disaggregation: Prefill/Decode Split | Ch 5.5 |

### Phase 2 — Infrastructure: Scaling Across Clusters

Getting the most out of many GPUs across clouds and regions.

| Topic | Book |
|-------|------|
| [GPU Architecture: SMs, Memory Hierarchy, HBM](./day17/) | Ch 3.1 |
| [GPU Generations: Hopper, Ada, Blackwell, Rubin](./day18/) | Ch 3.2 |
| [Multi-GPU Instances & Multi-Instance GPU (MIG)](./day19/) | Ch 3.3 |
| [Containerization: Docker & NVIDIA NIMs](./day20/) | Ch 7.1 |
| [Autoscaling: Concurrency, Batching & Cold Starts](./day21/) | Ch 7.2 |
| [Routing, Load Balancing & Queueing](./day22/) | Ch 7.2.3 |
| [Multi-Cloud Capacity Management](./day23/) | Ch 7.3 |
| [Zero-Downtime Deployment & Cost Estimation](./day24/) | Ch 7.4 |

### Phase 3 — Tooling: Productivity & Observability

The instrumentation layer that makes the other two debuggable.

| Topic | Book |
|-------|------|
| [Performance Benchmarking: Tooling & Profiling](./day25/) | Ch 4.5 |
| [Observability: Metrics, Tracing & Dashboards](./day26/) | Ch 7.4.3 |
| [Client Code: Streaming, Async & Protocol Support](./day27/) | Ch 7.5 |

---

### Phase 4 — Deep Implementation: Build It from Scratch

The book explains the concepts. Now implement them.

| Project |
|---------|
| Implement a BPE tokenizer from scratch |
| Build a bare autoregressive decoder loop in PyTorch |
| Implement scaled dot-product attention (SDPA) with masking |
| Implement Flash Attention (simplified, tiling in Python) |
| Profile attention memory growth across sequence lengths |
| Build an INT8 quantization pipeline: quantize → dequantize → measure error |
| Implement GPTQ-style round-to-nearest with Hessian weighting |
| Sweep quantization bit widths and plot perplexity vs compression |
| Simulate draft-target speculative decoding with acceptance sampling |
| Build a simple KV cache manager (block allocator, eviction policy) |
| Implement prefix caching with hash-based deduplication |
| Simulate tensor parallelism: split a matmul across N workers |
| Benchmark ops:byte ratio in practice across matrix sizes |
| CUDA profiling: profile a PyTorch model with `torch.profiler` |
| Write a custom elementwise CUDA kernel via Triton |
| Build a PyTorch custom op with CUDA backend |
| Deploy vLLM on spark-01, benchmark TTFT and throughput |
| Deploy SGLang, benchmark structured output latency |
| TensorRT-LLM: compile a model and compare with eager PyTorch |
| NVIDIA Dynamo: run a disaggregated prefill experiment |
| Simulate continuous batching: queue arrivals, dynamic batch formation |
| Visualize PagedAttention block layout and fragmentation |
| Benchmark TTFT vs throughput tradeoff across batch sizes |

---

### Phase 5 — Production Systems: From Notebook to Cluster

Ship it.

| Project |
|---------|
| Write a production Dockerfile for a vLLM inference server |
| Build and push a NIM-compatible container |
| Simulate an autoscaling policy: requests per second → replica count |
| Measure cold start latency: model load times at different sizes |
| Implement round-robin and least-connections load balancers |
| Build a priority request queue with batch formation |
| Multi-GPU tensor parallel benchmark across spark-01 and spark-02 |
| Configure MIG on a Spark GPU: profile different partition sizes |
| GPU cost model: $/token across instance types at different utilizations |
| Blue-green deployment: zero-downtime model version swap |
| Emit Prometheus metrics from an inference server |
| Build a Grafana dashboard: TTFT, TBT, queue depth, GPU utilization |
| Add distributed tracing (OpenTelemetry) to an inference request |
| Load test with Locust: ramp traffic, find saturation point |
| Profile with Nsight Systems: identify kernel launch overhead |
| Build a streaming inference client using SSE |
| Async batch inference client using `asyncio` + `aiohttp` |
| Multi-cloud routing: geo-aware latency-based request routing |
| GPU memory profiling: find where your memory budget goes |
| Benchmark quantization levels on real throughput: FP16 vs INT8 vs INT4 |
| Measure speculative decoding acceptance rates by draft model size |
| Measure KV cache hit rates across real traffic patterns |
| Tensor parallelism scaling: throughput and latency vs GPU count |
| End-to-end latency breakdown: tokenization → TTFT → TBT → detokenization |
| Build a reusable inference benchmark harness |

---

### Phase 6 — Modalities: Beyond Text

The book covers vision, audio, and video. Inference engineering applies to all of them.

| Topic | Book |
|-------|------|
| Vision Language Model (VLM) inference: image preprocessing and batching | Ch 6.1 |
| Embedding model inference: batching and throughput optimization | Ch 6.2 |
| ASR (Whisper): single-chunk and long-file latency optimization | Ch 6.3 |
| TTS: streaming real-time text-to-speech | Ch 6.4 |
| Image generation: diffusion model inference and kernel optimization | Ch 6.5 |
| Video generation: context parallelism and attention optimization | Ch 6.6 |
| Multi-modal batching: mixing text and image requests | Ch 6.1–6.2 |
| Embedding similarity search pipeline: embed → index → query | Ch 6.2 |
| Speech-to-speech pipeline: ASR → LLM → TTS end-to-end latency | Ch 6.3–6.4 |
| Long context: RoPE scaling, context parallelism across GPUs | Ch 5.3.4 |

---

### Phase 7 — Advanced Techniques

The frontier of inference research, made practical.

| Topic |
|-------|
| EAGLE speculative decoding: feature-level draft vs token-level |
| Medusa: multi-head speculative decoding, measure speedup |
| MoE routing from scratch: top-K gating, load balancing loss |
| Expert parallelism: simulate routing across N expert shards |
| Dynamic disaggregation with NVIDIA Dynamo |
| Cache-aware routing: route requests to maximize KV cache hits |
| Long context without context parallelism: chunked prefill |
| Fine-tuning a small model for inference quality vs a large quantized one |
| Distillation for inference: teacher-student latency/quality tradeoffs |
| Intelligence evaluation: build an eval harness for a deployed model |

---

### Phase 8 — Capstone: A Production Inference Stack

Build something real.

| Capstone Task |
|---------------|
| Design: sketch the full inference stack for a real use case |
| Build: FastAPI + vLLM inference server with health checks and metrics |
| Deploy: ship it to the home lab cluster with load balancing |
| Optimize: run the benchmark harness, find the bottleneck, fix it |
| Reflect: what I learned, what I'd do differently, what's next |

---

## Setup

**Hardware:** Two NVIDIA DGX Sparks (spark-01: `192.168.1.76`, spark-02: `192.168.1.77`)

**Each notebook is self-contained.** Run any topic independently:

```bash
ssh nvidia@192.168.1.76
cd ~/src/github.com/elizabetht/100-days-of-inference/dayNN
jupyter notebook
```

**Generate notebooks** with the Claude Code skill:

```bash
/learn-inference-eng next        # generate the next notebook
/learn-inference-eng 7           # jump to topic 07: vLLM
/learn-inference-eng quantization # fuzzy-match to topic 11
```

---

## Progress

| Phase | Status |
|-------|--------|
| Runtime Layer | 11 / 18 |
| Infrastructure Layer | 0 / 8 |
| Tooling Layer | 0 / 3 |
| Deep Implementation | 0 / 23 |
| Production Systems | 0 / 25 |
| Modalities | 0 / 10 |
| Advanced Techniques | 0 / 10 |
| Capstone | 0 / 5 |
| **Total** | **11 / 102** |

---

## Reference

- **Book:** *Inference Engineering* — Philip Kiely (Baseten Books, 2026)
- **Cluster:** spark-01 `192.168.1.76` · spark-02 `192.168.1.77`
- **Start:** 2026-03-31
