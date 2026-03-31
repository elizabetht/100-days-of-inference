# 100 Days of LLM Inference

A structured deep-dive into inference engineering — from CUDA kernels to multi-cloud autoscaling — built around *[Inference Engineering](https://www.baseten.co/library/inference-engineering/)* by Philip Kiely (Baseten Books, 2026).

Each day is a runnable Jupyter notebook. All experiments run on a home-lab cluster of two NVIDIA DGX Sparks.

---

## What is Inference Engineering?

> *"Doing inference well requires three layers: Runtime, Infrastructure, and Tooling."*
> — Philip Kiely, Inference Engineering

Inference engineering is the discipline of serving generative AI models in production — faster, cheaper, and more reliably. It spans the full stack from CUDA memory layouts to Kubernetes autoscaling policies. This challenge covers all three layers systematically.

---

## The 100-Day Plan

### Phase 1 — Runtime: Single-Instance Optimization (Days 1–16)

Getting the most out of one GPU. This is where most of the leverage lives.

| Day | Topic | Book |
|-----|-------|------|
| [01](./day01/) | LLM Inference Mechanics: Tokenization, KV State, Autoregressive Decoding | Ch 2.2 |
| [02](./day02/) | Transformer Blocks & Attention Deep Dive | Ch 2.2.2–2.2.3 |
| [03](./day03/) | Mixture of Experts (MoE) Routing | Ch 2.2.4 |
| [04](./day04/) | Ops:Byte Ratio & Arithmetic Intensity | Ch 2.4 |
| [05](./day05/) | CUDA Kernels, Kernel Selection & Kernel Fusion | Ch 4.1 |
| [06](./day06/) | PyTorch, Model File Formats, ONNX & TensorRT | Ch 4.2 |
| [07](./day07/) | vLLM: PagedAttention & Continuous Batching | Ch 4.3.1 |
| [08](./day08/) | SGLang: RadixAttention & Structured Outputs | Ch 4.3.2 |
| [09](./day09/) | TensorRT-LLM: Compilation & Plugin System | Ch 4.3.3 |
| [10](./day10/) | NVIDIA Dynamo: Disaggregated Serving | Ch 4.4 |
| [11](./day11/) | Quantization: Number Formats (FP8, INT8, INT4) | Ch 5.1.1 |
| [12](./day12/) | Quantization: GPTQ, AWQ, SmoothQuant | Ch 5.1.2 |
| [13](./day13/) | Speculative Decoding: Draft-Target, Medusa, EAGLE | Ch 5.2 |
| [14](./day14/) | KV Cache: Prefix Caching & Cache-Aware Routing | Ch 5.3 |
| [15](./day15/) | Model Parallelism: Tensor & Expert | Ch 5.4 |
| [16](./day16/) | Disaggregation: Prefill/Decode Split | Ch 5.5 |

### Phase 2 — Infrastructure: Scaling Across Clusters (Days 17–24)

Getting the most out of many GPUs across clouds and regions.

| Day | Topic | Book |
|-----|-------|------|
| [17](./day17/) | GPU Architecture: SMs, Memory Hierarchy, HBM | Ch 3.1 |
| [18](./day18/) | GPU Generations: Hopper, Ada, Blackwell, Rubin | Ch 3.2 |
| [19](./day19/) | Multi-GPU Instances & Multi-Instance GPU (MIG) | Ch 3.3 |
| [20](./day20/) | Containerization: Docker & NVIDIA NIMs | Ch 7.1 |
| [21](./day21/) | Autoscaling: Concurrency, Batching & Cold Starts | Ch 7.2 |
| [22](./day22/) | Routing, Load Balancing & Queueing | Ch 7.2.3 |
| [23](./day23/) | Multi-Cloud Capacity Management | Ch 7.3 |
| [24](./day24/) | Zero-Downtime Deployment & Cost Estimation | Ch 7.4 |

### Phase 3 — Tooling: Productivity & Observability (Days 25–27)

The instrumentation layer that makes the other two debuggable.

| Day | Topic | Book |
|-----|-------|------|
| [25](./day25/) | Performance Benchmarking: Tooling & Profiling | Ch 4.5 |
| [26](./day26/) | Observability: Metrics, Tracing & Dashboards | Ch 7.4.3 |
| [27](./day27/) | Client Code: Streaming, Async & Protocol Support | Ch 7.5 |

---

### Phase 4 — Deep Implementation: Build It from Scratch (Days 28–50)

The book explains the concepts. Now implement them.

| Day | Project |
|-----|---------|
| 28 | Implement a BPE tokenizer from scratch |
| 29 | Build a bare autoregressive decoder loop in PyTorch |
| 30 | Implement scaled dot-product attention (SDPA) with masking |
| 31 | Implement Flash Attention (simplified, tiling in Python) |
| 32 | Profile attention memory growth across sequence lengths |
| 33 | Build an INT8 quantization pipeline: quantize → dequantize → measure error |
| 34 | Implement GPTQ-style round-to-nearest with Hessian weighting |
| 35 | Sweep quantization bit widths and plot perplexity vs compression |
| 36 | Simulate draft-target speculative decoding with acceptance sampling |
| 37 | Build a simple KV cache manager (block allocator, eviction policy) |
| 38 | Implement prefix caching with hash-based deduplication |
| 39 | Simulate tensor parallelism: split a matmul across N workers |
| 40 | Benchmark ops:byte ratio in practice across matrix sizes |
| 41 | CUDA profiling: profile a PyTorch model with `torch.profiler` |
| 42 | Write a custom elementwise CUDA kernel via Triton |
| 43 | Build a PyTorch custom op with CUDA backend |
| 44 | Deploy vLLM on spark-01, benchmark TTFT and throughput |
| 45 | Deploy SGLang, benchmark structured output latency |
| 46 | TensorRT-LLM: compile a model and compare with eager PyTorch |
| 47 | NVIDIA Dynamo: run a disaggregated prefill experiment |
| 48 | Simulate continuous batching: queue arrivals, dynamic batch formation |
| 49 | Visualize PagedAttention block layout and fragmentation |
| 50 | Benchmark TTFT vs throughput tradeoff across batch sizes |

---

### Phase 5 — Production Systems: From Notebook to Cluster (Days 51–75)

Ship it.

| Day | Project |
|-----|---------|
| 51 | Write a production Dockerfile for a vLLM inference server |
| 52 | Build and push a NIM-compatible container |
| 53 | Simulate an autoscaling policy: requests per second → replica count |
| 54 | Measure cold start latency: model load times at different sizes |
| 55 | Implement round-robin and least-connections load balancers |
| 56 | Build a priority request queue with batch formation |
| 57 | Multi-GPU tensor parallel benchmark across spark-01 and spark-02 |
| 58 | Configure MIG on a Spark GPU: profile different partition sizes |
| 59 | GPU cost model: $/token across instance types at different utilizations |
| 60 | Blue-green deployment: zero-downtime model version swap |
| 61 | Emit Prometheus metrics from an inference server |
| 62 | Build a Grafana dashboard: TTFT, TBT, queue depth, GPU utilization |
| 63 | Add distributed tracing (OpenTelemetry) to an inference request |
| 64 | Load test with Locust: ramp traffic, find saturation point |
| 65 | Profile with Nsight Systems: identify kernel launch overhead |
| 66 | Build a streaming inference client using SSE |
| 67 | Async batch inference client using `asyncio` + `aiohttp` |
| 68 | Multi-cloud routing: geo-aware latency-based request routing |
| 69 | GPU memory profiling: find where your memory budget goes |
| 70 | Benchmark quantization levels on real throughput: FP16 vs INT8 vs INT4 |
| 71 | Measure speculative decoding acceptance rates by draft model size |
| 72 | Measure KV cache hit rates across real traffic patterns |
| 73 | Tensor parallelism scaling: throughput and latency vs GPU count |
| 74 | End-to-end latency breakdown: tokenization → TTFT → TBT → detokenization |
| 75 | Build a reusable inference benchmark harness |

---

### Phase 6 — Modalities: Beyond Text (Days 76–85)

The book covers vision, audio, and video. Inference engineering applies to all of them.

| Day | Topic | Book |
|-----|-------|------|
| 76 | Vision Language Model (VLM) inference: image preprocessing and batching | Ch 6.1 |
| 77 | Embedding model inference: batching and throughput optimization | Ch 6.2 |
| 78 | ASR (Whisper): single-chunk and long-file latency optimization | Ch 6.3 |
| 79 | TTS: streaming real-time text-to-speech | Ch 6.4 |
| 80 | Image generation: diffusion model inference and kernel optimization | Ch 6.5 |
| 81 | Video generation: context parallelism and attention optimization | Ch 6.6 |
| 82 | Multi-modal batching: mixing text and image requests | Ch 6.1–6.2 |
| 83 | Embedding similarity search pipeline: embed → index → query | Ch 6.2 |
| 84 | Speech-to-speech pipeline: ASR → LLM → TTS end-to-end latency | Ch 6.3–6.4 |
| 85 | Long context: RoPE scaling, context parallelism across GPUs | Ch 5.3.4 |

---

### Phase 7 — Advanced Techniques (Days 86–95)

The frontier of inference research, made practical.

| Day | Topic |
|-----|-------|
| 86 | EAGLE speculative decoding: feature-level draft vs token-level |
| 87 | Medusa: multi-head speculative decoding, measure speedup |
| 88 | MoE routing from scratch: top-K gating, load balancing loss |
| 89 | Expert parallelism: simulate routing across N expert shards |
| 90 | Dynamic disaggregation with NVIDIA Dynamo |
| 91 | Cache-aware routing: route requests to maximize KV cache hits |
| 92 | Long context without context parallelism: chunked prefill |
| 93 | Fine-tuning a small model for inference quality vs a large quantized one |
| 94 | Distillation for inference: teacher-student latency/quality tradeoffs |
| 95 | Intelligence evaluation: build an eval harness for a deployed model |

---

### Phase 8 — Capstone: A Production Inference Stack (Days 96–100)

Build something real.

| Day | Capstone Task |
|-----|---------------|
| 96 | Design: sketch the full inference stack for a real use case |
| 97 | Build: FastAPI + vLLM inference server with health checks and metrics |
| 98 | Deploy: ship it to the home lab cluster with load balancing |
| 99 | Optimize: run the benchmark harness, find the bottleneck, fix it |
| 100 | Reflect: what I learned, what I'd do differently, what's next |

---

## Setup

**Hardware:** Two NVIDIA DGX Sparks (spark-01: `192.168.1.76`, spark-02: `192.168.1.77`)

**Each notebook is self-contained.** Run any day independently:

```bash
ssh nvidia@192.168.1.76
cd ~/src/github.com/elizabetht/100-days-of-llm-inference/dayNN
jupyter notebook
```

**Generate notebooks** with the Claude Code skill:

```bash
/learn-inference-eng next        # generate the next notebook
/learn-inference-eng 7           # jump to Day 07: vLLM
/learn-inference-eng quantization # fuzzy-match to Day 11
```

---

## Progress

| Phase | Days | Status |
|-------|------|--------|
| Runtime Layer | 1–16 | 0 / 16 |
| Infrastructure Layer | 17–24 | 0 / 8 |
| Tooling Layer | 25–27 | 0 / 3 |
| Deep Implementation | 28–50 | 0 / 23 |
| Production Systems | 51–75 | 0 / 25 |
| Modalities | 76–85 | 0 / 10 |
| Advanced Techniques | 86–95 | 0 / 10 |
| Capstone | 96–100 | 0 / 5 |
| **Total** | **1–100** | **0 / 100** |

---

## Reference

- **Book:** *Inference Engineering* — Philip Kiely (Baseten Books, 2026)
- **Cluster:** spark-01 `192.168.1.76` · spark-02 `192.168.1.77`
- **Start:** 2026-03-31
