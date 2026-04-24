# Day 19 — GPU Architecture: SMs, Memory Hierarchy, and HBM

All the inference optimizations so far — kernel fusion, tensor parallelism, FlashAttention — reference GPU hardware without explaining it. The hardware is what makes those optimizations legible rather than magical.

An H100 SXM has 132 Streaming Multiprocessors (SMs). Each SM contains Tensor Cores (for General Matrix Multiplication, GEMM), CUDA cores (for scalar ops), and 256 KB of L1/shared memory (Static RAM, SRAM). On-chip SRAM delivers ~30 TB/s — about 9× faster than High-Bandwidth Memory (HBM). The full chip has 80 GB of HBM3 at 3.35 TB/s, shared across all SMs. The L2 cache is 50 MB, shared, at ~10 TB/s.

The pattern is inverse: capacity and bandwidth trade against each other. The biggest tier (HBM) is the slowest, the smallest (registers and per-SM SRAM) is the fastest, and L2 sits between on both axes. Large-and-fast memory is physically expensive, so the chip stages data through progressively faster, smaller tiers.

This hierarchy explains every major optimization. Model weights live in HBM — every forward pass streams gigabytes from the slow tier. FlashAttention tiles attention so Q, K, V blocks stay in 256 KB of on-chip Shared Memory rather than round-tripping to HBM, and the 9× bandwidth gap translates into a 2–4× end-to-end speedup at long sequences where attention dominates. Kernel fusion applies the same logic to other ops — keep intermediates in registers and L1, never spill to HBM. Tensor parallelism spreads weight matrices across GPUs so models that don't fit in a single GPU's 80 GB of HBM still serve. The hardware dictates the architecture.

The warp is the execution unit: 32 threads running in lockstep. Divergent code paths (if/else where different threads take different branches) serialize — half the threads sit idle. This is why LLM kernels avoid conditional logic in inner loops. Thread blocks are groups of warps assigned to a single SM; the SM scheduler switches between them to hide memory latency.

The hierarchy also reframes the spec sheet. "989 TFLOPS" assumes the data pipeline keeps up — otherwise throughput collapses to roughly half. "3.35 TB/s" assumes sequential HBM access — random access is slower. Real throughput is bounded by the weakest link in the memory hierarchy, not the peak spec.

Walkthrough with GPU memory hierarchy models and bandwidth math: [`gpu-architecture-sms-hbm.ipynb`](./gpu-architecture-sms-hbm.ipynb). Next up: GPU generations — Hopper, Ada, Blackwell, Rubin, and what changed between them.

Inspired by *Inference Engineering* (Philip Kiely, Baseten Books, 2026). Further reading: [NVIDIA H100 whitepaper](https://resources.nvidia.com/en-us-tensor-core) and [FlashAttention](https://arxiv.org/abs/2205.14135) (Dao et al. 2022).

#LLM #Inference #GPU #CUDA #HBM #GPUArchitecture #TensorCores #DeepLearning #AI #MLEngineering #100DaysOfInference
