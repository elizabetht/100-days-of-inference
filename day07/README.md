# Day 07 — CUDA Kernels and Kernel Fusion

Every PyTorch op dispatches a CUDA (Compute Unified Device Architecture) kernel — a function that runs in parallel across thousands of GPU threads. The hidden cost: back-to-back ops write intermediates to HBM (High Bandwidth Memory — the GPU's main off-chip DRAM) and read them back for the next op. Decode is already memory-bound, so every unnecessary HBM round-trip compounds the bottleneck.

Kernel fusion eliminates those intermediate writes. A fused SwiGLU kernel computes the gate, activation, and elementwise multiply in one pass — intermediates stay in registers or L1 cache, never touching HBM. For a typical FFN layer (d_model=4096, ffn_dim=11008), the unfused path reads and writes three separate tensors; the fused path reads once and writes once. At 3.35 TB/s, eliminating two HBM round-trips saves measurable microseconds per token.

Launch overhead stacks on top. Each kernel dispatch costs ~10-50μs. A single transformer layer involves dozens of operations, and at 200 tokens/second, that overhead accumulates across every decode step. Fusion removes launches, not just memory traffic.

FlashAttention is the canonical example. Standard attention materializes a full T×T matrix in HBM. Flash tiles the computation in on-chip SRAM — no intermediate matrix ever hits global memory. Mathematically identical, 2-4× lower memory traffic for long sequences. The reason this works comes down to SRAM vs DRAM: SRAM is on-chip, tiny (~20 MB shared memory/L1 per GPU on A100), and extremely fast (~19 TB/s aggregate). DRAM (where HBM lives) is off-chip, large (80 GB on A100), and ~5-6× slower at 3.35 TB/s. SRAM holds data in flip-flop circuits with no refresh needed; DRAM stores bits as capacitor charges that leak and must be refreshed millions of times per second. SRAM costs ~1000× more per bit, which is why GPUs have so little of it — and why FlashAttention's tiling strategy, which keeps the working set in SRAM, delivers such outsized gains.

The framework implication: vLLM and SGLang ship fused kernels for attention, LayerNorm, and activation functions. TensorRT-LLM goes further with graph-level fusion, auto-selecting the best fused kernel per tensor shape. The gains stack.

The notebook ([cuda-kernels.ipynb](https://github.com/elizabetht/100-days-of-inference/tree/main/day07/cuda-kernels.ipynb)) demonstrates kernel dispatch, measures memory round-trip costs for unfused vs fused FFN patterns, and benchmarks kernel launch overhead under realistic workloads.

#CUDAKernels #KernelFusion #FlashAttention #LLMInference #GPUOptimization #100DaysOfInference
