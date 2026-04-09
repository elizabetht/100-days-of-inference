# Day 06 — Ops:Byte Ratio and the Roofline Model

Every GPU has two ceilings: compute (FLOPS/s) and memory bandwidth (bytes/s). The ops:byte ratio is compute divided by bandwidth ie (FLOPS/bytes per second). An H100 SXM does 989 TFLOPS of BF16 math. Its memory bandwidth is 3.35 TB/s. Dividing one by the other: 989 TFLOPS / 3.35 TB/s ≈ 295 ops/byte. That ratio is the break-even between two different bottlenecks — and it explains why a 989 TFLOPS GPU can feel slow during inference. 

In the time it takes to read one byte from HBM, the compute units can execute 295 ops. An operation needing fewer than 295 ops per byte is memory-bound: compute sits idle waiting for data. More than 295, and compute becomes the ceiling.

LLM inference hits both sides. Prefill processes hundreds of tokens in parallel — high arithmetic intensity, compute-bound. Decode produces one token: Q is a single row, but K and V span the entire cached history. At sequence length 4096 with head dimension 128, attention arithmetic intensity is ~31 ops/byte. The H100 offers 295. Compute finishes and waits ~89% of the time. Decode is deeply memory-bound.

Batching is the lever. At batch=1, full model weights load to produce one token — ~2 ops/byte. At batch=8, the same weight load produces eight tokens: 8x compute, same memory traffic. By batch=64, the operation crosses into compute-bound territory. This is why serving frameworks optimize batch construction above everything else.

The notebook (https://github.com/elizabetht/100-days-of-inference/blob/main/day06/ops-byte-ratio.ipynb) plots roofline charts for H100, A100, RTX 4090, and DGX Spark, sweeps batch sizes to find the memory-to-compute crossover, and benchmarks real matrix multiplies to verify where theory meets measurement.

Identifying the bottleneck takes five minutes of arithmetic. Optimizing the wrong resource takes weeks.

#OpsbyteRatio #RooflineModel #LLMInference #MemoryBound #ArithmeticIntensity #InferenceEngineering #100DaysOfInference