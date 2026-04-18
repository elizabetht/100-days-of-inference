# Day 13 — Quantization: Number Formats (FP8, INT8, INT4, NVFP4)

A 70B parameter model in FP32 occupies 280 GB — four H100s just to hold the weights. In BF16, that's 140 GB. INT8: 70 GB. INT4: 35 GB, fitting on a single H100 with room for KV cache. NVFP4 lands around 40 GB at ~4.5 effective bits per weight, with quality closer to FP8 than INT4. Quantization is not just a compression trick; it's what makes large models economically servable.

The tradeoff is precision loss. FP32 has 23 mantissa bits. FP16 has 10. BF16 has 7 but the same exponent range as FP32, which is why it's preferred for training. INT8 has 7 bits of dynamic range with no floating-point representation. Each reduction cuts the representable space roughly in half. The question is how much accuracy you lose on your downstream task.

FP8 is the current sweet spot for production. Hopper's H100 has native FP8 Tensor Core support: 3.9 petaFLOPS in FP8 vs 1.98 petaFLOPS in FP16 — a 2× throughput gain if you can tolerate the precision reduction. FP8 comes in two formats: E4M3 (4 exponent bits, 3 mantissa, better for activations) and E5M2 (5 exponent, 2 mantissa, better for gradients). Most inference deployments use E4M3 for both weights and activations.

INT4 is the most aggressive widely-used format. GPTQ and AWQ both target INT4 for weights. The quantization error is visible on challenging benchmarks but acceptable for most chatbot workloads. The memory saving — 4× vs FP16 — is compelling enough that nearly all large model deployments at inference scale use some form of 4-bit quantization.

NVFP4 is the next step beyond FP8. Blackwell's native 4-bit floating point (E2M1) with microscaling — every block of 16 values shares an FP8 scale factor, recovering dynamic range that a raw 4-bit format loses. Blackwell Tensor Cores run FP4 GEMMs at 2x FP8 throughput. The effective storage cost is ~4.5 bits per weight (4 bits + amortized scale overhead), landing between INT4's aggressive compression and FP8's quality preservation.

The notebook (https://github.com/elizabetht/100-days-of-inference/tree/main/day13/quantization-number-formats.ipynb) implements FP8, INT8, INT4, and NVFP4 quantization from scratch, measures quantization error per format, and models memory footprint across model sizes from 7B to 405B.

#LLM #Inference #Quantization #FP8 #INT8 #INT4 #NVFP4 #Blackwell #DeepLearning #AI #MLEngineering #100DaysOfInference
