# Day 20 — GPU Generations: Hopper, Ada, Blackwell, Rubin

Picking the wrong NVIDIA GPU generation for an inference deployment is expensive. The differences between generations are not incremental — each introduces new precision formats, interconnect speeds, and architectural features that change which optimizations are even available.

Hopper (H100, 2022) is still the workhorse of production inference. It introduced FP8 Tensor Cores with native hardware support — 3.9 PFLOPS in FP8 versus 1.98 PFLOPS in FP16 (16-bit floating point). The Transformer Engine auto-selects FP8 or FP16 per layer at runtime. NVLink 4.0 at 900 GB/s bidirectional makes tight Tensor Parallelism (TP) within a node practical, and 80 GB of High-Bandwidth Memory (HBM3) at 3.35 TB/s feeds the SMs. TensorRT-LLM and vLLM both expose FP8 paths that lean directly on this hardware.

Blackwell (B100/B200/GB200, 2024) doubles HBM capacity to 192 GB per GPU at 8 TB/s and steps up to NVLink 5.0 at 1.8 TB/s. The headline addition is FP4 Tensor Cores — 2× the throughput of FP8 — compelling for throughput-bound workloads where the accuracy hit is acceptable. The GB200 NVL72 rack connects 72 B200s via NVSwitch at 1.8 TB/s between any pair, making disaggregated prefill/decode architectures practical at rack scale where InfiniBand between nodes was previously the bottleneck.

Ada Lovelace (RTX 40xx, 2022) is the consumer-grade sibling of Hopper. No SXM form factor, PCIe-only interconnect, and a 24 GB ceiling per card. Useful for development and small-scale serving, but not competitive with H100/H200 for production throughput.

The choice is mostly precision and capacity. H100s with FP8 cover most of today's production inference. B200s with FP4 buy 2× more throughput per dollar when the model tolerates 4-bit quantization. Rubin (R100), announced at CES 2026, pushes further: 288 GB of HBM4 at 22 TB/s per GPU (2.75× Blackwell's HBM3e bandwidth), 50 PFLOPS of FP4 compute (3.3× B300), and NVLink 6 at 3.6 TB/s per GPU on a TSMC 3nm dual-die package (336 B transistors). Sampling begins Q4 2026, and the Vera Rubin NVL72 rack — 72 R100s paired with 36 Arm-based Vera CPUs — ships to early customers in H2 2026.

Walkthrough with a Transformer Engine precision-selection simulation, an FP4 E2M1 format analysis, and an HBM-bounded decode-throughput projection across generations: [`gpu-generations-hopper-blackwell.ipynb`](./gpu-generations-hopper-blackwell.ipynb). Next up: Multi-Instance GPU (MIG) partitioning — slicing one H100 into seven isolated instances.

Inspired by *Inference Engineering* (Philip Kiely, Baseten Books, 2026). Further reading: [NVIDIA Blackwell architecture brief](https://resources.nvidia.com/en-us-blackwell-architecture) and [Hopper FP8 paper](https://arxiv.org/abs/2209.05433) (Micikevicius et al. 2022).

#LLM #Inference #H100 #Blackwell #FP8 #FP4 #GPUGenerations #DeepLearning #AI #MLEngineering #100DaysOfInference
