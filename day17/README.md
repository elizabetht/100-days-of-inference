# Day 17

**Topic:** GPU Architecture: SMs, Memory Hierarchy, HBM
**Date:** 2026-04-06
**Layer:** Infrastructure

## What I explored
Mapped the H100's memory hierarchy from registers to HBM, measured achievable HBM bandwidth on the target GPU, and analyzed SM occupancy limits from threads, registers, and shared memory constraints.

## Key insight
HBM on-package memory is the defining LLM inference hardware feature: at 3.35 TB/s, it's 33x faster than CPU DRAM — but it's still the bottleneck for memory-bound decode.

## Code / experiment
Notebook: [`gpu-architecture-sms-hbm.ipynb`](./gpu-architecture-sms-hbm.ipynb)
Key demo: HBM bandwidth measurement + SM occupancy analysis across kernel configurations

## References
- *Inference Engineering* Ch 3.1 (Philip Kiely, Baseten Books 2026)
- NVIDIA H100 Architecture White Paper
