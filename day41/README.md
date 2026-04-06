# Day 41

**Topic:** CUDA Profiling with torch.profiler
**Date:** 2026-04-06
**Layer:** Implementation

## What I explored
Profiled a 3-layer MLP with torch.profiler, capturing kernel timing, memory, and FLOP counts. Exported Chrome trace and analyzed the key profiling metrics.

## Key insight
The Chrome trace export is the most powerful profiler feature: it shows kernel launch, execution, and memory allocation as a timeline, making CPU-GPU synchronization stalls immediately visible.

## Code / experiment
Notebook: [`cuda-profiling-torch-profiler.ipynb`](./cuda-profiling-torch-profiler.ipynb)
Key demo: torch.profiler kernel timing table + Chrome trace export

## References
- *Inference Engineering* (Philip Kiely, Baseten Books 2026)
