# Day 15

**Topic:** Model Parallelism: Tensor & Expert
**Date:** 2026-04-06
**Layer:** Runtime

## What I explored
Implemented column-parallel and row-parallel linear layers simulating tensor parallelism across N GPUs. Analyzed AllReduce communication cost on NVLink vs InfiniBand vs PCIe, and simulated expert parallelism routing for MoE models.

## Key insight
Tensor parallelism's efficiency entirely depends on interconnect: at 900 GB/s NVLink the AllReduce overhead is negligible; at 25 GB/s InfiniBand it dominates for large TP degrees. This is why TP is intra-node and PP is inter-node.

## Code / experiment
Notebook: [`model-parallelism.ipynb`](./model-parallelism.ipynb)
Key demo: AllReduce cost vs TP size + expert parallelism communication analysis

## References
- *Inference Engineering* Ch 5.4 (Philip Kiely, Baseten Books 2026)
- Shoeybi et al. (2019), "Megatron-LM"
