# Day 19

**Topic:** Multi-GPU Instances & Multi-Instance GPU (MIG)
**Date:** 2026-04-06
**Layer:** Infrastructure

## What I explored
Explored MIG partitioning strategies for H100, mapped model sizes to MIG instance profiles, and compared MIG vs MPS vs time-sharing for multi-tenant inference workloads.

## Key insight
MIG's key advantage over software multi-tenancy: a noisy neighbor in a different MIG slice literally cannot steal your GPU resources — the memory bandwidth and compute engines are physically separate.

## Code / experiment
Notebook: [`mig-multi-instance-gpu.ipynb`](./mig-multi-instance-gpu.ipynb)
Key demo: Model-to-MIG-profile fit matrix + multi-tenancy strategy comparison

## References
- *Inference Engineering* Ch 3.3 (Philip Kiely, Baseten Books 2026)
- NVIDIA MIG User Guide
