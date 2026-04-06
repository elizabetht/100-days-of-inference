# Day 20

**Topic:** Containerization: Docker & NVIDIA NIMs
**Date:** 2026-04-06
**Layer:** Infrastructure

## What I explored
Designed a production-grade inference Dockerfile with optimal layer caching, explored NVIDIA Container Toolkit GPU passthrough, and analyzed NVIDIA NIMs as pre-optimized deployment packages.

## Key insight
Model weights must never be in the Docker image — they're 16-140GB and make images unmaintainable. Mount as volumes, bake the runtime only.

## Code / experiment
Notebook: [`containerization-docker-nims.ipynb`](./containerization-docker-nims.ipynb)
Key demo: Dockerfile layer structure analysis + NIM vs custom container comparison

## References
- *Inference Engineering* Ch 7.1 (Philip Kiely, Baseten Books 2026)
- NVIDIA Container Toolkit documentation
- NVIDIA NIM documentation
