# Day 51

**Topic:** Production Dockerfile for vLLM
**Date:** 2026-04-06
**Layer:** Production

## What I explored
Built a production-grade Dockerfile for vLLM with health checks, non-root user, and entrypoint that blocks traffic until the model is loaded.

## Key insight
The HEALTHCHECK is the key prod requirement: Kubernetes will not route traffic until health checks pass, naturally implementing a warm-up barrier.

## Code / experiment
Notebook: [`production-dockerfile.ipynb`](./production-dockerfile.ipynb)
Key demo: Production Dockerfile structure + entrypoint warmup barrier

## References
- *Inference Engineering* (Philip Kiely, Baseten Books 2026)
