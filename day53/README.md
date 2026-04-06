# Day 53

**Topic:** Simulate an Autoscaling Policy
**Date:** 2026-04-06
**Layer:** Production

## What I explored
Implemented a hysteresis-based autoscaler with separate scale-up and scale-down thresholds and cooldown periods. Simulated 200 time steps with varying concurrency.

## Key insight
Scale-up threshold (90%) must be lower than 100% to leave headroom — if you wait until saturation to scale, queues have already built up. Scale-down needs 2x longer cooldown to prevent thrashing.

## Code / experiment
Notebook: [`autoscaling-policy.ipynb`](./autoscaling-policy.ipynb)
Key demo: Autoscaler replica count vs concurrency time series

## References
- *Inference Engineering* (Philip Kiely, Baseten Books 2026)
