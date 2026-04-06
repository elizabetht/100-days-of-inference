# Day 54

**Topic:** Measure Cold Start Latency
**Date:** 2026-04-06
**Layer:** Production

## What I explored
Modeled cold start latency for model sizes from 1B to 405B parameters. Identified disk loading as the dominant component and evaluated mitigation strategies.

## Key insight
For a 70B model (140GB FP16), disk loading at 3 GB/s takes 47 seconds — this is why warm pools are not optional for latency-sensitive production deployments.

## Code / experiment
Notebook: [`cold-start-latency.ipynb`](./cold-start-latency.ipynb)
Key demo: Cold start time breakdown by model size + mitigation strategy evaluation

## References
- *Inference Engineering* (Philip Kiely, Baseten Books 2026)
