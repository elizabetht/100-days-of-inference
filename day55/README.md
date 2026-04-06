# Day 55

**Topic:** Round-Robin and Least-Connections Load Balancers
**Date:** 2026-04-06
**Layer:** Production

## What I explored
Implemented and benchmarked round-robin and least-connections load balancers under 1000 requests with lognormal duration distribution.

## Key insight
Least-connections outperforms round-robin for heavy-tailed request distributions because it naturally routes away from currently-busy workers — round-robin ignores the fact that some requests take 10x longer than others.

## Code / experiment
Notebook: [`load-balancer-implementation.ipynb`](./load-balancer-implementation.ipynb)
Key demo: Load balancer P99 latency + worker load distribution comparison

## References
- *Inference Engineering* (Philip Kiely, Baseten Books 2026)
