# Day 24

**Topic:** Zero-Downtime Deployment & Cost Estimation
**Date:** 2026-04-06
**Layer:** Infrastructure

## What I explored
Implemented blue-green and canary deployment strategies with health checks and automatic rollback on regression. Built a $/token cost model across GPU types and providers, showing how utilization and amortization dominate the cost equation.

## Key insight
Canary deployment with automatic rollback is essential: at 1% canary traffic, a regression only affects 1% of users — roll back the instant error rate exceeds threshold, with total impact limited.

## Code / experiment
Notebook: [`zero-downtime-deployment-cost.ipynb`](./zero-downtime-deployment-cost.ipynb)
Key demo: Canary rollout with regression detection + $/1M token cost comparison

## References
- *Inference Engineering* Ch 7.4 (Philip Kiely, Baseten Books 2026)
- Google SRE Book, Chapter 27
