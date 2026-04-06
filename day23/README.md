# Day 23

**Topic:** Multi-Cloud Capacity Management
**Date:** 2026-04-06
**Layer:** Infrastructure

## What I explored
Built a multi-cloud capacity model with on-prem, AWS, GCP, Azure, and Lambda pools. Implemented SLO-aware routing policies and demonstrated the hybrid on-prem/cloud cost optimization strategy on a realistic 24-hour traffic profile.

## Key insight
On-prem GPUs as base load with cloud bursting is ~40-60% cheaper than pure cloud for predictable workloads — the key insight is that hardware amortization breaks the linear cost scaling of pure cloud.

## Code / experiment
Notebook: [`multi-cloud-capacity.ipynb`](./multi-cloud-capacity.ipynb)
Key demo: Hybrid on-prem/cloud capacity allocation + daily cost comparison

## References
- *Inference Engineering* Ch 7.3 (Philip Kiely, Baseten Books 2026)
- Multi-cloud inference routing strategies
