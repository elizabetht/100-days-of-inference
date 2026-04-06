# Day 21

**Topic:** Autoscaling: Concurrency, Batching & Cold Starts
**Date:** 2026-04-06
**Layer:** Infrastructure

## What I explored
Built an inference autoscaler simulation showing why GPU utilization is a poor scaling signal. Modeled cold start latency for different model sizes and implemented a concurrency-based autoscaling policy with scale event tracking.

## Key insight
GPU utilization is nearly binary for LLM inference — it's high whenever any request is being processed. Concurrency and queue depth are the correct scaling signals.

## Code / experiment
Notebook: [`autoscaling-concurrency-cold-starts.ipynb`](./autoscaling-concurrency-cold-starts.ipynb)
Key demo: Concurrency vs GPU utilization metrics + autoscaler response simulation

## References
- *Inference Engineering* Ch 7.2 (Philip Kiely, Baseten Books 2026)
- KEDA autoscaling documentation
