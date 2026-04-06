# Day 44

**Topic:** Deploy vLLM, Benchmark TTFT and Throughput
**Date:** 2026-04-06
**Layer:** Implementation

## What I explored
Configured vLLM deployment for different model size classes, simulated async benchmark with 100 concurrent requests, measuring TTFT P50/P99 and token throughput.

## Key insight
GPU memory utilization is the key vLLM knob: setting it too high causes OOM on long requests; too low wastes KV cache capacity. 0.85-0.90 is the standard starting point.

## Code / experiment
Notebook: [`vllm-deployment-benchmark.ipynb`](./vllm-deployment-benchmark.ipynb)
Key demo: vLLM configuration guide + async benchmark simulation

## References
- *Inference Engineering* (Philip Kiely, Baseten Books 2026)
