# Day 46

**Topic:** Simulate Continuous Batching
**Date:** 2026-04-06
**Layer:** Implementation

## What I explored
Simulated continuous batching with 100 requests, measured GPU utilization improvement vs static batching, and visualized the utilization difference.

## Key insight
The head-of-line blocking problem in static batching: a single long request (1000 tokens) holds the entire batch hostage, starving 7 other users. Continuous batching routes around this by scheduling at iteration granularity.

## Code / experiment
Notebook: [`continuous-batching-simulation.ipynb`](./continuous-batching-simulation.ipynb)
Key demo: GPU utilization comparison: static vs continuous batching

## References
- *Inference Engineering* (Philip Kiely, Baseten Books 2026)
