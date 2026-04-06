# Day 22

**Topic:** Routing, Load Balancing & Queueing
**Date:** 2026-04-06
**Layer:** Infrastructure

## What I explored
Implemented and benchmarked four routing strategies (round-robin, least-connections, random, weighted), built a priority queue for interactive vs batch traffic, and applied Erlang C queueing theory to size worker pools for SLOs.

## Key insight
At 80% utilization, M/M/c queuing theory predicts 5x higher wait times than at 50%. This is why inference services must overprovision — running at 50-70% target utilization is not waste, it's SLO headroom.

## Code / experiment
Notebook: [`routing-load-balancing-queueing.ipynb`](./routing-load-balancing-queueing.ipynb)
Key demo: Routing strategy comparison + Erlang C SLO sizing table

## References
- *Inference Engineering* Ch 7.2.3 (Philip Kiely, Baseten Books 2026)
- Erlang C queueing formula
