# Day 24 — Routing, Load Balancing & Queueing

A Large Language Model (LLM) request can carry 10 input tokens or 100,000. Round-robin treats them as equal and lands a 100K-token prefill on the same replica as a chat turn — turning the chat turn into a thirty-second wait. Once a fleet has more than one replica, the routing layer becomes one of the highest-leverage knobs in the stack.

Routers and load balancers split that layer cleanly. Routers make per-request decisions ("where should this request go?"). Load balancers make system-level decisions ("where could this request go?"). Real systems chain both: a load balancer picks healthy candidates, a router picks among them using state the load balancer never sees — sequence length, prefix-cache contents, and Low-Rank Adaptation (LoRA) adapter status.

Token-aware routing is the simplest fix: track in-flight tokens per replica, not connections. On a heterogeneous workload (1,000 requests, four replicas), per-replica token load goes from Coefficient of Variation 0.07 under round-robin to 0.005 under token-aware — within 0.5% of the mean.

Key-Value (KV) cache-aware routing is the highest-leverage one. Chat workloads share prefixes — a 500-token system prompt may serve millions of conversations. A replica that already holds those tokens skips a 100 ms prefill for near-zero cost. Hash the prefix to a preferred replica, fall back when overloaded. Hash-based routing lifts fleet-wide hit rate from 9% to 31% — round-robin scatters the same prefix across replicas and each one thrashes its cache. NVIDIA Dynamo productionizes this. The same locality principle applies to LoRA adapters, eliminating ~200 ms swaps on the hot path.

Queues handle the gap between traffic and capacity. First-In-First-Out (FIFO) is the default; priority queues layer Service Level Agreement (SLA) tiers on top. Every request carries a deadline so stale work gets dropped, not served. Under a 10× burst, premium drains in thirty seconds while batch waits two minutes.

Little's Law (in-flight requests = arrival rate × mean response time) sets the replica floor: 10 requests/sec at a 3-second mean response keeps thirty in flight at all times. The M/M/c model — random arrivals, random service times, c parallel workers, one shared queue — predicts wait time curves up sharply as the fleet fills. At 80% utilization the line is manageable; at 95% tail latency is 5–10× worse, not 1.2×. Size for 70–80% headroom or watch the tail explode.

Walkthrough with three routing simulators, a priority-queue burst test, and the M/M/c utilization curves: [`routing-load-balancing-queueing.ipynb`](./routing-load-balancing-queueing.ipynb). Next up: Multi-Cloud Capacity Management.

Inspired by *Inference Engineering* (Philip Kiely, Baseten Books, 2026). Further reading: [NVIDIA Dynamo](https://github.com/ai-dynamo/dynamo).

#LLM #Inference #LoadBalancing #Routing #KVCache #LittlesLaw #DeepLearning #AI #MLEngineering #100DaysOfInference
