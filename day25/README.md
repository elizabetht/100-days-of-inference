# Day 25 — Multi-Cloud Capacity Management

A high-volume inference fleet runs thousands of Graphics Processing Units (GPUs) — and at that scale, three things break at once. No single cloud has the supply. No single region keeps global users inside their latency budget. And hardware failure becomes routine: Llama 3's training run lost a GPU roughly every three hours over 54 days. Multi-cloud absorbs all three.

True multi-cloud inference is more than running the same image in multiple accounts. The architecture splits into a global control plane (model deployment and global scaling decisions) and per-cloud workload planes (local traffic, in-cluster autoscaling). Workload planes must keep serving traffic if the control plane goes down — the global plane is for orchestration, not the request hot path.

The three GPU supply tiers each have a role. Hyperscalers (AWS, Google Cloud Platform, Azure) bring reliability and breadth. Neoclouds (CoreWeave, Nebius) are GPU-focused with better unit economics. Resellers like SF Compute Company sit on the secondary market — cheapest but most variable. Procurement mixes three mechanisms: Reserved blocks (months to years, 30–60% off), On-demand (full price, fully flexible), and Spot (up to 70% off, pre-emptible on minutes of notice). The playbook: reserved baseline + on-demand buffer + spot for peaks.

Geo-aware load balancing handles the latency side. Network round-trip costs ~5 ms per time zone crossed — New York to San Francisco is 15 ms one way, Singapore to San Francisco is 75 ms. With production Service Level Agreement (SLA) budgets often under 500 ms total, a misrouted request can blow the budget on network alone. A global load balancer matches users to the nearest healthy region; per-cloud workload planes handle local distribution.

Reliability is a posture choice layered on top of redundancy. Active-active runs multiple regions live, so a regional failure is absorbed without cutover. Active-passive keeps a hot standby idle and flips over only when the active plane fails. Multi-cluster also unlocks compliance — workloads requiring SOC 2 Type II or Health Insurance Portability and Accountability Act (HIPAA) coverage can be pinned to compliant providers; data-residency rules can be honored by pinning to specific regions.

Walkthrough with a procurement cost model and a geo-aware routing simulator over five real datacenter locations: [`multi-cloud-capacity.ipynb`](./multi-cloud-capacity.ipynb). Next up: zero-downtime deployment and cost estimation.

Inspired by *Inference Engineering* (Philip Kiely, Baseten Books, 2026). Further reading: [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783).

#LLM #Inference #MultiCloud #GPUProcurement #GeoRouting #Reliability #DeepLearning #AI #MLEngineering #100DaysOfInference
