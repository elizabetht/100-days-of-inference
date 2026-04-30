# Day 26 — Zero-Downtime Deployment & Cost Estimation

Models change. New checkpoints, quantization swaps, runtime upgrades, configuration tweaks — and none of it can ship with downtime on a live inference Application Programming Interface (API). Once the deployment problem is solved, the next question is what the hardware actually costs per million tokens.

Three deployment strategies bracket the safety-versus-capacity tradeoff. Blue-green runs two identical fleets — blue is currently serving, green is the new version. Once green passes health checks against a hold-out traffic slice, the load balancer flips weights atomically. Rollback is one weight flip. The cost: 2× Graphics Processing Unit (GPU) capacity during the overlap, typically 15–30 minutes for a careful cutover. Rolling updates replace replicas one at a time — cheapest, but some traffic always lands on the in-flight version during the swap. Canary shifts traffic in steps (1% → 5% → 25% → 100%) with automatic rollback if error rate or p99 latency breaches a threshold. Slowest path, but it catches subtle regressions that only show up at small traffic slices — prompt-format edge cases, numerical drift from a new quantization.

Cost estimation reduces to one formula: $/1M tokens = (GPU $/hr) ÷ (tokens/sec × utilization × 3600) × 10⁶. Three independent levers move that number, and each one can shift it by multiples. GPU choice (H100 vs B200 vs A100) sets the rate. Throughput is set by High-Bandwidth Memory (HBM) bandwidth and quantization — INT4 versus FP16 is roughly a 4× cost reduction per token because each token reads ¼ the weight bytes from HBM. Utilization is the silent killer: idle GPUs cost the same as busy ones. A 70B model on 4× H100 at 30% utilization costs ~2.3× more per token than the same fleet at 70% — a queueing-and-batching problem, not a hardware problem. Quantization and utilization compound: INT4 at 70% utilization is ~12× cheaper per token than FP16 at 30%.

Graceful shutdown closes the loop. When a Kubernetes pod is replaced, the preStop hook deregisters it from the load balancer, in-flight requests drain, and only then does SIGTERM land. Drain timeout should be set to p99 request duration plus buffer. Too short kills long-tail requests mid-stream and produces user-visible errors during every rollout. Too long and rollouts crawl, blocking subsequent deploys.

Walkthrough with blue-green and canary simulators, a quantization-aware cost model, and a graceful-shutdown drain loop: [`zero-downtime-deployment.ipynb`](./zero-downtime-deployment.ipynb). Next up: performance benchmarking and profiling tooling.

Inspired by *Inference Engineering* (Philip Kiely, Baseten Books, 2026). Further reading: [Google SRE — Release Engineering](https://sre.google/sre-book/release-engineering/).

#LLM #Inference #BlueGreen #CanaryDeployment #CostModel #ZeroDowntime #DeepLearning #AI #MLEngineering #100DaysOfInference
