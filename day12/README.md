# Day 12 — NVIDIA Dynamo: Disaggregated Serving

Even with the best inference engine, a fundamental tension remains: prefill and decode compete for the same GPU resources but have opposite characteristics. Prefill is compute-bound — it monopolizes the GPU for seconds on long prompts. Decode is memory-bound — it needs a small, steady flow of GPU bandwidth. Running them together means each interferes with the other's optimal operating point.

NVIDIA Dynamo, announced at GTC March 2025, decouples them. Dynamo is an orchestration layer that sits above inference engines and manages a fleet of workers: dedicated prefill workers (compute-optimized, can take large batches of prompts), dedicated decode workers (memory-optimized, keep KV cache resident for active sequences), and a KV transfer layer that ships the computed KV cache from prefill to decode workers after the forward pass.

The implication is that prefill and decode can now run on different hardware. A prefill worker on a tightly-packed GB200 with high compute density. Decode workers on instances with high HBM per GPU. The fleet is no longer a single configuration tradeoff — you allocate hardware to the phase that matches its requirements.

Dynamo also handles fleet-level scheduling: KV cache routing (send each request's decode to the worker holding that session's KV blocks), dynamic disaggregation thresholds (at low load, run prefill and decode on the same worker; at high load, split them), and elastic scaling across workers.

The notebook (https://github.com/elizabetht/100-days-of-inference/tree/main/day12/nvidia-dynamo.ipynb) implements a simplified Dynamo architecture with a router, simulates the performance difference between disaggregated and co-located serving under varying request arrival rates.

#LLM #Inference #NVIDIADynamo #DisaggregatedServing #PrefillDecode #DeepLearning #AI #MLEngineering #100DaysOfInference
