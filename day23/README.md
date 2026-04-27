# Day 23 — Autoscaling: Concurrency, Batching, and Cold Starts

Production traffic varies 10x or more across a day and can spike 5x in minutes. Static provisioning either burns money in the troughs or breaks during the peaks. Autoscaling adjusts the live replica count to match traffic — but for Large Language Model (LLM) inference, the policy levers are unlike stateless web serving.

Autoscaling policy comes down to five parameters: minimum replicas, maximum replicas, the autoscaling window, the scale-down delay, and a concurrency target per replica. Two signals drive decisions — traffic (request rate, queue depth) leads, while Graphics Processing Unit (GPU) utilization lags. Scale-down delay prevents flapping when traffic dips briefly; the cost is paying for replicas after traffic has actually left.

Concurrency per replica is bounded by Key-Value (KV) cache memory, not compute. After the model loads, leftover GPU memory divides among the per-token KV footprint of each active request. For Llama-3-8B on an H100 with 80 GB of High-Bandwidth Memory (HBM), 16 GB goes to weights and ~62 GB to KV — at a 2,048-token context that supports roughly 90 concurrent requests. Throughput is HBM-bandwidth-bound during decode and flattens once compute caps in, so batching past that point buys nothing.

Cold start is the central autoscaling constraint. For a 70B FP16 model, baseline cold start is ~15 minutes — GPU procurement, a 12 GB image pull, a 140 GB weight load, and engine startup. Model-weight loading dominates. A local Non-Volatile Memory Express (NVMe) weight cache combined with INT4 quantization (4× smaller weights) brings the total to under a minute.

The deployment strategy lives on a frontier between cold-start exposure and idle GPU cost. Always-on keeps one replica running around the clock — no cold start, but the H100 ($4.50/hr) bills during idle hours too. Warm pools double idle cost for instant scale-up. Predictive scale-up is cheap when traffic is forecastable. Scale-to-zero with a cached model drops idle cost to ~$0.10/hr at the price of ~90s first-request latency. Serverless drops idle cost to zero and pays the worst cold start. The right answer follows traffic shape, not a universal best-practice.

Walkthrough with a 24-hour autoscaling simulation, the KV-cache concurrency formula, the cold-start stage breakdown, and the strategy comparison: [`autoscaling-concurrency-cold-starts.ipynb`](./autoscaling-concurrency-cold-starts.ipynb). Next up: Routing, Load Balancing, and Queueing.

Inspired by *Inference Engineering* (Philip Kiely, Baseten Books, 2026). Further reading: [KEDA event-driven autoscaling](https://keda.sh/) and [vLLM serving documentation](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html).

#LLM #Inference #Autoscaling #ColdStarts #KVCache #GPUInfrastructure #DeepLearning #AI #MLEngineering #100DaysOfInference
