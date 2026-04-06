# Day 10

**Topic:** NVIDIA Dynamo: Disaggregated Serving
**Date:** 2026-04-06
**Layer:** Runtime

## What I explored
Studied disaggregated prefill/decode serving as implemented by NVIDIA Dynamo. Quantified the computational difference between prefill (compute-bound, O(seq²)) and decode (memory-bandwidth-bound). Measured KV transfer latency across different interconnects to determine when disaggregation is viable.

## Key insight
Disaggregation only helps if KV transfer latency < prefill savings from compute specialization. NVLink at 900 GB/s makes this practical; PCIe at 32 GB/s makes it marginal for prompts under 1K tokens.

## Code / experiment
Notebook: [`nvidia-dynamo-disaggregation.ipynb`](./nvidia-dynamo-disaggregation.ipynb)
Key demo: Disaggregated serving timeline + KV transfer latency across interconnect types

## References
- *Inference Engineering* Ch 4.4 (Philip Kiely, Baseten Books 2026)
- NVIDIA Dynamo documentation
- Zhong et al. (2024), "DistServe"
