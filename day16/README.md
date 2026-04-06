# Day 16

**Topic:** Disaggregation: Prefill/Decode Split
**Date:** 2026-04-06
**Layer:** Runtime

## What I explored
Built an analytical inference time model for prefill and decode phases and optimized the worker ratio for different traffic patterns. Modeled TTFT and TPOT SLOs under queuing theory and showed how optimal allocation changes with input:output length ratio.

## Key insight
The optimal prefill:decode ratio is not 50:50 — it depends entirely on the input/output length mix. Short prompts with long outputs need 6-7x more decode workers than prefill workers.

## Code / experiment
Notebook: [`prefill-decode-disaggregation.ipynb`](./prefill-decode-disaggregation.ipynb)
Key demo: Optimal worker allocation curves for different prompt length distributions

## References
- *Inference Engineering* Ch 5.5 (Philip Kiely, Baseten Books 2026)
- Patel et al. (2023), "Splitwise"
- Zhong et al. (2024), "DistServe"
