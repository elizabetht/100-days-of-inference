# Day 50

**Topic:** NVIDIA Dynamo: Disaggregated Prefill Experiment
**Date:** 2026-04-06
**Layer:** Implementation

## What I explored
Simulated disaggregated prefill/decode with KV transfer latency modeling. Computed break-even prompt length and showed when disaggregation saves total latency.

## Key insight
Disaggregation is most beneficial for long prompts: at seq=8192, prefill takes 200ms while KV transfer via NVLink takes <1ms — a clear win. At seq=128, the 10ms KV transfer overhead makes it marginal.

## Code / experiment
Notebook: [`dynamo-disaggregated-prefill.ipynb`](./dynamo-disaggregated-prefill.ipynb)
Key demo: Disaggregated vs coupled serving time breakdown + break-even analysis

## References
- *Inference Engineering* (Philip Kiely, Baseten Books 2026)
