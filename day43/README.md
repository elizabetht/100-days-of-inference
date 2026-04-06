# Day 43

**Topic:** PyTorch Custom Op with CUDA Backend
**Date:** 2026-04-06
**Layer:** Implementation

## What I explored
Implemented a SiLU custom op using torch.library, registered CPU and CUDA dispatch keys, and added abstract implementation for torch.compile compatibility.

## Key insight
The abstract_impl (shape/dtype inference) is the key to making custom ops work with torch.compile — without it, compile falls back to eager for that op.

## Code / experiment
Notebook: [`pytorch-custom-op.ipynb`](./pytorch-custom-op.ipynb)
Key demo: Custom op registration + torch.compile integration

## References
- *Inference Engineering* (Philip Kiely, Baseten Books 2026)
