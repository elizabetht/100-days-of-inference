# Day 39

**Topic:** Simulate Tensor Parallelism: Split MatMul Across N Workers
**Date:** 2026-04-06
**Layer:** Implementation

## What I explored
Built column-parallel and row-parallel matmul simulations for N workers. Modeled AllReduce overhead vs compute speedup, showing that TP efficiency is dominated by interconnect bandwidth at small batch sizes.

## Key insight
Row-parallel requires AllReduce (sum N partial outputs); column-parallel requires all-gather (concatenate N partial outputs). Megatron-LM pairs them to use exactly 1 AllReduce per transformer block.

## Code / experiment
Notebook: [`tensor-parallelism-simulation.ipynb`](./tensor-parallelism-simulation.ipynb)
Key demo: TP matmul correctness verification + communication overhead model

## References
- *Inference Engineering* (Philip Kiely, Baseten Books 2026)
