# Day 56

**Topic:** Priority Request Queue with Batch Formation
**Date:** 2026-04-06
**Layer:** Production

## What I explored
Built a priority queue that separates interactive (priority=0) from batch (priority=1) requests, forming batches that fill slots by priority order.

## Key insight
Priority queuing with batch formation ensures interactive latency SLOs are met even when the server is saturated with batch traffic — without it, batch jobs starve interactive users.

## Code / experiment
Notebook: [`priority-queue-batch.ipynb`](./priority-queue-batch.ipynb)
Key demo: Priority batch formation with high vs low priority slot allocation

## References
- *Inference Engineering* (Philip Kiely, Baseten Books 2026)
