# Day 27

**Topic:** Client Code: Streaming, Async & Protocol Support
**Date:** 2026-04-06
**Layer:** Tooling

## What I explored
Built a complete inference client toolkit: SSE streaming client with TTFT/TPOT measurement, async batch client with bounded concurrency using asyncio.Semaphore, and retry logic with exponential backoff. Compared REST, gRPC, and WebSocket protocol tradeoffs.

## Key insight
asyncio.Semaphore is the key primitive for production batch clients: it bounds concurrency to prevent overwhelming the server while maximizing throughput from the client side.

## Code / experiment
Notebook: [`client-streaming-async.ipynb`](./client-streaming-async.ipynb)
Key demo: Async concurrency sweep showing throughput vs latency tradeoff + SSE metrics

## References
- *Inference Engineering* Ch 7.5 (Philip Kiely, Baseten Books 2026)
- OpenAI API documentation
- asyncio documentation
