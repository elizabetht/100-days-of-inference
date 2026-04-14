# Day 09 — vLLM: PagedAttention and Continuous Batching

Naive LLM (Large Language Model) serving has two expensive problems. First, the KV (Key-Value) cache for each request needs contiguous memory, but output length isn't known in advance. Pre-allocate for the maximum sequence length and most of it goes unused. Allocate lazily and memory fragments. Second, static batching waits for every request in a batch to finish before starting the next one — short requests hold the GPU (Graphics Processing Unit) idle while long ones complete.

vLLM, the most widely adopted open-source inference engine, solves both. PagedAttention borrows the idea of virtual memory paging from operating systems. Instead of one contiguous KV cache allocation per request, GPU memory is divided into fixed-size blocks. A block table maps each request's logical token positions to physical block addresses, so the KV cache can be non-contiguous. Memory waste drops from O(max_seq_len) to O(1 block) of internal fragmentation per request. In practice, vLLM runs at 80-90% GPU memory utilization where naive allocators top out around 40-60%.

Continuous batching is the throughput fix. Rather than processing a static batch and waiting for it to drain, the scheduler slots new requests into the active batch as soon as any request finishes a generation step. Under load, GPU utilization goes from ~50% with static batching to 90%+.

The KV cache size per token is: 2 x n_layers x n_kv_heads x d_head x bytes_per_element. LLaMA 3 8B uses GQA (Grouped-Query Attention) with 8 KV heads — ~512 KB per token, significantly cheaper than 32-head MHA (Multi-Head Attention). This is exactly why GQA became the standard architecture for production models.

The notebook ([vllm-paged-attention.ipynb](https://github.com/elizabetht/100-days-of-inference/tree/main/day09/vllm-paged-attention.ipynb)) implements a block manager from scratch, simulates fragmentation under different allocation strategies, and models throughput improvement from continuous batching vs static batching.

#LLM #Inference #vLLM #PagedAttention #ContinuousBatching #KVCache #DeepLearning #AI #MLEngineering #100DaysOfInference
