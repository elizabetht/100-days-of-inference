# Day 01 — What Is LLM Inference?

An LLM does not plan a response and then write it. It predicts one token, appends it to the sequence, and predicts the next. This loop is inference.

GPT-2 (124M params, runs on CPU) makes a good teaching model for breaking this apart. Every request follows three stages:
- Tokenization converts text to integer IDs. "Kubernetes" becomes 4 tokens, "the" becomes 1, and that token count directly determines compute cost.
- Forward pass runs those IDs through the model and returns 50,257 probability scores, one per vocabulary entry.
- Decoding picks a single token from that distribution: greedy takes the argmax, temperature scaling flattens or sharpens the probabilities before sampling.

String those stages into a loop and that's autoregressive generation. The catch: at every step, the model re-reads the entire sequence from scratch. Step 1 processes 10 tokens. Step 40 processes 50. Per-token latency grows linearly with sequence length. Plot it and you get a staircase, each bar taller than the last.

That staircase is the problem every inference optimization exists to solve.
- KV caching stores intermediate attention results to avoid recomputation.
- Batching amortizes weight-loading cost across requests.
- Speculative decoding skips steps by predicting multiple tokens at once.

None of them make sense without understanding the naive loop first.

The notebook (https://github.com/elizabetht/100-days-of-inference/blob/main/day01/llm-inference-mechanics.ipynb) is the complete runnable walkthrough.

#LLM #Inference #MachineLearning #GPT2 #Tokenization #DeepLearning #AI #MLEngineering #100DaysOfInference
