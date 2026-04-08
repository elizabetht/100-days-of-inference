# Day 02 — What's Inside a Model

A model file is not a program. It's a container of numbered arrays.

GPT-2's weight file is 523 MB. Inside: a JSON header mapping tensor names to byte offsets, followed by raw floats packed end to end. The entire thing parses with Python's struct module and numpy. No ML library needed.

148 tensors. 124M learned parameters. The architecture repeats: two embedding tables up front, 12 identical transformer blocks, one final layer norm. Every block has the same structure: attention projections, FFN up/down projections, two layer norms.

The parameter distribution is lopsided. FFN holds 41%. Attention holds 30%. Embeddings hold 28%. Layer norms hold effectively 0%. When quantization or pruning comes up, FFN is where the weight lives.

The front door to all of this is tokenization. GPT-2 uses Byte Pair Encoding: start with individual bytes, merge the most frequent adjacent pair, repeat 50,000 times. At encoding time, the algorithm scans for adjacent pairs, picks the lowest-ranked merge, applies it, repeats. "the" becomes 1 token. "Kubernetes" becomes 4 subword pieces. Deterministic. No neural network involved.

Token count determines forward passes. Fewer tokens = lower latency = lower cost. This is why newer models push to 128K+ vocabularies — more merge rules, better compression.

Runnable notebooks:
- What's inside a model: https://github.com/elizabetht/100-days-of-inference/blob/main/day02/01-whats-inside-a-model.ipynb
- Tokenization from scratch: https://github.com/elizabetht/100-days-of-inference/blob/main/day02/02-tokenization.ipynb

#LLM #Inference #SafeTensors #GPT2 #Tokenization #BPE #DeepLearning #AI #MLEngineering #100DaysOfInference
