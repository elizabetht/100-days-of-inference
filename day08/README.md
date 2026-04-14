# Day 08 — PyTorch Model Formats, ONNX, and TensorRT

Day 06 built the diagnostic: compute the ops:byte ratio, plot the roofline, determine whether the workload is memory-bound or compute-bound. Day 07 introduced the fix: CUDA (Compute Unified Device Architecture) kernels and kernel fusion, the mechanism for eliminating unnecessary HBM (High Bandwidth Memory) round-trips once decode is known to be memory-bound. Today's topic is the next layer up — the frameworks and file formats that make it possible to deploy those kernels without hand-writing CUDA for every model. The inference software stack is deliberately layered by increasing abstraction: CUDA → frameworks and formats → inference engines → orchestration. Each layer builds on the one below — ops:byte reveals the bottleneck, kernel fusion attacks it, and the right format and framework make the result deployable.

Model serialization format determines portability, load speed, and what runtimes can optimize it. PyTorch's `.pt` format pickles the full Python object graph — flexible but fragile across versions and not portable beyond Python. SafeTensors fixes the safety issue with a flat, memory-mappable format that avoids arbitrary code execution on load. ONNX (Open Neural Network Exchange) is the interchange format: a cross-framework computation graph that TensorRT (TRT), ONNX Runtime, and other runtimes can ingest directly.

TRT takes an ONNX graph and compiles it into an optimized engine for a specific GPU (Graphics Processing Unit). At compile time it performs layer fusion, precision selection (FP16/INT8 where accuracy allows), and kernel auto-tuning — benchmarking multiple implementations for each tensor shape and keeping the fastest. The resulting engine is hardware-specific. A TRT engine compiled on A100 will not run on H100. That's the tradeoff: portability for performance.

Load time differences compound at scale. Loading safetensors with memory-mapping avoids an intermediate CPU copy — the OS can map pages directly to GPU-accessible memory. TRT engines skip graph parsing entirely; they're already compiled execution plans. For a 7B model, safetensors loads in ~2-3 seconds vs 10+ for full pickle format.

The format choice at deployment: weights on disk as safetensors, served via vLLM or SGLang. For NVIDIA-only production clusters with fixed batch sizes and sequence lengths — compile once to TRT, cache the engine, serve the engine. For research or multi-cloud: ONNX with ORT (ONNX Runtime) backend or native PyTorch.

The notebook ([pytorch-model-formats.ipynb](https://github.com/elizabetht/100-days-of-inference/tree/main/day08/pytorch-model-formats.ipynb)) builds a 16M-parameter transformer block, saves it in `.pt`, state dict, and SafeTensors formats, exports to ONNX, benchmarks `torch.compile`, and compares load times and file sizes across all formats.

#LLM #Inference #ONNX #TensorRT #SafeTensors #PyTorch #ModelFormats #DeepLearning #AI #MLEngineering #100DaysOfInference
