# Day 11 — TensorRT-LLM: Compilation and the Plugin System

vLLM and SGLang are general-purpose — they support many models and hardware configurations. Generality costs performance. TensorRT-LLM takes the opposite approach: narrow focus on NVIDIA GPUs, manual kernel selection, and access to closed-source NVIDIA-internal kernels unavailable elsewhere. Harder to use, best-in-class numbers.

TensorRT-LLM V1 (the modern version) is built directly on PyTorch rather than the original TRT graph compiler. The compilation advantage comes from three sources: static shape specialization (knowing batch size, sequence length, and target GPU lets you select kernels optimal for those exact dimensions), kernel auto-selection (choosing the best GEMM implementation for each matrix size), and kernel fusion (identifying and merging operation sequences that can be expressed as a single kernel). V0 ran TRT's graph optimization; V1 achieves similar results with more flexibility.

The plugin system is how NVIDIA integrates proprietary kernels without open-sourcing them. Flash Attention, FP8 GEMM on Hopper Tensor Cores, fused QKV projection, fused SwiGLU — these are all plugins. Some are open source; others are binary-only. The result: TRT-LLM on H100 with FP8 achieves throughput that neither vLLM nor SGLang can match on the same hardware, simply because it has access to kernels they don't.

The deployment tradeoff: TRT-LLM requires a separate compilation step per model per GPU architecture. Changing batch size range or sequence length means recompilation. This is acceptable for fixed production deployments but painful for research or multi-GPU heterogeneous fleets.

The notebook ([tensorrt-llm.ipynb](https://github.com/elizabetht/100-days-of-inference/tree/main/day11/tensorrt-llm.ipynb)) demonstrates compilation speedup using torch.compile as a proxy, models the plugin system architecture, and visualizes the TRT-LLM layer stack from user API down to CUDA runtime.

#LLM #Inference #TensorRTLLM #NVIDIA #Compilation #KernelOptimization #DeepLearning #AI #MLEngineering #100DaysOfInference
