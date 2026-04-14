# Day 08 — PyTorch Model Formats, ONNX, and TensorRT

Day 06 covered how the ops:byte ratio reveals whether a workload is memory-bound or compute-bound. Day 07 showed how CUDA (Compute Unified Device Architecture) kernel fusion reduces unnecessary round-trips to HBM (High Bandwidth Memory). This picks up at the next abstraction layer — the frameworks and file formats that make it possible to ship optimized models without hand-writing CUDA for every deployment. The inference stack layers by increasing abstraction: CUDA, then frameworks and formats, then inference engines, then orchestration.

Model serialization format determines portability, load speed, and what runtimes can optimize the result. PyTorch's `.pt` format pickles the full Python object graph — flexible but fragile across versions and not portable beyond Python. SafeTensors, created by Hugging Face, replaces it with a flat, memory-mappable format that holds only tensor data, no executable code. A pickle file can execute arbitrary Python on load; SafeTensors cannot. The speed difference shows at scale — for a 7B model, SafeTensors loads in ~2-3 seconds vs 10+ for pickle, because the OS maps pages directly to GPU (Graphics Processing Unit)-accessible memory.

ONNX (Open Neural Network Exchange) goes further than SafeTensors by capturing both the weights and the computation graph. Where SafeTensors stores just the numbers, ONNX stores the full mathematical recipe — every matrix multiply, every activation function — in a standardized operator set that any compatible runtime can execute. Export once from PyTorch, run on ONNX Runtime (ORT), TensorRT (TRT), or any other backend that implements the ONNX spec.

TRT takes an ONNX graph and compiles it into an optimized engine for a specific GPU. At compile time it performs layer fusion, precision selection (FP16/INT8 where accuracy allows), and kernel auto-tuning — benchmarking multiple kernel implementations for each tensor shape and keeping the fastest. The resulting engine is hardware-specific: a TRT engine compiled on A100 will not run on H100. Portability for performance — that's the tradeoff.

At deployment: SafeTensors served via vLLM or SGLang covers most production use cases. For NVIDIA-only clusters with fixed shapes — compile once to TRT, cache the engine. For research or multi-cloud — ONNX with ORT backend or native PyTorch.

The notebook ([pytorch-model-formats.ipynb](https://github.com/elizabetht/100-days-of-inference/tree/main/day08/pytorch-model-formats.ipynb)) builds a 16M-parameter transformer block, saves it in `.pt`, state dict, and SafeTensors formats, exports to ONNX, benchmarks `torch.compile`, and compares load times and file sizes across all formats.

#LLM #Inference #ONNX #TensorRT #SafeTensors #PyTorch #ModelFormats #DeepLearning #AI #MLEngineering #100DaysOfInference
