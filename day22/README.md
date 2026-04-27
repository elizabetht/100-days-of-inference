# Day 22 — Containerization: Docker and NVIDIA NIMs

Inference deployments fail in production because the serving environment doesn't match development. The CUDA version is wrong. The driver doesn't support FP8. The model weights are in the wrong format for the runtime. Containers fix this by packaging the CUDA runtime, framework, model artifacts, and configuration into a single reproducible unit that lands identically on every node.

A production vLLM Dockerfile builds from the official `nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04` base image, pins vLLM and FastAPI versions, and either bakes weights into the image or mounts them from a volume at runtime. The NVIDIA Container Toolkit passes GPU access into the container without exposing the host driver. The container starts, loads weights into GPU High-Bandwidth Memory (HBM), and begins serving on the OpenAI-compatible port. A `HEALTHCHECK` directive with a 120-second `start-period` gives the model time to load before the orchestrator marks the container unhealthy.

NVIDIA NIM (NVIDIA Inference Microservice) goes further: pre-built containers with TensorRT-LLM (TRT-LLM) engines already compiled for specific GPU families, an OpenAI-compatible REST API at `/v1/chat/completions`, Prometheus metrics at `/metrics`, liveness and readiness probes at `/health/live` and `/health/ready`, and NVIDIA Telemetry integration. The tradeoff is rigidity — NIMs are opinionated about the model, GPU, and API surface. The gain is skipping the hours of TRT-LLM engine compilation that custom containers require on every change.

The container networking model for production: multiple model containers behind an Nginx or Envoy load balancer, each exposing the same port, each scraped by Prometheus. Health checks drive the load balancer — unhealthy containers are removed from rotation before they serve a single broken request.

An inference container hits 10–15 GB before weights load — CUDA alone is 3 GB, PyTorch + CUDA wheels another 5.5 GB. Layer caching keeps redeploys cheap: when only application code changes, just the top ~5 MB ships per node. The NIM-vs-custom split is clean — popular models on supported NVIDIA GPUs with tight deadlines favor NIM; custom CUDA kernels, non-standard batching, or unsupported models favor a custom Dockerfile.

Walkthrough with a production vLLM Dockerfile, NIM container anatomy, an Nginx load-balancer config, and a docker-compose for the home-lab cluster: [`containerization-docker-nim.ipynb`](./containerization-docker-nim.ipynb). Next up: Autoscaling — Concurrency, Batching, and Cold Starts.

Inspired by *Inference Engineering* (Philip Kiely, Baseten Books, 2026). Further reading: [NVIDIA NIM documentation](https://docs.nvidia.com/nim/) and [vLLM Docker deployment guide](https://docs.vllm.ai/en/latest/deployment/docker.html).

#LLM #Inference #Docker #NVIDIANIM #vLLM #TensorRTLLM #Containerization #DeepLearning #AI #MLEngineering #100DaysOfInference
